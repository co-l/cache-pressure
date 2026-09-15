#!/usr/bin/env python3
"""Mid-thinking-abort prefix-retention benchmark (deployment-agnostic).

Reproduces the everyday "stop the model mid-thinking, then re-orient it"
event: a long thinking-mode request is streamed, the client aborts the
connection after a few hundred streamed tokens (keeping exactly the tokens
already sent), then a new request re-sends the conversation with the
captured partial assistant turn (thinking replayed via reasoning_content)
plus a new user instruction, max_tokens 1 — a prefill probe: one token of generation so the engine actually runs the prefill (ninfer treats max_tokens 0 as a no-op that never touches the KV cache).

The headline metric is the re-prompt's reuse: a healthy engine reuses the
whole processed prefix (context + captured thinking); a full re-prefill
(root path, computed ≈ prompt) is exactly the "massive cache miss" this
bench exists to catch. Cached tokens come from
usage.prompt_tokens_details.cached_tokens (ninfer, standard vLLM),
falling back to llama.cpp-compatible timings.cache_n; with --ninfer-log
the records are re-annotated with the ground truth from ninfer's
request log (prefix_cache_hit_tokens, computed_prefill_tokens,
prefix_reuse_path, which the API cannot distinguish).

Run:
  uvx --from cache-pressure abort-sim --base-url http://my-box:8000/v1 \
      --context-tokens 40000 --thinking-tokens 500 --output run.json
  uvx --from cache-pressure abort-sim --base-url http://my-box:8000/v1 \
      --salt 42                                    # deterministic A/B
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

from cache_pressure.agent_sim import fetch_max_context
from cache_pressure.core import make_body, resolve_model
from cache_pressure.ninfer_log import classify_reuse, parse_ninfer_log

SYSTEM = "You are a helpful assistant."
DEFAULT_URL = "http://localhost:8000/v1"
DEFAULT_CONTEXT_TOKENS = 40000
DEFAULT_THINKING_TOKENS = 500
DEFAULT_MAX_TOKENS = 32768
DEFAULT_REPROMPT_MAX_TOKENS = 1
DEFAULT_RUNS = 2
DEFAULT_MIN_REUSE = 0.95
TOKENS_PER_CHAR = 0.25

PROBLEM = (
    "Twelve coins look identical. Exactly one of them is counterfeit: it is "
    "either lighter or heavier than a genuine coin, but you do not know "
    "which. You have a balance scale. Show a strategy that identifies the "
    "counterfeit coin and determines whether it is lighter or heavier in "
    "exactly three weighings. Justify why the strategy always works, and "
    "argue why no strategy can succeed with only two weighings. Think "
    "through the full decision tree carefully before you answer."
)

REORIENT = (
    "Hold on — let's re-prioritize before you finish. Set that decision tree "
    "aside for now: in one or two sentences, what is the most important "
    "insight from your reasoning so far? Then sketch a completely different "
    "approach to the same problem that does not use the strategy you have "
    "been developing."
)


def build_prime_messages(salt, context_tokens):
    """[system, user(~context_tokens lorem + thinking-inducing problem)].

    Unique per salt so runs never ride on each other's cached content; the
    same salt always yields the same planned input (deterministic A/B).
    """
    body = make_body(int(context_tokens / TOKENS_PER_CHAR), salt=salt)
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": body + "\n\n" + PROBLEM},
    ]


def build_reprompt_messages(prime_messages, reasoning, content, reorient,
                            mode):
    """Re-send the conversation with the captured partial assistant turn.

    branch: the new user instruction (the re-orientation) follows the
    partial turn — the everyday "stop and redirect" event.
    continuation: the conversation simply ends on the partial turn.
    """
    assistant = {"role": "assistant", "content": content}
    if reasoning:
        assistant["reasoning_content"] = reasoning
    messages = list(prime_messages) + [assistant]
    if mode == "branch":
        messages.append({"role": "user", "content": reorient})
    return messages


def stream_thinking(url, model, messages, max_tokens, abort_after, timeout,
                    api_key=None, post_fn=None):
    """Stream the prime request and abort the connection mid-thinking.

    Counts one token per non-empty streamed delta (reasoning or content).
    Once `abort_after` tokens are captured, the connection is closed — the
    captured texts are exactly the tokens already sent to the client.
    Returns a record with the capture; `aborted=False` (with `error`) means
    the stream ended before the target, i.e. the scenario did not run.
    """
    if post_fn is None:
        post_fn = requests.post
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "reasoning_effort": "xhigh",
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    t0 = time.monotonic()
    tokens = 0
    reasoning_tokens = 0
    content_tokens = 0
    ttft = None
    reasoning_parts = []
    content_parts = []
    aborted = False
    error = None
    resp = None
    try:
        resp = post_fn(url, json=payload, headers=headers, stream=True,
                       timeout=timeout)
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode().strip() if isinstance(line, bytes) else str(line).strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
            content = delta.get("content") or ""
            if reasoning:
                if ttft is None:
                    ttft = time.monotonic() - t0
                reasoning_parts.append(reasoning)
                reasoning_tokens += 1
                tokens += 1
            if content:
                if ttft is None:
                    ttft = time.monotonic() - t0
                content_parts.append(content)
                content_tokens += 1
                tokens += 1
            if tokens >= abort_after:
                aborted = True
                break
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    finally:
        if resp is not None:
            resp.close()
    if error is None and not aborted:
        error = (f"stream finished before the abort target "
                 f"({tokens} < {abort_after} tokens) — the model stopped "
                 f"thinking early; raise --thinking-tokens or --max-tokens")
    return {
        "aborted": aborted,
        "tokens": tokens,
        "reasoning_tokens": reasoning_tokens,
        "content_tokens": content_tokens,
        "reasoning": "".join(reasoning_parts),
        "content": "".join(content_parts),
        "ttft": ttft,
        "wall": time.monotonic() - t0,
        "error": error,
    }


def send_reprompt(url, model, messages, max_tokens, timeout, api_key=None,
                  post_fn=None):
    """One non-streaming re-prompt (max_tokens may be 0: prefill only).

    Returns prompt/cached with the usage -> timings -> none source
    fallback, plus the wall time (the re-prefill cost the user sees).
    """
    if post_fn is None:
        post_fn = requests.post
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "reasoning_effort": "xhigh",
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    t0 = time.monotonic()
    error = None
    prompt = None
    cached_usage = cached_timings = None
    try:
        resp = post_fn(url, json=payload, headers=headers, timeout=timeout)
        data = resp.json()
        usage = data.get("usage") or {}
        prompt = usage.get("prompt_tokens")
        cached_usage = (usage.get("prompt_tokens_details") or {}) \
            .get("cached_tokens")
        timings = data.get("timings") or {}
        cached_timings = timings.get("cache_n")
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    if cached_usage is not None:
        cached, source = cached_usage, "usage"
    elif cached_timings is not None:
        cached, source = cached_timings, "timings"
    else:
        cached, source = None, "none"
    return {"prompt": prompt, "cached": cached, "cached_source": source,
            "wall": time.monotonic() - t0, "error": error}


def reuse_fraction(cached, prompt):
    """Cached tokens over the re-prompt's own prompt, or None if unknown."""
    if cached is None or not prompt:
        return None
    return cached / prompt


def verdict(rec, min_reuse):
    """A run passes: aborted mid-thinking, no error, reuse >= min_reuse,
    and no root re-prefill when the ground-truth path is known."""
    if not rec.get("aborted"):
        return False
    if rec.get("error"):
        return False
    reuse = rec.get("reuse")
    if reuse is None:
        return False
    if reuse < min_reuse:
        return False
    if rec.get("reuse_path") == "root":
        return False
    return True


def summarise(records, min_reuse):
    total = len(records)
    ok = sum(1 for r in records if r["ok"])
    return {"runs": total, "ok": ok, "all_ok": total > 0 and ok == total,
            "min_reuse": min_reuse}


def annotate_reprompt_records(records, entries, min_reuse,
                              tolerance_ms=15000):
    """Zip run records with request_done log entries.

    Primary match: an entry fits a record when its timestamp is within
    `tolerance_ms` of the record's `reprompt_ts` (epoch ms, recorded when
    the re-prompt response arrived) and the prompt sizes agree when both
    sides report them — robust against concurrent traffic on the same
    server. Records without a timestamp (or unmatched) fall back to the
    trailing-slice zip (idle-server layout). Reuse is recomputed from the
    ground truth (prefix_cache_hit_tokens / prompt) and the verdict
    re-evaluated (a root re-prefill downgrades a high API reuse). Returns
    (annotated, unmatched, extra_entries).
    """
    used = [False] * len(entries)
    matched = {}
    for idx, rec in enumerate(records):
        ts = rec.get("reprompt_ts")
        if ts is None:
            continue
        best = None
        for j, e in enumerate(entries):
            if used[j]:
                continue
            ets = e.get("ts")
            if ets is None:
                continue
            dist = abs(ets - ts)
            if dist > tolerance_ms:
                continue
            if (rec.get("prompt") and e.get("prompt")
                    and e["prompt"] != rec["prompt"]):
                continue
            if best is None or dist < best[0]:
                best = (dist, j)
        if best is not None:
            used[best[1]] = True
            matched[idx] = best[1]

    leftover = [j for j in range(len(entries)) if not used[j]]
    any_entry_ts = any(e.get("ts") is not None for e in entries)
    leftover_recs = [idx for idx in range(len(records))
                     if idx not in matched
                     and (not any_entry_ts
                          or records[idx].get("reprompt_ts") is None)]
    if leftover_recs:
        for rec_idx, ent_idx in zip(leftover_recs,
                                    leftover[-len(leftover_recs):]):
            matched[rec_idx] = ent_idx
            used[ent_idx] = True

    extra = [entries[j] for j in range(len(entries)) if not used[j]]
    out = []
    unmatched = []
    for idx, rec in enumerate(records):
        if idx not in matched:
            unmatched.append(rec)
            continue
        e = entries[matched[idx]]
        prompt = e.get("prompt") or rec.get("prompt")
        reuse = reuse_fraction(e.get("prefix_cache_hit_tokens"), prompt)
        path = e.get("prefix_reuse_path")
        annotated = {
            **rec,
            "reuse": reuse if reuse is not None else rec.get("reuse"),
            "computed": e.get("computed_prefill_tokens"),
            "reuse_path": path,
            "reuse_class": classify_reuse(reuse, path),
            "request_id": e.get("request_id"),
        }
        annotated["ok"] = verdict(annotated, min_reuse)
        out.append(annotated)
    return out, unmatched, extra


def _fmt(n):
    return "–" if n is None else f"{n:,}"


def _fmt_f(x, nd=1):
    return "–" if x is None else f"{x:.{nd}f}"


def run_bench(args, model):
    """Run all scenarios sequentially; return the records list."""
    url = args.base_url.rstrip("/") + "/chat/completions"
    records = []
    for i in range(args.runs):
        salt = args.salt + i * 100_000
        print(f"── run {i + 1}/{args.runs} · salt {salt} ──")
        prime = build_prime_messages(salt, args.context_tokens)
        print(f"  prime     ~{args.context_tokens:,} tok context + problem · "
              f"effort xhigh · budget {args.max_tokens:,}")
        cap = stream_thinking(url, model, prime, args.max_tokens,
                              args.thinking_tokens, args.timeout,
                              args.api_key)
        record = {
            "run": i,
            "salt": salt,
            "aborted": cap["aborted"],
            "tokens": cap["tokens"],
            "reasoning_tokens": cap["reasoning_tokens"],
            "content_tokens": cap["content_tokens"],
            "reasoning_chars": len(cap["reasoning"]),
            "ttft": cap["ttft"],
            "abort_wall": cap["wall"],
            "prompt": None,
            "cached": None,
            "cached_source": "none",
            "reuse": None,
            "reprompt_wall": None,
            "reprompt_ts": None,
            "error": cap["error"],
            "ok": False,
        }
        if not cap["aborted"] or cap["error"]:
            record["ok"] = False
            print(f"  abort     NOT REACHED — {cap['error']}")
            records.append(record)
            continue
        print(f"  abort     after {cap['tokens']} streamed tokens "
              f"({cap['reasoning_tokens']} reasoning / "
              f"{cap['content_tokens']} content) · ttft {_fmt_f(cap['ttft'])}s"
              f" · wall {_fmt_f(cap['wall'])}s")
        messages = build_reprompt_messages(prime, cap["reasoning"],
                                           cap["content"], REORIENT,
                                           args.reprompt_mode)
        rep = send_reprompt(url, model, messages, args.reprompt_max_tokens,
                            args.timeout, args.api_key)
        reuse = reuse_fraction(rep["cached"], rep["prompt"])
        record.update({
            "prompt": rep["prompt"],
            "cached": rep["cached"],
            "cached_source": rep["cached_source"],
            "reuse": reuse,
            "reprompt_wall": rep["wall"],
            "reprompt_ts": int(time.time() * 1000),
            "error": rep["error"] or record["error"],
        })
        record["ok"] = verdict(record, args.min_reuse)
        reuse_pct = "–" if reuse is None else f"{reuse * 100:.1f}%"
        print(f"  re-prompt prompt={_fmt(rep['prompt'])}  "
              f"cached={_fmt(rep['cached'])} ({rep['cached_source']})  "
              f"reuse={reuse_pct}  wall {_fmt_f(rep['wall'])}s")
        records.append(record)
    return records


def print_report(records, summary, min_reuse, extra_note=""):
    print("\n── verdict ──")
    for r in records:
        reuse = r.get("reuse")
        reuse_pct = "–" if reuse is None else f"{reuse * 100:.1f}%"
        if r["ok"]:
            path = f" · path {r['reuse_path']}" if r.get("reuse_path") else ""
            computed = (f" · computed {r['computed']:,}"
                        if r.get("computed") is not None else "")
            detail = (f"reuse {reuse_pct} ≥ {min_reuse * 100:.0f}%"
                      + path + computed)
        else:
            if not r.get("aborted") or r.get("error"):
                detail = r.get("error") or "not aborted"
            else:
                bits = []
                reuse = r.get("reuse")
                if reuse is None:
                    bits.append("no cache signal")
                elif reuse < min_reuse:
                    bits.append(f"reuse {reuse_pct} < {min_reuse * 100:.0f}%")
                if r.get("reuse_path") == "root":
                    bits.append("root re-prefill (full miss)")
                if r.get("computed") is not None:
                    bits.append(f"computed {r['computed']:,} tok")
                detail = ", ".join(bits) or "verdict failed"
        mark = "OK  " if r["ok"] else "FAIL"
        print(f"  [{mark}] run {r['run'] + 1}: {detail}")
    print(f"  runs passed: {summary['ok']}/{summary['runs']}{extra_note}")


def print_ground_truth(records, entries_count, unmatched_count):
    n_miss = sum(1 for r in records if r.get("reuse_class") == "miss")
    n_hit = sum(1 for r in records if r.get("reuse_class") == "hit")
    n_part = sum(1 for r in records if r.get("reuse_class") == "partial")
    print(f"  ninfer request-log: {entries_count} request_done entries, "
          f"{unmatched_count} records without an entry")
    print(f"  ground-truth reuse classes: {n_hit} hit / {n_part} partial "
          f"/ {n_miss} miss")
    for r in records:
        if r.get("reuse_path") is not None:
            hit = ((r.get("prompt") or 0) - r["computed"]
                   if r.get("computed") is not None else None)
            print(f"    run {r.get('run', 0) + 1}: path={r['reuse_path']}  "
                  f"computed={_fmt(r.get('computed'))}  hits={_fmt(hit)}")


def annotate_saved(path, log_path, min_reuse):
    """Re-annotate a saved run's JSON with a request log (box workflow:
    the log lives on the server, the run's output on the workstation)."""
    data = json.loads(Path(path).read_text())
    records = data.get("records") or []
    entries = parse_ninfer_log(log_path)
    records, unmatched, extra = annotate_reprompt_records(
        records, entries, min_reuse)
    print_ground_truth(records, len(entries), len(unmatched))
    summary = summarise(records, min_reuse)
    print_report(records, summary, min_reuse)
    data["records"] = records
    data["summary"] = summary
    Path(path).write_text(json.dumps(data, indent=2))
    print(f"  Annotated results written back to {path}")
    sys.exit(0 if summary["all_ok"] else 1)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Mid-thinking-abort prefix-retention benchmark")
    p.add_argument("--base-url", default=DEFAULT_URL)
    p.add_argument("--model", default=None,
                   help="Model id (default: auto-detected via GET /models)")
    p.add_argument("--api-key", default=None)
    p.add_argument("--context-tokens", type=int,
                   default=DEFAULT_CONTEXT_TOKENS,
                   help="prime context size in tokens (default: 40000)")
    p.add_argument("--thinking-tokens", type=int,
                   default=DEFAULT_THINKING_TOKENS,
                   help="abort after this many streamed tokens "
                        "(default: 500)")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                   help="generation budget of the prime request "
                        "(default: 32768)")
    p.add_argument("--reprompt-max-tokens", type=int,
                   default=DEFAULT_REPROMPT_MAX_TOKENS,
                   help="generation budget of the re-prompt (default: 1 — prefill plus a "
                        "single token; ninfer treats 0 as a no-op that runs no "
                        "prefill at all)")
    p.add_argument("--reprompt-mode", choices=("branch", "continuation"),
                   default="branch",
                   help="branch: new user instruction after the partial "
                        "turn (the re-orientation event); continuation: "
                        "the conversation ends on the partial turn "
                        "(default: branch)")
    p.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                   help="independent runs, fresh salt each (default: 2)")
    p.add_argument("--min-reuse", type=float, default=DEFAULT_MIN_REUSE,
                   help="pass threshold on the re-prompt reuse fraction "
                        "(default: 0.95)")
    p.add_argument("--max-context", type=int, default=None,
                   help="model max context (default: auto from GET /models)")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--salt", type=int, default=None,
                   help="run salt (default: time-based, fresh contexts)")
    p.add_argument("--ninfer-log", default=None,
                   help="path to ninfer's request-log JSONL (--request-log-jsonl); "
                        "re-annotates records with the ground-truth reuse "
                        "(prefix_cache_hit_tokens / computed_prefill_tokens / "
                        "prefix_reuse_path), which the API cannot distinguish")
    p.add_argument("--output", default=None)
    p.add_argument("--annotate", default=None,
                   help="re-annotate a saved run's JSON (--output file) "
                        "against a request log (--ninfer-log) instead of "
                        "running: the log lives on the server, the run on "
                        "the workstation")
    args = p.parse_args(argv)

    if args.annotate:
        if not args.ninfer_log:
            p.error("--annotate requires --ninfer-log")
        annotate_saved(args.annotate, args.ninfer_log, args.min_reuse)

    model = args.model
    if model is None:
        model = resolve_model(args.base_url, args.api_key, args.timeout)
        if model is None:
            print("  Could not auto-detect a model (GET /models failed or "
                  "returned none); pass --model")
            sys.exit(1)
        print(f"  Model: {model} (auto-detected)")

    max_context = args.max_context
    if max_context is None:
        max_context = fetch_max_context(args.base_url, args.api_key,
                                        args.timeout)
        if max_context is None:
            print("  Could not auto-detect max-context (GET /models failed); "
                  "pass --max-context")
            sys.exit(1)
    prime_budget = args.context_tokens + args.max_tokens
    if prime_budget > max_context:
        print(f"  Validation failed: prime context ({args.context_tokens:,} "
              f"tok) + budget ({args.max_tokens:,} tok) exceeds max-context "
              f"({max_context:,} tok)")
        sys.exit(1)

    salt = args.salt if args.salt is not None else int(time.time())
    args.salt = salt
    print(f"abort-sim: {args.runs} run(s) x ~{args.context_tokens:,} tok "
          f"context, abort after {args.thinking_tokens} streamed tokens, "
          f"re-prompt max_tokens {args.reprompt_max_tokens} "
          f"({args.reprompt_mode}), min-reuse {args.min_reuse:.0%}, salt {salt}")

    records = run_bench(args, model)

    if args.ninfer_log:
        entries = parse_ninfer_log(args.ninfer_log)
        records, unmatched, _extra = annotate_reprompt_records(
            records, entries, args.min_reuse)
        print_ground_truth(records, len(entries), len(unmatched))

    summary = summarise(records, args.min_reuse)
    print_report(records, summary, args.min_reuse)

    result = {
        "salt": salt,
        "runs": args.runs,
        "context_tokens": args.context_tokens,
        "thinking_tokens": args.thinking_tokens,
        "max_tokens": args.max_tokens,
        "reprompt_max_tokens": args.reprompt_max_tokens,
        "reprompt_mode": args.reprompt_mode,
        "reasoning_effort": "xhigh",
        "min_reuse": args.min_reuse,
        "model": model,
        "max_context": max_context,
        "summary": summary,
        "records": records,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"  Results written to {args.output}")
    sys.exit(0 if summary["all_ok"] else 1)


if __name__ == "__main__":
    main()
