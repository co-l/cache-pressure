#!/usr/bin/env python3
"""Performance benchmark: prefill and generation speed vs context length.

Grows a single context from 0 toward the model's max context in full
increments (default 10k tokens of true lorem ipsum). Each step generates a
fixed output budget (default 256 tokens) and replays the reply into the
next request, so the prefix stays stable — including the responses — the
same organic growth agent-sim simulates, but sequential (concurrency 1)
and measured.

Per-step metrics:
  pp — incremental prefill speed: (prompt - cached) / ttft. The first
       step is a full cold prefill; later steps prefill only the new
       increment, so the curve is pp speed at each context level.
  tg — generation speed: completion / (wall - ttft).

Run:
  uvx --from . perf-sim --base-url http://my-box:8000/v1 --output run.json
  uvx --from . perf-sim --base-url http://my-box:8000/v1 --salt 42  # A/B
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

from cache_pressure.core import resolve_model
from cache_pressure.agent_sim import (FALLBACK_REPLY, SYSTEM, chat_stream,
                                      fetch_max_context, fmt_elapsed,
                                      fmt_ktok)

DEFAULT_URL = "http://localhost:8000/v1"
DEFAULT_STEP_TOKENS = 10000
DEFAULT_OUTPUT_TOKENS = 256
OVERHEAD = 128

LOREM_SENTENCES = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.",
    "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum.",
    "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia.",
    "Nulla facilisi, curabitur gravida arcu augue, varius quis massa.",
    "Vestibulum ante ipsum primis in faucibus orci luctus et ultrices.",
    "Maecenas sed ante dolor, euismod et tellus ac convallis nisl.",
    "Cras ultricies ligula sed magna dictum porta, etiam dignissim.",
)


def make_lorem(chars, seed=0):
    """Deterministic true-lorem-ipsum text, exactly chars characters.

    Seeded random sentence order: two seeds share no sentence order, so
    per-step chunks never ride on each other's cached content. The same
    seed always yields the same text, so two runs replay identical
    contexts for a fair A/B.
    """
    rng = random.Random(seed)
    n = len(LOREM_SENTENCES)
    pool = list(range(n))
    rng.shuffle(pool)
    parts = []
    total = 0
    i = 0
    while total < chars:
        parts.append(LOREM_SENTENCES[pool[i % n]])
        total += len(parts[-1]) + 1
        i += 1
        if i % n == 0:
            rng.shuffle(pool)
    return " ".join(parts)[:chars]


def build_steps(steps, step_tokens, salt):
    """Per-step lorem chunks (~4 chars/token target), unique per step."""
    return [make_lorem(int(step_tokens / 0.25), seed=salt + i * 1000)
            for i in range(steps)]


def plan_steps(step_tokens, output_tokens, max_context):
    """Number of full steps that fit under max context.

    The prompt of step i (1-based) is ~OVERHEAD + i*step_tokens +
    (i-1)*output_tokens (replies are replayed). Returns the largest i that
    fits, 0 when even one full step does not.
    """
    if max_context < OVERHEAD + step_tokens:
        return 0
    growth = step_tokens + output_tokens
    return max(1, (max_context - OVERHEAD + output_tokens) // growth)


def pp_speed(prompt, cached, ttft):
    """Incremental prefill speed in tokens/s: uncached prompt / ttft.

    A cold prefill (nothing cached) is the full prompt; if the engine
    reports more cached tokens than the prompt, fall back to the full
    prompt. None when the inputs are missing.
    """
    if not prompt or not ttft:
        return None
    uncached = prompt - (cached or 0)
    if uncached <= 0:
        uncached = prompt
    return uncached / ttft


def tg_speed(completion, wall, ttft):
    """Generation speed in tokens/s: completion / (wall - ttft)."""
    if not completion or not wall or not ttft or wall <= ttft:
        return None
    return completion / (wall - ttft)


def _fmt_row(step, r):
    ctx = "–" if r["prompt"] is None else fmt_ktok(r["prompt"])
    ttft = "–" if r["ttft"] is None else f"{r['ttft']:.2f}s"
    pp = "–" if r["pp"] is None else f"{r['pp']:,.0f}"
    tg = "–" if r["tg"] is None else f"{r['tg']:,.0f}"
    out = r["completion"] if r["completion"] is not None else "–"
    return (f"  step {step + 1:>2}  ctx={ctx:>7}  ttft={ttft:>7}  "
            f"pp={pp:>8} tps  tg={tg:>6} tps  out={out}  "
            f"err={r['error'] or '–'}")


def run_perf(base_url, model, api_key, timeout, chunks, output_tokens,
             thinking, records, run=0):
    """Sequential run: one step = new lorem chunk + output_tokens generated,
    reply replayed into the next request (stable prefix, responses
    included). Records one row per step; stops on the first error."""
    url = base_url.rstrip("/") + "/chat/completions"
    messages = [{"role": "system", "content": SYSTEM}]
    for step, chunk in enumerate(chunks):
        messages.append({"role": "user", "content": chunk})
        res = chat_stream(url, model, messages, output_tokens, timeout,
                          api_key, thinking)
        record = {
            "step": step,
            "run": run,
            "prompt": res["prompt"],
            "cached": res["cached"],
            "cached_source": res["cached_source"],
            "ttft": res["ttft"],
            "wall": res["wall"],
            "completion": res["completion"],
            "pp": pp_speed(res["prompt"], res["cached"], res["ttft"]),
            "tg": tg_speed(res["completion"], res["wall"], res["ttft"]),
            "error": res["error"],
        }
        records.append(record)
        print(_fmt_row(step, record), flush=True)
        if res["error"]:
            break
        messages.append({"role": "assistant",
                         "content": res["content"] or FALLBACK_REPLY})
    return records


def fmt_level_table(records):
    """One row per context level, averaged across runs, as plain lines."""
    levels = {}
    for r in records:
        if r["error"] is None:
            levels.setdefault(r["step"], []).append(r)
    lines = [f"  {'level':>5}  {'ctx':>8}  {'pp':>10}  {'tg':>10}"]
    for i in sorted(levels):
        rs = levels[i]

        def _avg(key):
            vals = [r[key] for r in rs if r[key] is not None]
            return sum(vals) / len(vals) if vals else None

        ctx = _avg("prompt")
        pp = _avg("pp")
        tg = _avg("tg")
        lines.append(
            f"  {i + 1:>5}  "
            f"{fmt_ktok(int(ctx)) if ctx is not None else '–':>8}  "
            f"{(f'{pp:,.0f}' if pp is not None else '–'):>10}  "
            f"{(f'{tg:,.0f}' if tg is not None else '–'):>10}")
    return lines


def summarise(records, runs, elapsed=None):
    """Per-level arrays from the completed steps, averaged across runs."""
    ok = [r for r in records if r["error"] is None]
    levels = {}
    for r in ok:
        levels.setdefault(r["step"], []).append(r)

    def _avg(key):
        out = []
        for i in sorted(levels):
            vals = [r[key] for r in levels[i] if r[key] is not None]
            out.append(sum(vals) / len(vals) if vals else None)
        return out

    return {
        "runs": runs,
        "steps": len(records),
        "completed": len(ok),
        "context": _avg("prompt"),
        "pp": _avg("pp"),
        "tg": _avg("tg"),
        "elapsed": elapsed,
    }


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Prefill/generation speed vs context length")
    p.add_argument("--base-url", default=DEFAULT_URL)
    p.add_argument("--model", default=None,
                   help="Model id (default: auto-detected via GET /models)")
    p.add_argument("--api-key", default=None)
    p.add_argument("--step-tokens", type=int, default=DEFAULT_STEP_TOKENS,
                   help="context growth per step in tokens")
    p.add_argument("--output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS,
                   help="generation budget per step (replayed as the reply)")
    p.add_argument("--max-context", type=int, default=None,
                   help="model max context (default: auto from GET /models)")
    p.add_argument("--runs", type=int, default=3,
                   help="runs to average (same salt; fresh conversation "
                        "per run, per-level pp/tg averaged in the summary)")
    p.add_argument("--timeout", type=int, default=1200)
    p.add_argument("--salt", type=int, default=None,
                   help="run salt (default: time-based, fresh contexts)")
    p.add_argument("--thinking", action="store_true",
                   help="enable reasoning/thinking (default: off — "
                        "reasoning_effort none, so replies are plain text "
                        "that replay verbatim into the next request)")
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)

    if args.step_tokens < 1:
        p.error("--step-tokens must be >= 1")
    if args.output_tokens < 1:
        p.error("--output-tokens must be >= 1")
    if args.runs < 1:
        p.error("--runs must be >= 1")

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
            print("  Could not auto-detect max-context (GET /models "
                  "failed); pass --max-context")
            sys.exit(1)
        print(f"  Max context: {max_context:,} tokens (auto-detected)")

    steps = plan_steps(args.step_tokens, args.output_tokens, max_context)
    if steps == 0:
        print(f"  No full {args.step_tokens:,}-token step fits under "
              f"max-context ({max_context:,} tokens)")
        if args.max_context is None:
            print("  the detected limit looks small; pass --max-context "
                  "to override it")
        else:
            print("  pass a smaller --step-tokens")
        sys.exit(1)

    salt = args.salt if args.salt is not None else int(time.time())
    chunks = build_steps(steps, args.step_tokens, salt)
    print(f"perf-sim: {args.runs} run(s) x {steps} steps x "
          f"{args.step_tokens:,} tok true lorem "
          f"({args.output_tokens} tok replies replayed), toward max "
          f"{max_context:,} tok, thinking={'on' if args.thinking else 'off'}, "
          f"salt {salt}")

    t0 = time.monotonic()
    records = []
    for run in range(args.runs):
        if args.runs > 1:
            print(f"── run {run + 1}/{args.runs} ──", flush=True)
        run_perf(base_url=args.base_url, model=model, api_key=args.api_key,
                 timeout=args.timeout, chunks=chunks,
                 output_tokens=args.output_tokens, thinking=args.thinking,
                 records=records, run=run)
    elapsed = time.monotonic() - t0

    print("\n── speed vs context (averaged per level) ──")
    for line in fmt_level_table(records):
        print(line)

    summary = summarise(records, args.runs, elapsed=elapsed)
    print("\n── summary ──")
    print(f"  run time: {fmt_elapsed(summary['elapsed'])}")
    print(f"  completed steps: {summary['completed']}/{summary['steps']}")
    if summary["completed"] < summary["steps"]:
        print("  WARNING: run stopped early (a step failed); the curve "
              "is incomplete")

    result = {
        "salt": salt,
        "model": model,
        "runs": args.runs,
        "step_tokens": args.step_tokens,
        "output_tokens": args.output_tokens,
        "max_context": max_context,
        "steps": steps,
        "summary": summary,
        "records": records,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"\n  Results written to {args.output}")
    sys.exit(0 if summary["completed"] == summary["steps"] else 1)


if __name__ == "__main__":
    main()
