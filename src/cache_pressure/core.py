#!/usr/bin/env python3
"""KV-cache retention pressure benchmark (deployment-agnostic).

Quantifies how much context a prefix cache actually retains under overflow
pressure — independent of the advertised capacity number, which is an
active-request planning budget, not a ceiling on cacheable content. Works
against any OpenAI-compatible endpoint.

Procedure:
  1. capacity  - the engine's advertised KV size in tokens, passed with
                 --kv-size (required).
  2. calibrate - tokens/char probe (length-dependent; iterates to
                 convergence), then a threshold calibration: one probe
                 context sent cold (miss baseline) and again from cache
                 (hit baseline); the verify threshold is their midpoint.
  3. phase A   - hydrate N unique ~8K-token contexts sequentially,
                 max_tokens 1 each (prefill-only; the full context is
                 committed regardless of output length). Default
                 N = ceil(capacity/8000)+5 so the cache overflows.
  4. phase B   - re-send each context in reverse order (newest first),
                 max_tokens 1, classify hit/miss by TTFT against the
                 calibrated threshold. The first miss (going back in time)
                 is the oldest evicted context = the real retained capacity
                 under pressure. The verify loop stops at that first miss:
                 under LRU everything older is evicted by construction.

  --compare A.json B.json diffs two result files (e.g. fixed vs control
  image) for a before/after verdict.

Run:
  uvx cache-pressure --kv-size 2000000 --output run.json
  uvx cache-pressure --compare fix.json control.json
"""
import argparse
import json
import math
import random
import sys
import time

import requests

LOREM = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "Sphinx of black quartz, judge my vow. "
    "The five boxing wizards jump quickly. "
    "Jazz and swing fans love funky rhythms. "
    "Amazingly few discotheques provide jukeboxes. "
    "Cozy sphinx waves quart jug of bad milk. "
    "A mad boxer shot a quick, gloved jab to the jaw of his dizzy opponent. "
    "We promptly judged antique ivory buckles for the next prize. "
)

DEFAULT_URL = "http://localhost:8000/v1"
DEFAULT_CONTEXT_TOKENS = 8000
DEFAULT_MARGIN = 5
DEFAULT_HIT_THRESHOLD = 1.5
PROBE_CHARS = 2000


def resolve_model(base_url, api_key=None, timeout=10, get_fn=None):
    """Auto-detect the served model via GET /models (OpenAI-compatible).

    Returns the model id, or None when detection fails (endpoint missing,
    error, or empty list). Picks the first served model when several are
    available, with a note to pass --model to disambiguate.
    """
    if get_fn is None:
        def get_fn(url, headers=None, timeout=10):
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = base_url.rstrip("/") + "/models"
    try:
        data = get_fn(url, headers=headers, timeout=timeout)
    except Exception:
        return None
    seen = set()
    ids = []
    if isinstance(data, list):
        items = data
    else:
        items = data.get("data") or data.get("models") or []
    for m in items:
        mid = m.get("id") or m.get("model")
        if mid and mid not in seen:
            seen.add(mid)
            ids.append(mid)
    if not ids:
        return None
    if len(ids) > 1:
        print(f"  Multiple models served ({', '.join(ids)}); using {ids[0]} "
              "(pass --model to disambiguate)")
    return ids[0]


def calibrate_threshold(base_url, model, api_key, timeout, context_tokens,
                        tokens_per_char, request_fn=None):
    """Measure cold-prefill and cache-hit TTFT at the real context size.

    Sends one unique probe context twice: the first send is a cold prefill
    (miss baseline), the second is served from the prefix cache (hit
    baseline). Returns (threshold, cold_ttft, hit_ttft) where threshold is
    their midpoint — a separator that stays valid regardless of engine
    speed — or (None, ...) when calibration is unreliable.

    The probe context uses a run-unique salt so it can never collide with a
    prefix a previous run left cached: a deterministic probe would be
    served from that leftover and read as a fake "hit" on its first send.
    """
    if request_fn is None:
        request_fn = do_request
    url = base_url.rstrip("/") + "/chat/completions"
    salt = random.randrange(100000, 1000000)
    msgs = build_messages(make_body(int(context_tokens / tokens_per_char),
                                    salt=salt))
    cold_ttft, _e2e, _out, _prompt, err1 = request_fn(
        url, model, msgs, 1, timeout, api_key)
    hit_ttft, _e2e, _out, _prompt, err2 = request_fn(
        url, model, msgs, 1, timeout, api_key)
    if err1 or err2:
        print(f"  Threshold calibration failed: probe error "
              f"(cold={err1 or '–'}, hit={err2 or '–'})")
        return None, None, None
    if not cold_ttft or not hit_ttft:
        print("  Threshold calibration failed: no token received on a probe")
        return None, None, None
    if hit_ttft >= cold_ttft:
        print(f"  Threshold calibration failed: hit ({hit_ttft:.3f}s) not "
              f"faster than cold ({cold_ttft:.3f}s) — engine not caching, "
              f"or the probe collided with cached content")
        return None, cold_ttft, hit_ttft
    return (cold_ttft + hit_ttft) / 2.0, cold_ttft, hit_ttft


def make_body(chars, salt=0):
    """Deterministic ~chars-char text, unique per salt.

    Seeded random word order from the lorem vocabulary: two salts share no
    256-token block (different RNG sequences), so contexts never ride on
    each other's cached content. The same salt always yields the same text,
    so the fix and control runs replay identical contexts for a fair A/B.
    """
    rng = random.Random(7919 + salt)
    words = LOREM.split()
    parts = []
    n = 0
    while n < chars:
        w = rng.choice(words)
        parts.append(w)
        n += len(w) + 1
    return " ".join(parts)[:chars]


def build_messages(context):
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": context},
    ]


def do_request(url, model, messages, max_tokens, timeout, api_key=None):
    """Stream one chat completion, return (ttft, e2e, out_tokens, prompt_tokens, error)."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    t0 = time.monotonic()
    first = True
    ttft = 0.0
    output_tokens = 0
    prompt_tokens = None
    error = None
    try:
        resp = requests.post(url, json=payload, headers=headers, stream=True, timeout=timeout)
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            choices = chunk.get("choices", [])
            delta = choices[0].get("delta", {}) if choices else {}
            tok = (delta.get("content", "")
                   or delta.get("reasoning", "")
                   or delta.get("reasoning_content", ""))
            if tok:
                if first:
                    ttft = time.monotonic() - t0
                    first = False
                output_tokens += 1
            usage = chunk.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
    except Exception as e:
        error = str(e)
        if first:
            ttft = time.monotonic() - t0
    return ttft, time.monotonic() - t0, output_tokens, prompt_tokens, error


def _probe(base_url, model, api_key, timeout, chars):
    url = base_url.rstrip("/") + "/chat/completions"
    msgs = build_messages(make_body(chars, salt=9999))
    ttft, e2e, out, prompt, err = do_request(url, model, msgs, 1, timeout, api_key)
    return (prompt, ttft, e2e, out, err)


def calibrate(base_url, model, api_key, timeout, target_tokens):
    """Iteratively measure tokens/char at the real context length.

    Tokens/char is length-dependent for this generator (BPE compresses
    repetitive text better at length: ~0.30 at 2K chars vs ~0.21 at 156K),
    so a single small probe undersizes contexts. Refine by probing at the
    estimated target char length until the size converges (each probe is a
    real prefill, max ~4 iterations). Returns (tokens_per_char, converged
    chars) or (None, None) on probe failure.
    """
    tpc = 0.25
    chars = PROBE_CHARS
    for _ in range(6):
        prompt, *_rest, err = _probe(base_url, model, api_key, timeout, chars)
        if err or not prompt:
            return None, None
        tpc = prompt / chars
        new_chars = int(target_tokens / tpc)
        if abs(new_chars - chars) / chars < 0.01:
            break
        chars = new_chars
    return tpc, chars


def generate_contexts(n, target_tokens, tokens_per_char):
    """Unique contexts sized to ~target_tokens prompt tokens each."""
    chars = int(target_tokens / tokens_per_char)
    return [make_body(chars, salt=i) for i in range(n)]


def fmt_progress(used, capacity):
    return f"[{used // 1000}K/{capacity // 1000}K]"


def hydrate_phase(url, model, contexts, max_tokens, timeout, api_key, capacity):
    rows = []
    used = 0
    for i, ctx in enumerate(contexts):
        ttft, e2e, out, prompt, err = do_request(
            url, model, build_messages(ctx), max_tokens, timeout, api_key)
        pref = (prompt / ttft) if ttft > 0 and prompt else None
        pref_s = f"{pref:5.0f}tok/s" if pref is not None else "     –"
        rows.append({"index": i, "ttft": ttft, "e2e": e2e, "output_tokens": out,
                     "prompt_tokens": prompt, "error": err})
        used += prompt or 0
        print(f"  [{i + 1}/{len(contexts)}] hydrate  ttft={ttft:7.2f}s  "
              f"e2e={e2e:7.2f}s  prompt={prompt or '?':>6}  out={out:>4}  "
              f"prefill={pref_s}  err={err or '–'}  {fmt_progress(used, capacity)}")
    return rows


def verify_phase(url, model, contexts, max_tokens, timeout, api_key, threshold,
                 capacity, request_fn=None):
    """Reverse-order verify, stopping after the first MISS.

    request_fn is injectable for tests (defaults to do_request). A genuine
    MISS (no error, ttft >= threshold) means the oldest surviving context
    was found: everything older was evicted by construction under LRU, so
    the remaining probes are skipped.
    """
    if request_fn is None:
        request_fn = do_request
    rows = []
    used = 0
    order = list(range(len(contexts) - 1, -1, -1))
    for pos, i in enumerate(order):
        ttft, e2e, out, prompt, err = request_fn(
            url, model, build_messages(contexts[i]), max_tokens, timeout, api_key)
        hit = err is None and ttft < threshold
        rows.append({"index": i, "ttft": ttft, "e2e": e2e, "output_tokens": out,
                     "prompt_tokens": prompt, "error": err, "hit": hit})
        verdict = "HIT " if hit else ("MISS" if err is None else "ERR ")
        used += prompt or 0
        print(f"  [{pos + 1}/{len(order)}] verify   ctx#{i:>3}  {verdict}  "
              f"ttft={ttft:7.2f}s  prompt={prompt or '?':>6}  err={err or '–'}  "
              f"{fmt_progress(used, capacity)}")
        if not hit and err is None:
            print(f"  first miss at ctx#{i} — stopping "
                  "(older contexts evicted by construction)")
            break
    return rows


def summarise(capacity, contexts, verify_rows, threshold):
    retained = 0
    retained_tokens = 0
    first_miss = None
    for r in verify_rows:
        if r["hit"]:
            retained += 1
            retained_tokens += r.get("prompt_tokens") or 0
        elif first_miss is None:
            first_miss = r["index"]
    pct = (100.0 * retained_tokens / capacity) if capacity else 0.0
    return {
        "retained": retained,
        "retained_tokens": retained_tokens,
        "retained_pct": round(pct, 2),
        "oldest_evicted_index": first_miss,
        "threshold": threshold,
    }


def run_pressure(args):
    url = args.base_url.rstrip("/") + "/chat/completions"
    capacity = args.kv_size

    model = args.model
    if model is None:
        model = resolve_model(args.base_url, args.api_key, args.timeout)
        if model is None:
            print("  Could not auto-detect a model (GET /models failed or "
                  "returned none); pass --model")
            sys.exit(1)
        print(f"  Model: {model} (auto-detected)")

    tokens_per_char = args.tokens_per_char
    if not args.skip_calibrate:
        tpc, conv_chars = calibrate(args.base_url, model, args.api_key,
                                    args.timeout, args.context_tokens)
        if tpc is not None:
            tokens_per_char = tpc
            print(f"  Calibrated tokens/char: {tokens_per_char:.4f} "
                  f"(converged at {conv_chars:,} chars ≈ {args.context_tokens:,} tok)")
        else:
            print(f"  Calibration failed; falling back to {tokens_per_char} "
                  f"chars/token (contexts will be mis-sized)")

    threshold = args.hit_threshold
    cold_ttft = hit_ttft = None
    if threshold is None:
        threshold, cold_ttft, hit_ttft = calibrate_threshold(
            args.base_url, model, args.api_key, args.timeout,
            args.context_tokens, tokens_per_char)
        if threshold is None:
            threshold = DEFAULT_HIT_THRESHOLD
            print(f"  Threshold calibration failed; using fallback "
                  f"{threshold:.1f}s")
        else:
            print(f"  Threshold calibrated: cold={cold_ttft:.2f}s  "
                  f"hit={hit_ttft:.2f}s  ->  {threshold:.2f}s")

    budget = args.context_tokens + args.max_tokens_hydrate
    n = args.num_contexts
    if n is None:
        n = math.ceil(capacity / budget) + args.margin
    print(f"  Contexts: {n} x ~{args.context_tokens:,} tok prompt "
          f"(budget {budget:,}/ctx, total ~{n * budget:,} vs capacity {capacity:,})")
    print()

    contexts = generate_contexts(n, args.context_tokens, tokens_per_char)
    print(f"── Phase A: hydrate {n} contexts (max_tokens={args.max_tokens_hydrate}) ──")
    hydrate = hydrate_phase(url, model, contexts,
                            args.max_tokens_hydrate, args.timeout, args.api_key,
                            capacity)
    actual_total = sum((r.get("prompt_tokens") or 0) + (r.get("output_tokens") or 0)
                       for r in hydrate if r["error"] is None)
    print(f"  actual cache used: {actual_total:,} tokens "
          f"({100.0 * actual_total / capacity:.1f}% of capacity)")
    if actual_total <= capacity:
        print("  ⚠ WARNING: total did not overflow capacity — this run cannot "
              "measure retention under pressure (bump --num-contexts).")
    print()
    print(f"── Phase B: verify in reverse order (max_tokens={args.max_tokens_verify}, "
          f"hit < {threshold:.2f}s) ──")
    verify = verify_phase(url, model, contexts,
                          args.max_tokens_verify, args.timeout, args.api_key,
                          threshold, capacity)
    print()

    summary = summarise(capacity, contexts, verify, threshold)
    print("── Retention under pressure ──")
    print(f"  capacity:            {capacity:,} tokens")
    print(f"  retained contexts:   {summary['retained']}/{n}")
    print(f"  retained tokens:     {summary['retained_tokens']:,}")
    print(f"  retained % capacity: {summary['retained_pct']}%")
    if summary["oldest_evicted_index"] is None:
        print("  oldest evicted:      none (everything still cached)")
    else:
        print(f"  oldest evicted:      context #{summary['oldest_evicted_index']} "
              f"(older contexts evicted)")

    result = {
        "capacity_tokens": capacity,
        "model": model,
        "context_tokens_target": args.context_tokens,
        "budget_per_context": budget,
        "num_contexts": n,
        "tokens_per_char": tokens_per_char,
        "threshold": threshold,
        "cold_ttft": cold_ttft,
        "hit_ttft": hit_ttft,
        "hydrate": hydrate,
        "verify": verify,
        "retained": summary["retained"],
        "retained_tokens": summary["retained_tokens"],
        "retained_pct": summary["retained_pct"],
        "oldest_evicted_index": summary["oldest_evicted_index"],
    }
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Results written to {args.output}")
    return result


def load_result(path):
    with open(path) as f:
        return json.load(f)


def fmt(d, key):
    v = d.get(key)
    return f"{v:,}" if isinstance(v, int) else ("–" if v is None else str(v))


def compare(a_path, b_path):
    a, b = load_result(a_path), load_result(b_path)
    print(f"  {'':26} {'fix':>14} {'control':>14}")
    for key in ("capacity_tokens", "num_contexts", "retained",
                "retained_tokens", "retained_pct", "oldest_evicted_index"):
        print(f"  {key:26} {fmt(a, key):>14} {fmt(b, key):>14}")
    da = a.get("retained_tokens") or 0
    db = b.get("retained_tokens") or 0
    delta = da - db
    print(f"  {'retained delta':26} {delta:>+14,} tokens")
    if a.get("retained_pct") and b.get("retained_pct"):
        print(f"  {'retained % delta':26} "
              f"{a['retained_pct'] - b['retained_pct']:+,.2f} pp")


def main(argv=None):
    p = argparse.ArgumentParser(description="KV-cache retention pressure benchmark")
    p.add_argument("--base-url", default=DEFAULT_URL)
    p.add_argument("--model", default=None,
                   help="Model id (default: auto-detected via GET /models)")
    p.add_argument("--api-key", default=None)
    p.add_argument("--kv-size", type=int, default=None,
                   help="Advertised KV cache size in tokens (required)")
    p.add_argument("--context-tokens", type=int, default=DEFAULT_CONTEXT_TOKENS)
    p.add_argument("--tokens-per-char", type=float, default=0.25,
                   help="Fallback tokens/char when calibration is skipped/fails")
    p.add_argument("--skip-calibrate", action="store_true")
    p.add_argument("--num-contexts", type=int, default=None,
                   help="Override N (default: ceil(capacity/budget)+margin)")
    p.add_argument("--margin", type=int, default=DEFAULT_MARGIN)
    p.add_argument("--max-tokens-hydrate", type=int, default=1)
    p.add_argument("--max-tokens-verify", type=int, default=1)
    p.add_argument("--hit-threshold", type=float, default=None,
                   help="TTFT threshold in seconds for hit/miss (default: "
                        "calibrated midpoint between cold prefill and cache hit)")
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--output", default=None)
    p.add_argument("--compare", nargs=2, metavar=("FIX_JSON", "CONTROL_JSON"),
                   help="Diff two result files instead of running")
    args = p.parse_args(argv)

    if args.compare:
        compare(*args.compare)
        return
    if args.kv_size is None:
        p.error("the following arguments are required: --kv-size")
    run_pressure(args)


if __name__ == "__main__":
    main()
