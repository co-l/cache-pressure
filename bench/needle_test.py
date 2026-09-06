#!/usr/bin/env python3
"""Needle-in-haystack retrieval validation for the DS4 prod engine.

Sweeps increasing context lengths, each with a unique haystack and a single
needle sentence deep in the text (default 0.8 depth). The model must
retrieve the exact secret code — end-to-end proof that long-context
attention/KV is intact on the live engine. Haystacks are unique per run
(random salt), so no prefix-cache hit can mask a broken prefill.

Run:
  python3 needle_test.py                      # 50K/100K/200K/300K/450K
  python3 needle_test.py --lengths 50000,100000
  python3 needle_test.py --thinking           # let it reason (slower)
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import requests

from cache_pressure import calibrate, make_body

DEFAULT_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_LENGTHS = [50000, 100000, 200000, 300000, 450000]
NEEDLE_WORDS = ["plover", "zirconium", "octopus", "thimble", "wombat", "kestrel"]
QUESTION = (
    "What is the secret code written in square brackets in the text? "
    "Reply with the code only."
)


def chat_stream(url, model, messages, max_tokens, timeout, api_key=None, thinking=None):
    """Stream one completion, capturing full content + reasoning + usage."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if thinking is not None:
        payload["chat_template_kwargs"] = {"thinking": thinking}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    content, reasoning = [], []
    ttft = None
    prompt_tokens = completion_tokens = 0
    error = None
    t0 = time.monotonic()
    try:
        resp = requests.post(url, json=payload, headers=headers, stream=True,
                             timeout=timeout)
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
            usage = chunk.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta", {})
                c = delta.get("content") or ""
                r = delta.get("reasoning") or delta.get("reasoning_content") or ""
                if ttft is None and (c or r):
                    ttft = time.monotonic() - t0
                if c:
                    content.append(c)
                if r:
                    reasoning.append(r)
    except Exception as e:
        error = str(e)
    text = "".join(content)
    return text, text + "".join(reasoning), prompt_tokens, completion_tokens, \
        ttft, time.monotonic() - t0, error


def make_needle(rng):
    code = f"{rng.choice(NEEDLE_WORDS)}-{rng.randrange(1000, 9999)}"
    return f"[the secret code is {code}]", code


def build_haystack(chars, salt, needle, pos):
    text = make_body(chars, salt=salt)
    idx = int(len(text) * pos)
    return text[:idx] + "\n" + needle + "\n" + text[idx:]


def run_length(url, model, target, tpc, salt, pos, max_tokens, timeout,
               api_key, thinking):
    rng = random.Random(4242 + salt)
    needle, code = make_needle(rng)
    chars = int(target / tpc)
    haystack = build_haystack(chars, salt, needle, pos)
    messages = [
        {"role": "system", "content": "You are a careful reader."},
        {"role": "user", "content": haystack + "\n\n" + QUESTION},
    ]
    text, full, prompt, comp, ttft, e2e, err = chat_stream(
        url, model, messages, max_tokens, timeout, api_key, thinking)
    ok = err is None and code.lower() in full.lower()
    verdict = "PASS" if ok else "FAIL"
    print(f"  {target // 1000:>3}K: {verdict}  prompt={prompt:,}  "
          f"out={comp}  ttft={(ttft or 0.0):.1f}s  "
          f"total={e2e:.0f}s")
    print(f"      needle={code}  response={full[:160]!r}")
    if err:
        print(f"      error={err}")
    return {
        "target": target, "ok": ok, "prompt_tokens": prompt,
        "completion_tokens": comp, "ttft": ttft, "time_s": round(e2e, 1),
        "needle": code, "error": err, "response": full[:300],
    }


def main():
    p = argparse.ArgumentParser(description="Needle-in-haystack retrieval validation")
    p.add_argument("--base-url", default=DEFAULT_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--api-key", default=None)
    p.add_argument("--lengths", default=",".join(str(x) for x in DEFAULT_LENGTHS),
                   help="Comma-separated context lengths in tokens")
    p.add_argument("--needle-pos", type=float, default=0.8,
                   help="Needle insertion depth in the haystack (0..1)")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--thinking", action="store_true",
                   help="Enable model thinking (default: off)")
    p.add_argument("--timeout", type=int, default=1200)
    p.add_argument("--tokens-per-char", type=float, default=None,
                   help="Skip calibration and use this ratio")
    p.add_argument("--salt", type=int, default=None,
                   help="Run salt (default: time-based, fresh haystacks)")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    salt = args.salt if args.salt is not None else int(time.time())
    url = args.base_url.rstrip("/") + "/chat/completions"
    rng = random.Random(salt)

    max_len = max(lengths)
    tpc = args.tokens_per_char
    if tpc is None:
        tpc, _ = calibrate(args.base_url, args.model, args.api_key,
                           args.timeout, max_len)
        if tpc is None:
            print("  Calibration failed — pass --tokens-per-char")
            sys.exit(1)
        print(f"  Calibrated tokens/char: {tpc:.4f} (at ~{max_len:,} tok)")
    else:
        print(f"  Using tokens/char: {tpc:.4f} (no calibration)")

    print(f"Needle sweep: {', '.join(f'{x//1000}K' for x in lengths)} "
          f"(depth {args.needle_pos:.0%}, thinking={'on' if args.thinking else 'off'}, "
          f"salt {salt})")
    results = []
    for target in sorted(lengths):
        results.append(run_length(url, args.model, target, tpc,
                                  salt + target, args.needle_pos,
                                  args.max_tokens, args.timeout,
                                  args.api_key, args.thinking))

    passed = sum(1 for r in results if r["ok"])
    print(f"\nRESULT: {passed}/{len(results)} needles found")
    if args.output:
        Path(args.output).write_text(json.dumps({
            "salt": salt, "needle_pos": args.needle_pos,
            "tokens_per_char": tpc, "results": results,
        }, indent=2))
        print(f"  Results written to {args.output}")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
