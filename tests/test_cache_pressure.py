#!/usr/bin/env python3
"""Tests for the cache-pressure benchmark (cache_pressure).

Verifies:
  - verify_phase stops after the first MISS: the reverse-order verify walks
    back from newest to oldest, and under LRU everything older than the first
    miss is evicted by construction — so remaining probes are skipped.
  - summarise reflects the truncated verify rows (retained hits + first miss).
  - resolve_model auto-detects the served model from GET /models.
  - calibrate_threshold derives the hit/miss threshold from measured
    cold-prefill and cache-hit TTFT.

Run:  python3 tests/test_cache_pressure.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
from cache_pressure import (calibrate_threshold, do_request, hydrate_phase,
                            resolve_model, summarise, verify_phase)

FAILURES = []


def check(name, cond, detail=""):
    status = "OK  " if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def fake_do_request(calls):
    def req(url, model, messages, max_tokens, timeout, api_key=None):
        idx = int(messages[1]["content"].split(":")[1])
        calls.append(idx)
        if idx <= 1:
            return (20.0, 21.0, 1, 39000, None)
        return (0.4, 0.6, 1, 39000, None)
    return req


def fake_get(payload):
    def get_fn(url, headers=None, timeout=10):
        return payload
    return get_fn


def test_verify_stops_after_first_miss():
    calls = []
    contexts = [f"ctx:{i}" for i in range(5)]
    rows = verify_phase("http://x", "m", contexts, 1, 60, None, 3.0, 2047000,
                        request_fn=fake_do_request(calls))
    check("stopped after first miss (4 probes, not 5)",
          len(calls) == 4, f"calls={len(calls)} ({calls})")
    check("rows cover indices 4..1",
          [r["index"] for r in rows] == [4, 3, 2, 1],
          str([r["index"] for r in rows]))
    check("first miss is ctx#1 (older not sent)",
          rows[-1]["index"] == 1 and not rows[-1]["hit"],
          str(rows[-1]))
    summary = summarise(2047000, contexts, rows, 3.0)
    check("retained counts the 3 hits", summary["retained"] == 3,
          str(summary["retained"]))
    check("oldest evicted = ctx#1",
          summary["oldest_evicted_index"] == 1,
          str(summary["oldest_evicted_index"]))


def test_resolve_model_single():
    payload = {"object": "list", "data": [{"id": "m1", "object": "model"}]}
    got = resolve_model("http://x/v1", get_fn=fake_get(payload))
    check("single model detected", got == "m1", str(got))


def test_resolve_model_multiple_picks_first():
    payload = {"object": "list", "data": [{"id": "m1"}, {"id": "m2"}]}
    got = resolve_model("http://x/v1", get_fn=fake_get(payload))
    check("first of multiple models picked", got == "m1", str(got))


def test_resolve_model_failure_returns_none():
    def boom(url, headers=None, timeout=10):
        raise Exception("boom")
    got = resolve_model("http://x/v1", get_fn=boom)
    check("endpoint failure -> None", got is None)


def test_resolve_model_llamacpp_models_key():
    payload = {"models": [{"name": "qwen-27b", "model": "qwen-27b"}]}
    got = resolve_model("http://x/v1", get_fn=fake_get(payload))
    check("llama.cpp 'models' key detected", got == "qwen-27b", str(got))


def test_resolve_model_bare_list():
    payload = [{"id": "m9", "object": "model"}]
    got = resolve_model("http://x/v1", get_fn=fake_get(payload))
    check("bare list payload detected", got == "m9", str(got))


def test_calibrate_threshold_midpoint():
    calls = []

    def req(url, model, messages, max_tokens, timeout, api_key=None):
        calls.append(1)
        if len(calls) == 1:
            return (20.0, 21.0, 1, 8000, None)
        return (0.5, 0.7, 1, 8000, None)

    th, cold, hit = calibrate_threshold(
        "http://x", "m", None, 60, 8000, 0.25, request_fn=req)
    check("midpoint threshold", th is not None and abs(th - 10.25) < 1e-9,
          str(th))
    check("cold ttft recorded", cold == 20.0, str(cold))
    check("hit ttft recorded", hit == 0.5, str(hit))


def test_calibrate_threshold_error_returns_none():
    def req(url, model, messages, max_tokens, timeout, api_key=None):
        return (0.0, 1.0, 0, None, "error")

    th, cold, hit = calibrate_threshold(
        "http://x", "m", None, 60, 8000, 0.25, request_fn=req)
    check("probe error -> (None, None, None)",
          th is None and cold is None and hit is None,
          f"{th} {cold} {hit}")


def test_calibrate_threshold_hit_not_faster_returns_none():
    def req(url, model, messages, max_tokens, timeout, api_key=None):
        return (0.4, 0.5, 1, 8000, None)

    th, cold, hit = calibrate_threshold(
        "http://x", "m", None, 60, 8000, 0.25, request_fn=req)
    check("hit not faster than cold -> (None, cold, hit)",
          th is None and cold == 0.4 and hit == 0.4, f"{th} {cold} {hit}")


def test_calibrate_threshold_uses_fresh_probe():
    seen = set()

    def req(url, model, messages, max_tokens, timeout, api_key=None):
        seen.add(messages[1]["content"])
        return (1.0, 1.1, 1, 8000, None)

    calibrate_threshold("http://x", "m", None, 60, 8000, 0.25, request_fn=req)
    calibrate_threshold("http://x", "m", None, 60, 8000, 0.25, request_fn=req)
    check("probe context is run-unique (no cached-leftover collision)",
          len(seen) >= 2, f"{len(seen)} distinct probe texts")


def test_kv_size_required():
    try:
        import cache_pressure
        cache_pressure.main(["--num-contexts", "3"])
        check("--kv-size required (argparse exit 2)", False)
    except SystemExit as e:
        check("--kv-size required (argparse exit 2)", e.code == 2,
              f"exit code {e.code}")


class FakeStream:
    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        pass

    def iter_lines(self):
        return iter(self._lines)


def test_do_request_counts_reasoning_content_for_ttft():
    import cache_pressure
    lines = [
        b'data: {"choices":[{"delta":{"role":"assistant"}}]}',
        b'data: {"choices":[{"delta":{"reasoning_content":" thinking"}}]}',
        b'data: {"choices":[{"delta":{}}],'
        b'"usage":{"prompt_tokens":42,"completion_tokens":1}}',
        b'data: [DONE]',
    ]
    orig = cache_pressure.requests.post
    cache_pressure.requests.post = lambda *a, **k: FakeStream(lines)
    try:
        ttft, _e2e, out, prompt, err = do_request(
            "http://x/v1/chat/completions", "m", [], 1, 60)
    finally:
        cache_pressure.requests.post = orig
    check("ttft set from reasoning_content", ttft > 0, str(ttft))
    check("reasoning token counted as output", out == 1, str(out))
    check("prompt tokens from usage", prompt == 42, str(prompt))
    check("no error", err is None, str(err))


def test_hydrate_phase_survives_zero_ttft():
    import contextlib
    import io

    import cache_pressure
    orig = cache_pressure.do_request
    cache_pressure.do_request = lambda *a, **k: (0.0, 0.0, 0, 8000, None)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            hydrate_phase("http://x", "m", ["ctx"], 1, 60, None, 100000)
    finally:
        cache_pressure.do_request = orig
    check("hydrate_phase does not crash on zero ttft", "prefill" in buf.getvalue(),
          buf.getvalue().strip())


def main():
    test_verify_stops_after_first_miss()
    test_resolve_model_single()
    test_resolve_model_multiple_picks_first()
    test_resolve_model_failure_returns_none()
    test_resolve_model_llamacpp_models_key()
    test_resolve_model_bare_list()
    test_calibrate_threshold_midpoint()
    test_calibrate_threshold_error_returns_none()
    test_calibrate_threshold_hit_not_faster_returns_none()
    test_calibrate_threshold_uses_fresh_probe()
    test_kv_size_required()
    test_do_request_counts_reasoning_content_for_ttft()
    test_hydrate_phase_survives_zero_ttft()
    print(f"\n{len(FAILURES)} failure(s)")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
