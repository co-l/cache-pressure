#!/usr/bin/env python3
"""Tests for the abort-sim mid-thinking-abort prefix-retention benchmark.

Verifies (pure logic, no network):
  - build_prime_messages is deterministic per salt, unique across salts, and
    appends the thinking-inducing problem to the sized lorem body.
  - build_reprompt_messages carries the captured partial reply verbatim:
    branch mode inserts it before the new user instruction, continuation
    mode ends the conversation with it; empty reasoning is omitted.
  - stream_thinking aborts the connection at the abort target, capturing
    exactly the tokens streamed so far (mixed reasoning/content); a stream
    that ends before the target is a scenario failure, as is a transport
    error.
  - send_reprompt reads prompt/cached usage from a non-streaming response,
    with the usage->timings->none source fallback.
  - reuse_fraction / verdict / summarise boundary behavior.
  - annotate_reprompt_records zips ground-truth request-log entries over the
    records (trailing slice), recomputes reuse from the log, and downgrades
    a root re-prefill to a failure.

Run:  uv run python tests/test_abort_sim.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cache_pressure.abort_sim as abort_sim
from cache_pressure.abort_sim import (REORIENT, PROBLEM, SYSTEM,
                                      annotate_reprompt_records,
                                      build_prime_messages,
                                      build_reprompt_messages,
                                      main, reuse_fraction,
                                      send_reprompt, stream_thinking,
                                      summarise, verdict)

FAILURES = []


def check(name, cond, detail=""):
    status = "OK  " if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


# ── fakes ──────────────────────────────────────────────────────────────

def sse(obj):
    return f"data: {json.dumps(obj)}"


def delta_chunk(reasoning=None, content=None):
    delta = {}
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    if content is not None:
        delta["content"] = content
    return {"choices": [{"delta": delta}]}


class FakeStreamResp:
    def __init__(self, lines):
        self.lines = lines
        self.closed = False

    def iter_lines(self):
        return iter(self.lines)

    def close(self):
        self.closed = True


class FakeJsonResp:
    def __init__(self, data):
        self.data = data

    def json(self):
        return self.data


def fake_stream_post(lines):
    holder = {}

    def post(url, json=None, headers=None, stream=None, timeout=None):
        resp = FakeStreamResp(lines)
        holder["resp"] = resp
        return resp
    return post, holder


def fake_json_post(data):
    def post(url, json=None, headers=None, timeout=None):
        return FakeJsonResp(data)
    return post


# ── prime messages ─────────────────────────────────────────────────────

def test_prime_messages():
    a1 = build_prime_messages(salt=123, context_tokens=400)
    a2 = build_prime_messages(salt=123, context_tokens=400)
    b = build_prime_messages(salt=456, context_tokens=400)
    check("prime messages deterministic per salt", a1 == a2)
    check("prime messages unique across salts", a1 != b)
    check("prime messages shape [system, user]",
          len(a1) == 2 and a1[0]["role"] == "system" and a1[1]["role"] == "user")
    check("prime system message", a1[0]["content"] == SYSTEM)
    check("prime user message ends with the problem",
          a1[1]["content"].endswith(PROBLEM))
    check("prime body ~4 chars/token",
          1500 <= len(a1[1]["content"]) <= 2100,
          f"len={len(a1[1]['content'])}")


# ── reprompt messages ──────────────────────────────────────────────────

def test_reprompt_branch():
    prime = build_prime_messages(salt=1, context_tokens=100)
    msgs = build_reprompt_messages(prime, "tho ugh t h i n k",
                                   "partial answer", REORIENT, "branch")
    check("branch keeps the prime prefix verbatim",
          msgs[:2] == prime and msgs[0] is not prime[0] or msgs[:2] == prime)
    check("branch assistant carries the capture verbatim",
          msgs[2] == {"role": "assistant", "content": "partial answer",
                      "reasoning_content": "tho ugh t h i n k"},
          repr(msgs[2]))
    check("branch ends with the new user instruction",
          len(msgs) == 4 and msgs[3] == {"role": "user", "content": REORIENT})


def test_reprompt_continuation():
    prime = build_prime_messages(salt=1, context_tokens=100)
    msgs = build_reprompt_messages(prime, "thinking", "", REORIENT,
                                   "continuation")
    check("continuation appends only the assistant turn", len(msgs) == 3)
    check("continuation ends on the assistant capture",
          msgs[-1]["role"] == "assistant"
          and msgs[-1]["reasoning_content"] == "thinking"
          and msgs[-1]["content"] == "")


def test_reprompt_omits_empty_reasoning():
    prime = build_prime_messages(salt=1, context_tokens=100)
    msgs = build_reprompt_messages(prime, "", "text", REORIENT, "branch")
    check("empty reasoning key omitted",
          "reasoning_content" not in msgs[2] and msgs[2]["content"] == "text")


# ── stream_thinking ────────────────────────────────────────────────────

def test_stream_aborts_at_target():
    lines = [sse(delta_chunk(reasoning=f"r{i}")) for i in range(9)] + \
            ["data: [DONE]"]
    post, holder = fake_stream_post(lines)
    rec = stream_thinking("http://x/v1/chat/completions", "m", [], 1000,
                          abort_after=5, timeout=10, post_fn=post)
    check("aborted at the target", rec["aborted"] is True)
    check("exactly 5 tokens captured", rec["tokens"] == 5,
          f"got {rec['tokens']}")
    check("reasoning tokens counted", rec["reasoning_tokens"] == 5)
    check("no content tokens", rec["content_tokens"] == 0)
    check("capture is the verbatim prefix",
          rec["reasoning"] == "r0r1r2r3r4" and rec["content"] == "")
    check("no error", rec["error"] is None)
    check("connection closed on abort", holder["resp"].closed is True)


def test_stream_mixed_delta_abort():
    lines = [
        sse(delta_chunk(reasoning="a")),
        sse(delta_chunk(content="b")),
        sse(delta_chunk(reasoning="c")),
        sse(delta_chunk(content="d")),
        sse(delta_chunk(reasoning="e")),
    ]
    post, _holder = fake_stream_post(lines)
    rec = stream_thinking("http://x", "m", [], 1000, abort_after=4,
                          timeout=10, post_fn=post)
    check("mixed abort at 4", rec["aborted"] and rec["tokens"] == 4)
    check("mixed counts", rec["reasoning_tokens"] == 2
          and rec["content_tokens"] == 2)
    check("mixed capture", rec["reasoning"] == "ac" and rec["content"] == "bd")


def test_stream_early_finish_is_scenario_failure():
    lines = [sse(delta_chunk(reasoning="a")),
             sse(delta_chunk(content="b")),
             "data: [DONE]"]
    post, _holder = fake_stream_post(lines)
    rec = stream_thinking("http://x", "m", [], 1000, abort_after=10,
                          timeout=10, post_fn=post)
    check("not aborted", rec["aborted"] is False)
    check("full output still captured", rec["reasoning"] == "a"
          and rec["content"] == "b" and rec["tokens"] == 2)
    check("scenario failure reported", rec["error"] is not None
          and "abort target" in rec["error"], str(rec["error"]))


def test_stream_transport_error():
    def boom(url, json=None, headers=None, stream=None, timeout=None):
        raise RuntimeError("connection reset")
    rec = stream_thinking("http://x", "m", [], 1000, abort_after=10,
                          timeout=10, post_fn=boom)
    check("error recorded", rec["error"] == "connection reset")
    check("nothing captured", rec["tokens"] == 0 and not rec["aborted"])


# ── send_reprompt ──────────────────────────────────────────────────────

def test_send_reprompt_usage():
    data = {"usage": {"prompt_tokens": 1000,
                      "prompt_tokens_details": {"cached_tokens": 950}},
            "timings": {"cache_n": 940}}
    rec = send_reprompt("http://x", "m", [], 0, timeout=10,
                        post_fn=fake_json_post(data))
    check("prompt tokens read", rec["prompt"] == 1000)
    check("usage source wins", rec["cached"] == 950
          and rec["cached_source"] == "usage")
    check("no error", rec["error"] is None)


def test_send_reprompt_timings_fallback_and_none():
    data = {"usage": {"prompt_tokens": 1000},
            "timings": {"cache_n": 940}}
    rec = send_reprompt("http://x", "m", [], 0, timeout=10,
                        post_fn=fake_json_post(data))
    check("timings fallback", rec["cached"] == 940
          and rec["cached_source"] == "timings")
    rec = send_reprompt("http://x", "m", [], 0, timeout=10,
                        post_fn=fake_json_post({"usage": {"prompt_tokens": 5}}))
    check("no cache signal -> none", rec["cached"] is None
          and rec["cached_source"] == "none")


def test_send_reprompt_error():
    def boom(url, json=None, headers=None, timeout=None):
        raise RuntimeError("bad request: max_tokens")
    rec = send_reprompt("http://x", "m", [], 0, timeout=10, post_fn=boom)
    check("error recorded", rec["error"] == "bad request: max_tokens")
    check("no usage", rec["prompt"] is None and rec["cached"] is None)


# ── verdict / summarise ────────────────────────────────────────────────

def _rec(**kw):
    base = {"run": 0, "salt": 0, "aborted": True, "tokens": 500,
            "reasoning_tokens": 500, "content_tokens": 0,
            "reasoning": "r", "content": "", "ttft": 1.0, "wall": 10.0,
            "prompt": 41000, "cached": 39000, "cached_source": "usage",
            "reuse": 39000 / 41000, "reprompt_wall": 2.0, "error": None,
            "ok": False}
    base.update(kw)
    return base


def test_reuse_fraction():
    check("full", abs(reuse_fraction(950, 1000) - 0.95) < 1e-12)
    check("zero", reuse_fraction(0, 1000) == 0.0)
    check("none cached -> None", reuse_fraction(None, 1000) is None)
    check("no prompt -> None", reuse_fraction(10, 0) is None)


def test_verdict():
    check("pass at threshold", verdict(_rec(reuse=0.95), 0.95) is True)
    check("fail below threshold", verdict(_rec(reuse=0.9499), 0.95) is False)
    check("fail without reuse", verdict(_rec(reuse=None, cached=None),
                                        0.95) is False)
    check("fail when not aborted", verdict(_rec(aborted=False,
                                                error="stream finished before "
                                                      "the abort target"),
                                           0.95) is False)
    check("fail on error", verdict(_rec(error="connection reset"),
                                   0.95) is False)
    check("root re-prefill downgrades a high reuse",
          verdict(_rec(reuse=0.99, reuse_path="root"), 0.95) is False)
    check("non-root path passes",
          verdict(_rec(reuse=0.99, reuse_path="private_turn_closure"),
                  0.95) is True)


def test_summarise():
    recs = [_rec(ok=True), _rec(ok=False)]
    s = summarise(recs, 0.95)
    check("counts", s["runs"] == 2 and s["ok"] == 1 and s["all_ok"] is False)
    check("all ok", summarise([_rec(ok=True)], 0.95)["all_ok"] is True)
    check("empty run is not ok", summarise([], 0.95)["all_ok"] is False)


# ── annotation with the ground-truth request log ───────────────────────

def _entry(prompt, hit, path, computed, request_id=None, ts=None):
    return {"request_id": request_id, "prompt": prompt,
            "computed_prefill_tokens": computed,
            "prefix_cache_hit_tokens": hit, "prefix_reuse_path": path,
            "ts": ts}


def test_annotate():
    recs = [_rec(ok=True), _rec(ok=True)]
    entries = [
        _entry(99999, 99000, "private_turn_closure", 999),   # earlier run
        _entry(41000, 0, "root", 41000),                     # this run: miss
        _entry(41000, 40800, "private_turn_closure", 200),   # this run: hit
    ]
    out, unmatched, extra = annotate_reprompt_records(recs, entries, 0.95)
    check("earlier traffic excluded", len(extra) == 1
          and extra[0]["prefix_reuse_path"] == "private_turn_closure")
    check("no unmatched", not unmatched)
    check("ground truth recomputes reuse",
          abs(out[0]["reuse"]) > 0 or out[0]["reuse"] == 0.0)
    check("miss run downgraded", out[0]["ok"] is False
          and out[0]["reuse_path"] == "root"
          and out[0]["reuse_class"] == "miss"
          and out[0]["computed"] == 41000)
    check("hit run kept", out[1]["ok"] is True
          and out[1]["reuse_path"] == "private_turn_closure"
          and out[1]["reuse_class"] == "hit"
          and abs(out[1]["reuse"] - 40800 / 41000) < 1e-12)


def test_annotate_short_log():
    recs = [_rec(ok=True), _rec(ok=True)]
    entries = [_entry(41000, 40800, "private_turn_closure", 200)]
    out, unmatched, extra = annotate_reprompt_records(recs, entries, 0.95)
    check("one unmatched", len(unmatched) == 1 and not extra)
    check("annotated one kept", out[0]["ok"] is True)
    check("unmatched one untouched",
          unmatched[0].get("reuse_path") is None and unmatched[0]["ok"]
          is True)


def test_annotate_timestamp_matching():
    # concurrent traffic interleaves the log: the trailing slice would
    # mismatch, timestamp matching must not
    t0 = 1_000_000_000_000
    recs = [
        _rec(ok=True, reprompt_ts=t0 + 10_000, prompt=38390),
        _rec(ok=True, reprompt_ts=t0 + 60_000, prompt=38488),
    ]
    entries = [
        _entry(100, 100, 0, "root", request_id=1, ts=t0 + 5_000),
        _entry(38390, 0, "root", 38390, request_id=11, ts=t0 + 10_200),
        _entry(140000, 139538, 462, "private_endpoint",
               request_id=20, ts=t0 + 40_000),
        _entry(38488, 0, "root", 38488, request_id=21, ts=t0 + 60_300),
    ]
    out, unmatched, extra = annotate_reprompt_records(recs, entries, 0.95)
    check("no unmatched", not unmatched)
    check("run 1 matched its own entry, not the concurrent one",
          out[0]["request_id"] == 11 and out[0]["reuse_path"] == "root"
          and out[0]["ok"] is False)
    check("run 2 matched its own entry",
          out[1]["request_id"] == 21 and out[1]["reuse_path"] == "root"
          and out[1]["ok"] is False)
    check("concurrent entries reported as extra", len(extra) == 2)


def test_annotate_prompt_mismatch_skipped():
    t0 = 1_000_000_000_000
    recs = [_rec(ok=True, reprompt_ts=t0 + 10_000, prompt=38390)]
    entries = [
        _entry(12345, 0, "root", 12345, request_id=1, ts=t0 + 10_100),
        _entry(38390, 0, "root", 38390, request_id=2, ts=t0 + 10_400),
    ]
    out, unmatched, extra = annotate_reprompt_records(recs, entries, 0.95)
    check("wrong-size entry in the window is skipped",
          not unmatched and out[0]["request_id"] == 2
          and len(extra) == 1)


def test_annotate_outside_tolerance_unmatched():
    t0 = 1_000_000_000_000
    recs = [_rec(ok=True, reprompt_ts=t0 + 10_000, prompt=38390)]
    entries = [_entry(38390, 0, "root", 38390, request_id=1,
                      ts=t0 + 10_000 + 60_000)]
    out, unmatched, extra = annotate_reprompt_records(recs, entries, 0.95)
    check("outside the tolerance window -> unmatched",
          len(unmatched) == 1 and not out)


# ── CLI surface ────────────────────────────────────────────────────────

def test_cli_rejects_bad_mode():
    try:
        main(["--reprompt-mode", "bogus"])
        check("bad mode rejected", False)
    except SystemExit as exc:
        check("bad mode rejected", exc.code != 0)


def test_default_salt_resolved_before_run_bench():
    orig = abort_sim.run_bench
    captured = {}

    def fake_run_bench(args, model):
        captured["salt"] = args.salt
        return []

    abort_sim.run_bench = fake_run_bench
    try:
        try:
            main(["--model", "m", "--max-context", "1000",
                  "--context-tokens", "10", "--runs", "1",
                  "--max-tokens", "8"])
        except SystemExit as exc:
            check("no-salt CLI reaches the bench and exits", exc.code == 1)
    finally:
        abort_sim.run_bench = orig
    check("salt resolved to int before run_bench",
          isinstance(captured.get("salt"), int))


if __name__ == "__main__":
    test_prime_messages()
    test_reprompt_branch()
    test_reprompt_continuation()
    test_reprompt_omits_empty_reasoning()
    test_stream_aborts_at_target()
    test_stream_mixed_delta_abort()
    test_stream_early_finish_is_scenario_failure()
    test_stream_transport_error()
    test_send_reprompt_usage()
    test_send_reprompt_timings_fallback_and_none()
    test_send_reprompt_error()
    test_reuse_fraction()
    test_verdict()
    test_summarise()
    test_annotate()
    test_annotate_short_log()
    test_annotate_timestamp_matching()
    test_annotate_prompt_mismatch_skipped()
    test_annotate_outside_tolerance_unmatched()
    test_cli_rejects_bad_mode()
    test_default_salt_resolved_before_run_bench()
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s):")
        for name in FAILURES:
            print(f"  - {name}")
        sys.exit(1)
    print("All checks passed.")
