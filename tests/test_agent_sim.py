#!/usr/bin/env python3
"""Tests for the agent-sim concurrent-session benchmark.

Verifies (pure logic, no network):
  - make_body is deterministic per salt and unique across salts.
  - build_session_plan produces the expected per-session structure: main
    chunks, sub-agent windows, finalize chunk, all unique across sessions.
  - compute_reuse derives the reused-prefix fraction correctly.
  - validate_sizes rejects workloads that would overflow max-context.
  - summarise computes the headline metrics (main-continuation reuse,
    finalize survival, sub coldness) from a record list.
  - run_session assembles messages so the finalize turn builds on exactly
    the main-conversation prefix as it stood before the sub-agent calls
    (sub windows never touch the main message list), and forwards the
    thinking flag to the transport.
  - thinking is off by default (reasoning_effort none).

Run:  uv run python tests/test_agent_sim.py
"""
import json
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cache_pressure.agent_sim as agent_sim
from cache_pressure.agent_sim import (SYSTEM, LiveView, _bar, _fmt_record,
                                      build_session_plan, compute_reuse,
                                      finalize_ttft_verdict, format_finalize,
                                      fmt_elapsed, fmt_ktok, fmt_tps,
                                      label_finalize, render_frame, summarise,
                                      validate_sizes, viz_enabled)
from cache_pressure.core import make_body

FAILURES = []


def check(name, cond, detail=""):
    status = "OK  " if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def test_make_body_deterministic_and_unique():
    a1 = make_body(20000, salt=123)
    a2 = make_body(20000, salt=123)
    b = make_body(20000, salt=456)
    check("make_body deterministic per salt", a1 == a2)
    check("make_body unique across salts", a1 != b)
    check("make_body target length", len(a1) <= 20000 < len(a1) + 20)


def test_build_session_plan():
    plan = build_session_plan(salt=42, step_tokens=5000, steps=6,
                              sub_tokens=40000, sub_windows=2, finalize_new=2000)
    check("main has 6 chunks", len(plan["main"]) == 6)
    check("subs has 2 windows", len(plan["subs"]) == 2)
    check("finalize present", isinstance(plan["finalize"], str) and plan["finalize"])
    check("main chunks ~5K tokens", all(
        18000 <= len(c) <= 22000 for c in plan["main"]))
    check("sub windows ~40K tokens", all(
        155000 <= len(c) <= 165000 for c in plan["subs"][0]))


def test_plans_unique_across_sessions():
    p1 = build_session_plan(salt=100, step_tokens=5000, steps=3,
                            sub_tokens=40000, sub_windows=2, finalize_new=2000)
    p2 = build_session_plan(salt=200, step_tokens=5000, steps=3,
                            sub_tokens=40000, sub_windows=2, finalize_new=2000)
    check("main chunks differ across sessions", p1["main"] != p2["main"])
    check("subs differ across sessions", p1["subs"] != p2["subs"])
    check("finalize differs across sessions", p1["finalize"] != p2["finalize"])


def test_compute_reuse():
    check("full reuse", abs(compute_reuse(9900, 10000) - 0.99) < 1e-9)
    check("no cache", compute_reuse(0, 10000) == 0.0)
    check("none cached -> 0", compute_reuse(None, 10000) == 0.0)
    check("no previous prompt -> 0", compute_reuse(100, 0) == 0.0)


def test_validate_sizes():
    ok = validate_sizes(main_size=150000, sub_tokens=40000,
                        finalize_new=2000, max_context=252928)
    check("valid workload accepted", ok is None, str(ok))
    err1 = validate_sizes(main_size=260000, sub_tokens=40000,
                          finalize_new=2000, max_context=252928)
    check("main overflow rejected", err1 is not None, str(err1))
    err2 = validate_sizes(main_size=150000, sub_tokens=260000,
                          finalize_new=2000, max_context=252928)
    check("sub overflow rejected", err2 is not None, str(err2))


def _records(session, phase, step, prompt, cached, reuse=None):
    return {"session": session, "phase": phase, "step": step,
            "prompt": prompt, "cached": cached, "reuse": reuse}


def test_summarise():
    recs = []
    # 2 sessions, each: main t0 cold, main t1+t2 reused, 2 cold subs,
    # finalize reused
    for s in (0, 1):
        recs.append(_records(s, "main", 0, 5000, 0, 0.0))
        recs.append(_records(s, "main", 1, 10000, 9900, 0.99))
        recs.append(_records(s, "main", 2, 15000, 14800, 0.99))
        recs.append(_records(s, "sub", 0, 40000, 0, 0.0))
        recs.append(_records(s, "sub", 1, 40000, 5, 0.0))
        recs.append(_records(s, "finalize", 0, 17000, 15000, 0.99))

    out = summarise(recs, num_sessions=2, sub_windows=2, threshold=0.95)
    check("all main continuations reused", out["main_reused"] == 4 and
          out["main_total"] == 4)
    check("both finalize survived", out["finalize_ok"] == 2 and
          out["finalize_total"] == 2)
    check("subs cold", out["subs_cold"] == 4 and out["subs_total"] == 4)
    check("elapsed defaults to None", out["elapsed"] is None)

    out_e = summarise(recs, num_sessions=2, sub_windows=2, threshold=0.95,
                      elapsed=73.4)
    check("elapsed passed through", out_e["elapsed"] == 73.4)

    bad = [_records(0, "main", 1, 10000, 100, 0.01),
           _records(0, "finalize", 0, 12000, 500, 0.04)]
    out2 = summarise(bad, num_sessions=1, sub_windows=2, threshold=0.95)
    check("degraded run reported", out2["main_reused"] == 0 and
          out2["finalize_ok"] == 0)

    out66 = summarise([_records(0, "main", 1, 10000, 8000, 0.8)],
                      num_sessions=1, sub_windows=2, threshold=0.66)
    check("80% reuse counts as reused at the 66% threshold",
          out66["main_reused"] == 1 and out66["main_total"] == 1)
    out95 = summarise([_records(0, "main", 1, 10000, 8000, 0.8)],
                      num_sessions=1, sub_windows=2, threshold=0.95)
    check("80% reuse is a miss at the 95% threshold",
          out95["main_reused"] == 0 and out95["main_total"] == 1)


def test_summarise_excludes_aborted():
    recs = [
        _records(0, "main", 0, 5000, 0, 0.0),
        dict(_records(0, "main", 1, 10000, 0, 0.01), aborted=True),
        _records(0, "main", 2, 15000, 14800, 0.99),
        dict(_records(0, "sub", 0, 40000, 0, 0.0), aborted=True),
        _records(0, "finalize", 0, 17000, 15000, 0.99),
    ]
    out = summarise(recs, num_sessions=1, sub_windows=2, threshold=0.95)
    check("aborted main continuation excluded from the miss count",
          out["main_total"] == 1 and out["main_reused"] == 1)
    check("aborted sub excluded from the cold count",
          out["subs_total"] == 0 and out["subs_cold"] == 0)
    check("completed finalize kept", out["finalize_total"] == 1 and
          out["finalize_ok"] == 1)


def test_finalize_ttft_verdict():
    def rec(sid, phase, step, ttft, aborted=False):
        r = {"session": sid, "phase": phase, "step": step, "ttft": ttft}
        if aborted:
            r["aborted"] = True
        return r

    recs = []
    # s0: clean 10s baseline, fast finalize -> 0.3x ok
    recs.append(rec(0, "main", 0, 5.0))
    for i in (1, 2, 3):
        recs.append(rec(0, "main", i, 10.0))
    recs.append(rec(0, "finalize", 0, 3.0))
    # s1: spike in the middle, median must ignore it -> 2.5x fail at 2.0
    recs.append(rec(1, "main", 0, 5.0))
    for i, t in ((1, 10.0), (2, 40.0), (3, 10.0)):
        recs.append(rec(1, "main", i, t))
    recs.append(rec(1, "finalize", 0, 25.0))
    # s2: late spike, median filters it -> 1.81x ok
    recs.append(rec(2, "main", 0, 5.0))
    for i, t in ((1, 16.0), (2, 46.0), (3, 16.0)):
        recs.append(rec(2, "main", i, t))
    recs.append(rec(2, "finalize", 0, 29.0))

    out = finalize_ttft_verdict(recs, ratio=2.0)
    check("s0 fast finalize passes", out[0]["ttft_ok"] is True
          and out[0]["ttft_ratio"] == 0.3, str(out[0]))
    check("s1 degraded finalize fails", out[1]["ttft_ok"] is False
          and out[1]["ttft_ratio"] == 2.5, str(out[1]))
    check("s2 median ignores the spike", out[2]["ttft_ok"] is True
          and out[2]["baseline_ttft"] == 16.0
          and out[2]["ttft_ratio"] == 29.0 / 16.0, str(out[2]))

    check("median-of-3, not median-of-all",
          finalize_ttft_verdict(
              [rec(9, "main", 0, 5.0), rec(9, "main", 1, 10.0),
               rec(9, "main", 2, 10.0), rec(9, "main", 3, 10.0),
               rec(9, "main", 4, 45.0), rec(9, "main", 5, 45.0),
               rec(9, "finalize", 0, 60.0)], ratio=2.0)[9]["ttft_ratio"] == 1.3333333333333333)

    check("exact 2x is not a pass",
          finalize_ttft_verdict(
              [rec(8, "main", 0, 5.0), rec(8, "main", 1, 10.0),
               rec(8, "main", 2, 10.0), rec(8, "finalize", 0, 20.0)],
              ratio=2.0)[8]["ttft_ok"] is False)

    check("custom ratio widens the gate",
          finalize_ttft_verdict(recs, ratio=3.0)[1]["ttft_ok"] is True)

    boundary = [rec(6, "main", 0, 5.0), rec(6, "main", 1, 10.0),
                rec(6, "main", 2, 10.0), rec(6, "finalize", 0, 18.0)]
    check("1.8x fails the 1.75 default",
          finalize_ttft_verdict(boundary)[6]["ttft_ok"] is False)
    check("1.8x passes the wider 2.0 gate",
          finalize_ttft_verdict(boundary, ratio=2.0)[6]["ttft_ok"] is True)

    unverified = finalize_ttft_verdict(
        [rec(4, "main", 0, 5.0), rec(4, "finalize", 0, 3.0)], ratio=2.0)[4]
    check("no main-continuation baseline -> unverified, never a silent pass",
          unverified["baseline_ttft"] is None and unverified["ttft_ratio"] is None
          and unverified["ttft_ok"] is False)

    aborted = finalize_ttft_verdict(
        [rec(5, "main", 0, 5.0), rec(5, "main", 1, 10.0),
         rec(5, "main", 2, 10.0), rec(5, "main", 3, 999.0, aborted=True),
         rec(5, "finalize", 0, 25.0)], ratio=2.0)[5]
    check("aborted turns are excluded from the baseline",
          aborted["ttft_ratio"] == 2.5 and aborted["ttft_ok"] is False,
          str(aborted))


def test_label_finalize():
    def v(reuse, ratio_, aborted=False):
        return {"reuse": reuse, "ttft_ratio": ratio_, "aborted": aborted}
    check("clean finalize is ok",
          label_finalize(v(1.0, 0.3), 0.66, 1.75) == "ok")
    check("reported miss is evicted",
          label_finalize(v(0.0, 1.74), 0.66, 1.75) == "evicted")
    check("slow hit is degraded",
          label_finalize(v(1.0, 3.9), 0.66, 1.75) == "degraded")
    check("both signals failing",
          label_finalize(v(0.0, 3.9), 0.66, 1.75) == "evicted, degraded")
    check("boundary ratio fails the strict gate",
          label_finalize(v(1.0, 1.75), 0.66, 1.75) == "degraded")
    check("just under the boundary is ok",
          label_finalize(v(1.0, 1.74), 0.66, 1.75) == "ok")
    check("no baseline is unverified",
          label_finalize(v(None, None), 0.66, 1.75) == "unverified")
    check("aborted wins", label_finalize(v(0.0, 1.74, True), 0.66, 1.75) == "aborted")


def test_format_finalize():
    def v(base, fin, ratio_, ok, verdict_, aborted=False):
        return {"baseline_ttft": base, "finalize_ttft": fin,
                "ttft_ratio": ratio_, "ttft_ok": ok,
                "reuse": 0.0 if verdict_ != "ok" else 1.0,
                "aborted": aborted, "verdict": verdict_}

    good = {"finalize_ttft": {
        0: v(12.6, 2.2, 0.17, True, "ok"),
        1: v(12.4, 3.6, 0.29, True, "ok"),
        2: v(10.5, 1.9, 0.18, True, "ok")}}
    lines = format_finalize(good, 1.75)
    check("all-ok headline", lines[0] == "  FINALIZE survival: 3/3 OK",
          lines[0])
    check("per-session line shows the numbers",
          lines[1] == "    s0  base 12.60s  finalize 2.20s  0.17x  OK",
          lines[1])

    bad = {"finalize_ttft": {
        0: v(18.56, None, None, False, "aborted", True),
        1: v(18.86, None, None, False, "aborted", True),
        2: v(21.64, 37.71, 1.74, True, "evicted")}}
    lines = format_finalize(bad, 1.75)
    check("failure headline counts completed finalizes only",
          lines[0] == "  FINALIZE survival: 0/1 — s2 evicted", lines[0])
    check("aborted sessions are not shown", len(lines) == 2)
    check("evicted line keeps the numbers",
          lines[1] == "    s2  base 21.64s  finalize 37.71s  1.74x  EVICTED",
          lines[1])

    lines = format_finalize({"finalize_ttft": {
        0: v(None, None, None, False, "aborted", True),
        1: v(None, None, None, False, "aborted", True)}}, 1.75)
    check("all aborted", lines[0] ==
          "  FINALIZE survival: 0/0 (no finalize completed)", lines[0])

    lines = format_finalize({"finalize_ttft": {}}, 1.75)
    check("no finals", lines[0] ==
          "  FINALIZE survival: 0/0 (no finalize completed)", lines[0])


def test_summarise_ttft_extension():
    recs = []
    for s in (0, 1):
        recs.append(dict(_records(s, "main", 0, 5000, 0, 0.0), ttft=9.0))
        for i, t in ((1, 10.0), (2, 10.0)):
            recs.append(dict(_records(s, "main", i, 15000, 14800, 0.99), ttft=t))
        recs.append(dict(_records(s, "sub", 0, 40000, 0, 0.0), ttft=50.0))
        recs.append(dict(_records(s, "finalize", 0, 17000, 15000, 0.99),
                         ttft=3.0 if s == 0 else 40.0))

    out = summarise(recs, num_sessions=2, sub_windows=2, threshold=0.95)
    check("legacy finalize gate unchanged",
          out["finalize_ok"] == 2 and out["finalize_total"] == 2)
    check("one degraded finalize flagged by ttft",
          out["finalize_ttft_ok"] == 1 and out["finalize_ttft_verified"] == 2)
    check("ttft pct reported", out["finalize_ttft_pct"] == 50.0)
    check("per-session verdicts attached",
          out["finalize_ttft"][0]["ttft_ok"] is True
          and out["finalize_ttft"][1]["ttft_ratio"] == 4.0)

    legacy = summarise([_records(0, "main", 1, 10000, 9000, 0.9),
                        _records(0, "finalize", 0, 12000, 11000, 0.99)],
                       num_sessions=1, sub_windows=2, threshold=0.66)
    check("records without ttft stay unverified",
          legacy["finalize_ttft_verified"] == 0
          and legacy["finalize_ttft_ok"] == 0
          and legacy["finalize_ttft_pct"] is None)


def _hit(prompt=1000):
    return {"prompt": prompt, "cached": prompt, "cached_source": "usage",
            "ttft": 0.01, "wall": 0.01, "content": "ok", "error": None}


def _miss(prompt=1000):
    return {"prompt": prompt, "cached": 0, "cached_source": "usage",
            "ttft": 0.01, "wall": 0.01, "content": "ok", "error": None}


def _partial(prompt=1000, reuse=0.8):
    return {"prompt": prompt, "cached": int(prompt * reuse),
            "cached_source": "usage",
            "ttft": 0.01, "wall": 0.01, "content": "ok", "error": None}


def _run_session_with_fake(fake, plan, thinking=False, stop_event=None,
                           miss_threshold=None):
    """Drive run_session against a fake chat_stream.

    Returns (calls, records, kwargs): the message lists, the recorded
    records, and the extra kwargs (e.g. abort_event) each call received.
    """
    calls = []
    kwargs = []

    def _default(url, model, messages, max_tokens, timeout, api_key,
                 thinking_flag):
        return _hit()

    impl = _default if fake is None else fake

    def wrapped(url, model, messages, max_tokens, timeout, api_key,
                thinking_flag, **kw):
        calls.append(list(messages))
        kwargs.append(kw)
        return impl(url, model, messages, max_tokens, timeout, api_key,
                    thinking_flag)

    records = []
    orig = agent_sim.chat_stream
    agent_sim.chat_stream = wrapped
    try:
        plan = plan or build_session_plan(salt=7, step_tokens=5000, steps=2,
                                          sub_tokens=40000, sub_windows=1,
                                          finalize_new=2000)
        run_kwargs = {"base_url": "http://x/v1", "model": "m", "api_key": None,
                      "timeout": 5, "plan": plan, "session_id": 0,
                      "barrier": threading.Barrier(1), "max_tokens": 8,
                      "thinking": thinking, "records": records,
                      "lock": threading.Lock(), "stop_event": stop_event}
        if miss_threshold is not None:
            run_kwargs["miss_threshold"] = miss_threshold
        agent_sim.run_session(**run_kwargs)
    finally:
        agent_sim.chat_stream = orig
    return calls, records, kwargs


def _user_count(messages):
    return sum(1 for m in messages if m["role"] == "user")


def test_run_session_finalize_prefix():
    plan = build_session_plan(salt=7, step_tokens=5000, steps=2,
                              sub_tokens=40000, sub_windows=1, finalize_new=2000)
    calls, records, _kw = _run_session_with_fake(None, plan)
    system = {"role": "system", "content": SYSTEM}
    u0 = {"role": "user", "content": plan["main"][0]}
    u1 = {"role": "user", "content": plan["main"][1]}
    uf = {"role": "user", "content": plan["finalize"]}
    a = {"role": "assistant", "content": "ok"}
    check("2 main + 1 sub + 1 finalize request",
          len(calls) == 4, f"{len(calls)}")
    check("main step 0 cold", calls[0] == [system, u0])
    check("main step 1 extends prefix",
          calls[1] == [system, u0, a, u1])
    check("sub window isolated from main",
          calls[2] == [system, {"role": "user", "content": plan["subs"][0][0]}])
    check("finalize builds on the pre-sub prefix",
          calls[3] == [system, u0, a, u1, a, uf])
    finals = [r for r in records if r["phase"] == "finalize"]
    check("finalize reuse vs last main prompt",
          len(finals) == 1 and finals[0]["reuse"] == 1.0,
          f"reuse={finals[0]['reuse'] if finals else None}")


def test_run_session_thinking_flag():
    seen = []

    def fake(url, model, messages, max_tokens, timeout, api_key, thinking_flag,
             **kw):
        seen.append(thinking_flag)
        return _hit()
    plan = build_session_plan(salt=7, step_tokens=5000, steps=2,
                              sub_tokens=40000, sub_windows=1, finalize_new=2000)
    _run_session_with_fake(fake, plan, thinking=False)
    _run_session_with_fake(fake, plan, thinking=True)
    check("thinking flag forwarded to every request",
          seen == [False, False, False, False, True, True, True, True],
          f"{seen}")


def test_run_session_aborts_on_main_miss():
    plan = build_session_plan(salt=7, step_tokens=5000, steps=3,
                              sub_tokens=40000, sub_windows=1,
                              finalize_new=2000)

    def fake(url, model, messages, max_tokens, timeout, api_key, thinking_flag):
        return _miss() if _user_count(messages) == 3 else _hit()
    stop = threading.Event()
    calls, records, _kw = _run_session_with_fake(fake, plan, stop_event=stop)
    check("main continuation miss trips the stop event", stop.is_set())
    check("no requests issued after the miss", len(calls) == 3, f"{len(calls)}")
    check("only executed turns recorded", len(records) == 3, f"{len(records)}")


def test_run_session_aborts_on_finalize_miss():
    plan = build_session_plan(salt=7, step_tokens=5000, steps=2,
                              sub_tokens=40000, sub_windows=1,
                              finalize_new=2000)

    def fake(url, model, messages, max_tokens, timeout, api_key, thinking_flag):
        return _miss() if _user_count(messages) == 3 else _hit()
    stop = threading.Event()
    calls, records, _kw = _run_session_with_fake(fake, plan, stop_event=stop)
    check("finalize miss trips the stop event", stop.is_set())
    check("finalize is the last turn", len(calls) == 4, f"{len(calls)}")


def test_run_session_lower_threshold_tolerates_partial_reuse():
    plan = build_session_plan(salt=7, step_tokens=5000, steps=3,
                              sub_tokens=40000, sub_windows=1,
                              finalize_new=2000)

    def fake(url, model, messages, max_tokens, timeout, api_key, thinking_flag):
        return _partial() if _user_count(messages) == 3 else _hit()
    stop = threading.Event()
    calls, records, _kw = _run_session_with_fake(fake, plan, stop_event=stop)
    check("partial reuse below the 66% default is tolerated",
          not stop.is_set())
    check("run completes all turns", len(calls) == 5, f"{len(calls)}")


def test_run_session_strict_threshold_aborts_on_partial_reuse():
    plan = build_session_plan(salt=7, step_tokens=5000, steps=3,
                              sub_tokens=40000, sub_windows=1,
                              finalize_new=2000)

    def fake(url, model, messages, max_tokens, timeout, api_key, thinking_flag):
        return _partial() if _user_count(messages) == 3 else _hit()
    stop = threading.Event()
    calls, records, _kw = _run_session_with_fake(fake, plan, stop_event=stop,
                                                 miss_threshold=0.95)
    check("same partial reuse trips a strict threshold", stop.is_set())
    check("aborted after the partial turn", len(calls) == 3, f"{len(calls)}")


def test_run_session_hit_run_never_aborts():
    stop = threading.Event()
    plan = build_session_plan(salt=7, step_tokens=5000, steps=3,
                              sub_tokens=40000, sub_windows=1,
                              finalize_new=2000)
    calls, records, _kw = _run_session_with_fake(None, plan, stop_event=stop)
    check("hit run does not trip the stop event", not stop.is_set())
    check("hit run completes all turns", len(calls) == 5, f"{len(calls)}")
    check("hit run records every turn", len(records) == 5, f"{len(records)}")


def test_run_session_skips_when_stop_already_set():
    stop = threading.Event()
    stop.set()
    plan = build_session_plan(salt=7, step_tokens=5000, steps=2,
                              sub_tokens=40000, sub_windows=1,
                              finalize_new=2000)
    calls, records, _kw = _run_session_with_fake(None, plan, stop_event=stop)
    check("no turns run when already stopped",
          len(calls) == 0 and len(records) == 0,
          f"calls={len(calls)} records={len(records)}")


def test_run_session_shared_stop_halts_other_session():
    plan = build_session_plan(salt=7, step_tokens=5000, steps=3,
                              sub_tokens=40000, sub_windows=1,
                              finalize_new=2000)
    stop = threading.Event()

    def miss_on_main2(url, model, messages, max_tokens, timeout, api_key,
                      thinking_flag):
        return _miss() if _user_count(messages) == 3 else _hit()
    _run_session_with_fake(miss_on_main2, plan, stop_event=stop)
    check("first session tripped the stop event", stop.is_set())

    calls, records, _kw = _run_session_with_fake(None, plan, stop_event=stop)
    check("second session issued no turns once stopped",
          len(calls) == 0 and len(records) == 0,
          f"calls={len(calls)} records={len(records)}")


def test_run_session_forwards_abort_event():
    stop = threading.Event()
    plan = build_session_plan(salt=7, step_tokens=5000, steps=2,
                              sub_tokens=40000, sub_windows=1,
                              finalize_new=2000)
    calls, _records, kwargs = _run_session_with_fake(None, plan,
                                                     stop_event=stop)
    check("abort_event forwarded on every turn",
          len(kwargs) == 4
          and all(kw.get("abort_event") is stop for kw in kwargs),
          f"{len(kwargs)} calls")


def test_run_session_aborted_turn():
    plan = build_session_plan(salt=7, step_tokens=5000, steps=3,
                              sub_tokens=40000, sub_windows=1,
                              finalize_new=2000)
    stop = threading.Event()

    def fake(url, model, messages, max_tokens, timeout, api_key,
             thinking_flag):
        return {**_hit(), "aborted": True} if _user_count(messages) == 2 \
            else _hit()
    calls, records, _kw = _run_session_with_fake(fake, plan, stop_event=stop)
    check("aborted turn recorded with the flag",
          len(records) == 2 and records[1]["aborted"] is True,
          f"{len(records)} records")
    check("no further turns after the abort", len(calls) == 2,
          f"{len(calls)}")
    check("aborted turn stops the run", stop.is_set())


# ── live progress view ─────────────────────────────────────────────────


def _state(sid=0, phase="main", step=0, main_value=0.0, sub_value=0.0,
           sub_done=0, turns=0, sum_reuse=0.0, pp=None, tg=None,
           done=False, ok=None, err=None):
    return {"sid": sid, "phase": phase, "step": step,
            "main_value": main_value, "sub_value": sub_value, "sub_done": sub_done,
            "turns": turns, "sum_reuse": sum_reuse, "pp": pp, "tg": tg,
            "done": done, "ok": ok, "err": err}


def _states():
    return [
        _state(0, "main", 16, main_value=135000, turns=16,
               sum_reuse=16 * 0.99, pp=12400, tg=46),
        _state(1, "sub", main_value=150000, sub_value=40000, sub_done=1,
               pp=9100, tg=52),
        _state(2, "main", 13, main_value=128000, turns=13,
               sum_reuse=13 * 0.99, pp=7700, tg=44),
    ]


def test_bar():
    check("_bar empty", _bar(0.0, 10) == ("", "", "░" * 10))
    check("_bar half", _bar(0.5, 10) == ("█████", "", "░" * 5))
    check("_bar partial cell", _bar(0.45, 10) == ("████", "▌", "░" * 5))
    check("_bar clamps above 1", _bar(1.5, 4) == ("████", "", ""))
    check("_bar tiny fraction", _bar(0.05, 10) == ("", "▌", "░" * 9))
    joined = "".join(_bar(0.45, 10))
    check("_bar parts fill the width", len(joined) == 10)


def test_formats():
    check("fmt_ktok k", fmt_ktok(135000) == "135k")
    check("fmt_ktok k2", fmt_ktok(152000) == "152k")
    check("fmt_ktok small", fmt_ktok(999) == "999")
    check("fmt_ktok M", fmt_ktok(1200000) == "1.2M")
    check("fmt_tps k", fmt_tps(12400) == "12.4k")
    check("fmt_tps plain", fmt_tps(46.4) == "46")
    check("fmt_tps none", fmt_tps(None) == "–")
    check("fmt_elapsed s", fmt_elapsed(42.3) == "42.3s")
    check("fmt_elapsed min", fmt_elapsed(83.0) == "1m23s")
    check("fmt_elapsed hour", fmt_elapsed(3661.7) == "1h01m")
    check("fmt_elapsed none", fmt_elapsed(None) == "–")


def test_fmt_record():
    rec = {"session": 2, "phase": "main", "step": 7,
           "prompt": 100_000, "cached": 99_500,
           "cached_source": "device", "reuse": 0.995, "ttft": 0.42,
           "wall": 1.2, "error": None}
    line = _fmt_record(rec)
    check("_fmt_record clean line",
          line == ("  s2 main      step  7  prompt= 100,000  "
                   "cached=  99,500  reuse= 99.5%  ttft=0.42s  err=–"))
    rec2 = dict(rec, error="boom", reuse_class="hit", cached=None, prompt=None)
    line2 = _fmt_record(rec2)
    check("_fmt_record err + reuse_class + None shown",
          line2 == ("  s2 main      step  7  prompt=       –  "
                    "cached=       –  reuse= 99.5%  hit  ttft=0.42s  err=boom"))


def test_fmt_record_aborted():
    rec = {"session": 2, "phase": "main", "step": 7,
           "prompt": None, "cached": None, "cached_source": "none",
           "reuse": 0.0, "ttft": 0.42, "wall": 1.2, "error": None,
           "aborted": True}
    line = _fmt_record(rec)
    check("_fmt_record aborted marker",
          line == ("  s2 main      step  7  prompt=       –  "
                   "cached=       –  reuse=  0.0%  ttft=0.42s  err=–  ABORTED"))


def test_render_frame_side_by_side():
    frame = render_frame(_states(), main_size=150000, sub_tokens=40000,
                         sub_total=2, steps_total=30, elapsed=72.0,
                         term_w=160, color=False)
    check("no ANSI in plain frame", "\x1b" not in frame)
    lines = frame.split("\n")
    check("header + 4 row lines", len(lines) == 5)
    check("all lines equal width", len({len(l) for l in lines}) == 1)
    check("header has elapsed", "01:12" in lines[0])
    check("main bars side by side",
          all(f"s{i}  main" in lines[1] for i in (0, 1, 2)))
    check("main bar value", "135k/150k" in lines[1])
    check("full main bar at 150k", "150k/150k" in lines[1])
    check("sub bar row", "░░░░░░░░░░ 0/2" in lines[2]
          and "█████░░░░░ 1/2 · cold" in lines[2])
    check("phase row per session",
          "main 16/30 · hit 99%" in lines[3]
          and "sub 1/2 · cold" in lines[3]
          and "main 13/30 · hit 99%" in lines[3])
    check("tps row", "pp 12.4k tps · tg 46 tps" in lines[4]
          and "pp 9.1k tps · tg 52 tps" in lines[4])


def test_render_frame_stacked():
    frame = render_frame(_states(), main_size=150000, sub_tokens=40000,
                         sub_total=2, steps_total=30, elapsed=72.0,
                         term_w=80, color=False)
    lines = frame.split("\n")
    check("stacked: header + 3 blocks + 2 separators", len(lines) == 15)
    check("stacked block order", lines[1].startswith("s0")
          and lines[6].startswith("s1") and lines[11].startswith("s2"))
    check("blank separators between blocks", lines[5].strip() == ""
          and lines[10].strip() == "")


def test_render_frame_done_and_err():
    states = [_state(0, done=True, ok=True, main_value=152000),
              _state(1, done=True, ok=False, main_value=152000),
              _state(2, phase="main", err="boom", main_value=135000)]
    frame = render_frame(states, main_size=150000, sub_tokens=40000,
                         sub_total=2, steps_total=30, elapsed=10.0,
                         term_w=160, color=False)
    check("done ok mark", "done ✓" in frame)
    check("done lost mark", "done ✗" in frame)
    check("error mark", "err ✗" in frame)


def test_render_frame_color():
    frame = render_frame([_state(0, main_value=135000)], main_size=150000,
                         sub_tokens=40000, sub_total=2, steps_total=30,
                         elapsed=1.0, term_w=160, color=True)
    check("green bar escape", "\x1b[32m" in frame)
    check("resets color", "\x1b[0m" in frame)


def _res(prompt, cached, ttft, wall, completion, error=None):
    return {"prompt": prompt, "cached": cached, "ttft": ttft, "wall": wall,
            "completion": completion, "error": error}


def test_live_view():
    v = LiveView(sessions=2, main_size=150000, sub_tokens=40000, sub_total=2)
    v.turn_start(0, "main", 0, 20000)
    s = v.snapshot()[0]
    check("mid-turn main grows (tpc default 4.0)",
          abs(s["main_value"] - 5000) < 0.01)

    v.delta(0, 64)
    v.turn_end(0, _res(5000, 0, 0.4, 0.5, 64), 0.0)
    v.turn_start(0, "main", 1, 20000)
    v.turn_end(0, _res(10000, 9900, 0.05, 0.2, 60), 0.99)
    s = v.snapshot()[0]
    check("main_value tracks last prompt", s["main_value"] == 10000.0)
    check("pp = (prompt-cached)/ttft", abs(s["pp"] - 2000.0) < 1e-9)
    check("tg = completion/(wall-ttft)", abs(s["tg"] - 400.0) < 1e-9)
    check("continuation turns counted", s["turns"] == 1
          and abs(s["sum_reuse"] - 0.99) < 1e-9)
    check("tpc calibrated from cold turn (chars/token)",
          3.9 < v.tpc < 4.1)

    v.turn_start(0, "sub", 0, 160000)
    v.turn_end(0, _res(40000, 0, 1.0, 1.2, 32), 0.0)
    s = v.snapshot()[0]
    check("sub window counted", s["sub_done"] == 1 and s["sub_value"] == 40000.0)
    check("main frozen during subs", s["main_value"] == 10000.0)

    v.turn_start(0, "sub", 1, 120000)
    s = v.snapshot()[0]
    check("sub mid-flight stays within scale",
          40000.0 <= s["sub_value"] < 80000.0)

    v.turn_start(0, "finalize", 0, 4000)
    v.turn_end(0, _res(10400, 10200, 0.06, 0.3, 60), 0.98)
    s = v.snapshot()[0]
    check("finalize grows main bar", s["main_value"] == 10400.0)

    v.finish(0, True)
    s = v.snapshot()[0]
    check("finish flags done/ok", s["done"] is True and s["ok"] is True)
    check("second session untouched",
          v.snapshot()[1]["phase"] == "main"
          and v.snapshot()[1]["main_value"] == 0.0)


def test_chat_stream_completion_tokens():
    lines = [
        b"data: " + json.dumps(
            {"choices": [{"delta": {"content": "hel"}}]}).encode(),
        b"data: " + json.dumps(
            {"choices": [{"delta": {"content": "lo!"}}]}).encode(),
        b"data: " + json.dumps({
            "choices": [],
            "usage": {"prompt_tokens": 1200, "completion_tokens": 5,
                      "prompt_tokens_details": {"cached_tokens": 1000}},
        }).encode(),
        b"data: [DONE]",
    ]

    class FakeResp:
        closed = 0

        def raise_for_status(self):
            pass

        def close(self):
            type(self).closed += 1

        def iter_lines(self):
            yield from lines

    orig_post = agent_sim.requests.post
    agent_sim.requests.post = lambda *a, **k: FakeResp()
    try:
        deltas = []
        res = agent_sim.chat_stream("http://x/v1/chat/completions", "m",
                                    [{"role": "user", "content": "hi"}], 64,
                                    5, None, False, on_delta=deltas.append)
    finally:
        agent_sim.requests.post = orig_post
    check("content assembled", res["content"] == "hello!")
    check("completion_tokens captured", res["completion"] == 5)
    check("prompt from usage", res["prompt"] == 1200)
    check("cached from usage details", res["cached"] == 1000)
    check("ttft measured", res["ttft"] is not None)
    check("on_delta per content chunk", deltas == [3, 3])
    check("streaming response closed", FakeResp.closed == 1)


def test_chat_stream_aborts_pre_post():
    abort = threading.Event()
    abort.set()
    posted = []

    class FakeResp:
        def raise_for_status(self):
            pass

        def close(self):
            pass

        def iter_lines(self):
            return iter([])

    orig_post = agent_sim.requests.post
    agent_sim.requests.post = lambda *a, **k: posted.append(1) or FakeResp()
    try:
        res = agent_sim.chat_stream("http://x/v1/chat/completions", "m",
                                    [{"role": "user", "content": "hi"}], 64,
                                    5, None, False, abort_event=abort)
    finally:
        agent_sim.requests.post = orig_post
    check("no request posted when abort pre-set", not posted)
    check("aborted flag set", res["aborted"] is True)
    check("nothing captured", res["prompt"] is None
          and res["cached"] is None and res["content"] == "")


def test_chat_stream_aborts_mid_stream():
    abort = threading.Event()

    class FakeResp:
        closed = 0

        def raise_for_status(self):
            pass

        def close(self):
            type(self).closed += 1

        def iter_lines(self):
            yield b"data: " + json.dumps(
                {"choices": [{"delta": {"content": "hel"}}]}).encode()
            abort.set()
            yield b"data: " + json.dumps(
                {"choices": [{"delta": {"content": "lo!"}}]}).encode()

    orig_post = agent_sim.requests.post
    agent_sim.requests.post = lambda *a, **k: FakeResp()
    try:
        res = agent_sim.chat_stream("http://x/v1/chat/completions", "m",
                                    [{"role": "user", "content": "hi"}], 64,
                                    5, None, False, abort_event=abort)
    finally:
        agent_sim.requests.post = orig_post
    check("aborted mid-stream", res["aborted"] is True)
    check("only pre-abort content captured", res["content"] == "hel")
    check("no usage seen", res["prompt"] is None and res["cached"] is None)
    check("connection closed on abort", FakeResp.closed == 1)


def test_viz_enabled():
    check("default: only on a tty", viz_enabled(True) and not viz_enabled(False))
    check("force wins over piped stdout", viz_enabled(False, force=True))
    check("no-viz wins over tty", not viz_enabled(True, disable=True))
    check("no-viz wins over force", not viz_enabled(True, force=True, disable=True))


def main():
    test_make_body_deterministic_and_unique()
    test_build_session_plan()
    test_plans_unique_across_sessions()
    test_compute_reuse()
    test_validate_sizes()
    test_summarise()
    test_run_session_finalize_prefix()
    test_run_session_thinking_flag()
    test_run_session_aborts_on_main_miss()
    test_run_session_aborts_on_finalize_miss()
    test_run_session_lower_threshold_tolerates_partial_reuse()
    test_run_session_strict_threshold_aborts_on_partial_reuse()
    test_run_session_hit_run_never_aborts()
    test_run_session_skips_when_stop_already_set()
    test_run_session_shared_stop_halts_other_session()
    test_run_session_forwards_abort_event()
    test_run_session_aborted_turn()
    test_viz_enabled()
    test_bar()
    test_formats()
    test_fmt_record()
    test_fmt_record_aborted()
    test_render_frame_side_by_side()
    test_render_frame_stacked()
    test_render_frame_done_and_err()
    test_render_frame_color()
    test_live_view()
    test_chat_stream_completion_tokens()
    test_chat_stream_aborts_pre_post()
    test_chat_stream_aborts_mid_stream()
    test_summarise_excludes_aborted()
    test_finalize_ttft_verdict()
    test_label_finalize()
    test_format_finalize()
    test_summarise_ttft_extension()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("all tests passed")


if __name__ == "__main__":
    main()
