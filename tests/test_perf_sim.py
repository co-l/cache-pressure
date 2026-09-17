#!/usr/bin/env python3
"""Tests for the perf-sim speed-vs-context benchmark.

Verifies (pure logic, no network):
  - make_lorem is deterministic per seed, unique across seeds, exact length.
  - plan_steps counts the full steps that fit under max context (replies
    included in the growth, no partial final step).
  - build_steps produces unique per-step chunks of the right size.
  - pp_speed / tg_speed compute the headline speeds (cold, incremental,
    and degenerate cases).
  - run_perf assembles messages as a stable prefix: each request is the
    previous messages + the new user chunk + the previous assistant reply,
    forwards the output budget and thinking flag, and stops on error.
  - summarise collects the per-level arrays.
  - plain row formatting.

Run:  uv run python tests/test_perf_sim.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cache_pressure.perf_sim as perf_sim
from cache_pressure.agent_sim import FALLBACK_REPLY, SYSTEM
from cache_pressure.perf_sim import (OVERHEAD, build_steps, fmt_level_table,
                                     make_lorem, plan_steps, pp_speed,
                                     summarise, tg_speed, _fmt_row)

FAILURES = []


def check(name, cond, detail=""):
    status = "OK  " if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def test_make_lorem():
    a1 = make_lorem(16000, seed=11)
    a2 = make_lorem(16000, seed=11)
    b = make_lorem(16000, seed=12)
    check("make_lorem deterministic per seed", a1 == a2)
    check("make_lorem unique across seeds", a1 != b)
    check("make_lorem exact length", len(a1) == 16000, f"{len(a1)}")
    check("make_lorem is real lorem ipsum",
          "dolor sit amet" in a1 and "consectetur" in a1)


def test_plan_steps():
    check("100k max -> 9 full steps",
          plan_steps(10000, 256, 100000) == 9)
    check("131072 max -> 12 full steps",
          plan_steps(10000, 256, 131072) == 12)
    check("steps grow with max context",
          plan_steps(10000, 256, 200000) > plan_steps(10000, 256, 100000))
    check("replies count toward the growth",
          plan_steps(10000, 0, 100000)
          >= plan_steps(10000, 4096, 100000))
    check("tiny max -> 0 steps", plan_steps(10000, 256, 5000) == 0)


def test_plan_steps_never_overflows():
    step, out, mx = 10000, 256, 131072
    n = plan_steps(step, out, mx)
    for i in range(1, n + 1):
        fit = OVERHEAD + i * step + (i - 1) * out
        check(f"step {i} fits under max context", fit <= mx,
              f"{fit} > {mx}")


def test_build_steps():
    steps = build_steps(4, 10000, salt=7)
    check("4 chunks", len(steps) == 4)
    check("chunk size is 4 chars/token",
          all(len(c) == 40000 for c in steps))
    check("unique per step", len(set(steps)) == 4)
    other = build_steps(4, 10000, salt=8)
    check("unique across salts", other[0] != steps[0])


def test_pp_speed():
    check("incremental pp = uncached / ttft",
          abs(pp_speed(110000, 100000, 0.5) - 20000.0) < 1e-9)
    check("cold pp = full prompt / ttft (cached None)",
          abs(pp_speed(10000, None, 2.0) - 5000.0) < 1e-9)
    check("cold pp = full prompt / ttft (cached 0)",
          abs(pp_speed(10000, 0, 2.0) - 5000.0) < 1e-9)
    check("over-reported cached falls back to full prompt",
          abs(pp_speed(10000, 12000, 1.0) - 10000.0) < 1e-9)
    check("no ttft -> None", pp_speed(10000, 0, None) is None)
    check("no prompt -> None", pp_speed(None, 0, 1.0) is None)


def test_tg_speed():
    check("tg = completion / (wall - ttft)",
          abs(tg_speed(256, 5.0, 1.0) - 64.0) < 1e-9)
    check("no completion -> None", tg_speed(None, 5.0, 1.0) is None)
    check("no ttft -> None", tg_speed(256, 5.0, None) is None)
    check("wall <= ttft -> None", tg_speed(256, 1.0, 1.0) is None)


def test_summarise():
    recs = [
        {"step": 0, "prompt": 10000, "ttft": 0.4, "pp": 25000.0,
         "tg": 64.0, "completion": 256, "error": None},
        {"step": 1, "prompt": 20000, "ttft": 0.5, "pp": 20000.0,
         "tg": 58.0, "completion": 256, "error": None},
        {"step": 2, "prompt": 30000, "ttft": None, "pp": None,
         "tg": None, "completion": None, "error": "boom"},
    ]
    out = summarise(recs, runs=1, elapsed=42.0)
    check("runs recorded", out["runs"] == 1)
    check("steps counted", out["steps"] == 3)
    check("completed counted", out["completed"] == 2)
    check("context array from completed steps",
          out["context"] == [10000, 20000])
    check("pp array", out["pp"] == [25000.0, 20000.0])
    check("tg array", out["tg"] == [64.0, 58.0])
    check("elapsed passed through", out["elapsed"] == 42.0)
    check("elapsed defaults to None", summarise([], runs=1)["elapsed"] is None)


def test_summarise_multi_run():
    recs = []
    for run in (0, 1):
        recs.append({"step": 0, "prompt": 10000, "run": run, "ttft": 0.4,
                     "pp": 25000.0, "tg": 64.0, "completion": 256,
                     "error": None})
        recs.append({"step": 1, "prompt": 20000, "run": run, "ttft": 0.5,
                     "pp": 20000.0, "tg": 58.0, "completion": 256,
                     "error": None})
        recs.append({"step": 2, "prompt": 30000, "run": run, "ttft": None,
                     "pp": None, "tg": None, "completion": None,
                     "error": "boom"})
    out = summarise(recs, runs=2, elapsed=90.0)
    check("runs recorded", out["runs"] == 2)
    check("steps counted across runs", out["steps"] == 6)
    check("completed counted across runs", out["completed"] == 4)
    check("context averaged per level", out["context"] == [10000.0, 20000.0])
    check("pp averaged per level", out["pp"] == [25000.0, 20000.0])
    check("tg averaged per level", out["tg"] == [64.0, 58.0])
    check("elapsed passed through", out["elapsed"] == 90.0)

    partial = [
        {"step": 0, "prompt": 10000, "run": 0, "pp": 25000.0,
         "tg": 64.0, "error": None},
        {"step": 0, "prompt": 10000, "run": 1, "pp": 26000.0,
         "tg": 60.0, "error": None},
        {"step": 1, "prompt": 20000, "run": 0, "pp": 20000.0,
         "tg": 58.0, "error": None},
        {"step": 1, "prompt": 20000, "run": 1, "pp": None,
         "tg": None, "error": "boom"},
    ]
    out2 = summarise(partial, runs=2)
    check("levels average across runs", out2["pp"][0] == 25500.0,
          f"{out2['pp']}")
    check("failed run excluded from the level average",
          out2["pp"][1] == 20000.0, f"{out2['pp']}")


def test_fmt_row():
    rec = {"step": 1, "prompt": 110000, "cached": 100000, "ttft": 0.5,
           "wall": 5.0, "completion": 256, "pp": 20000.0, "tg": 57.6,
           "error": None}
    check("clean row",
          _fmt_row(1, rec) ==
          ("  step  2  ctx=   110k  ttft=  0.50s  pp=  20,000 tps  "
           "tg=    58 tps  out=256  err=–"))
    bad = {"step": 0, "prompt": None, "cached": None, "ttft": None,
           "wall": None, "completion": None, "pp": None, "tg": None,
           "error": "boom"}
    check("degenerate row shows dashes and the error",
          _fmt_row(0, bad) ==
          ("  step  1  ctx=      –  ttft=      –  pp=       – tps  "
           "tg=     – tps  out=–  err=boom"))


def _hit(prompt=1000):
    return {"prompt": prompt, "cached": 0, "cached_source": "usage",
            "ttft": 0.4, "wall": 0.5, "content": "reply", "completion": 256,
            "error": None, "aborted": False}


def _run_perf_with_fake(fake, chunks, output_tokens=256, thinking=False):
    """Drive run_perf against a fake chat_stream.

    Returns (calls, records): the message lists and the recorded records.
    """
    calls = []

    def _default(url, model, messages, max_tokens, timeout, api_key,
                 thinking_flag, **kw):
        return _hit(len(messages) * 1000)

    impl = _default if fake is None else fake

    def wrapped(url, model, messages, max_tokens, timeout, api_key,
                thinking_flag, **kw):
        calls.append(list(messages))
        return impl(url, model, messages, max_tokens, timeout, api_key,
                    thinking_flag)

    records = []
    orig = perf_sim.chat_stream
    perf_sim.chat_stream = wrapped
    try:
        perf_sim.run_perf(base_url="http://x/v1", model="m", api_key=None,
                          timeout=5, chunks=chunks,
                          output_tokens=output_tokens, thinking=thinking,
                          records=records)
    finally:
        perf_sim.chat_stream = orig
    return calls, records


def test_run_perf_stable_prefix():
    chunks = build_steps(3, 1000, salt=1)
    calls, records = _run_perf_with_fake(None, chunks)
    system = {"role": "system", "content": SYSTEM}
    u0 = {"role": "user", "content": chunks[0]}
    u1 = {"role": "user", "content": chunks[1]}
    u2 = {"role": "user", "content": chunks[2]}
    a = {"role": "assistant", "content": "reply"}
    check("3 requests issued", len(calls) == 3, f"{len(calls)}")
    check("step 0 is cold (system + first chunk)", calls[0] == [system, u0])
    check("step 1 replays the previous reply",
          calls[1] == [system, u0, a, u1])
    check("step 2 extends the stable prefix",
          calls[2] == [system, u0, a, u1, a, u2])
    check("one record per step", len(records) == 3, f"{len(records)}")
    r0 = records[0]
    check("cold step pp = prompt / ttft",
          r0["pp"] is not None
          and abs(r0["pp"] - r0["prompt"] / 0.4) < 1e-9)
    check("tg = completion / (wall - ttft)",
          r0["tg"] is not None
          and abs(r0["tg"] - 256 / (0.5 - 0.4)) < 1e-9)
    check("cached tokens recorded", r0["cached"] == 0
          and r0["cached_source"] == "usage")
    check("run index recorded on every record",
          all(r["run"] == 0 for r in records), f"{records}")


def test_run_perf_fallback_reply():
    def fake(url, model, messages, max_tokens, timeout, api_key,
             thinking_flag, **kw):
        return {**_hit(), "content": ""}
    chunks = build_steps(2, 1000, salt=1)
    calls, _records = _run_perf_with_fake(fake, chunks)
    system = {"role": "system", "content": SYSTEM}
    check("empty reply replaced by the fallback",
          calls[1] == [system,
                       {"role": "user", "content": chunks[0]},
                       {"role": "assistant", "content": FALLBACK_REPLY},
                       {"role": "user", "content": chunks[1]}],
          f"{calls[1] if len(calls) > 1 else 'only one call'}")


def test_run_perf_max_tokens_forwarded():
    seen = []

    def fake(url, model, messages, max_tokens, timeout, api_key,
             thinking_flag, **kw):
        seen.append(max_tokens)
        return _hit()
    chunks = build_steps(2, 1000, salt=1)
    _run_perf_with_fake(fake, chunks, output_tokens=128)
    check("output budget forwarded on every step", seen == [128, 128],
          f"{seen}")


def test_run_perf_thinking_flag():
    seen = []

    def fake(url, model, messages, max_tokens, timeout, api_key,
             thinking_flag, **kw):
        seen.append(thinking_flag)
        return _hit()
    chunks = build_steps(2, 1000, salt=1)
    _run_perf_with_fake(fake, chunks, thinking=False)
    _run_perf_with_fake(fake, chunks, thinking=True)
    check("thinking flag forwarded on every step",
          seen == [False, False, True, True], f"{seen}")


def test_run_perf_stops_on_error():
    def fake(url, model, messages, max_tokens, timeout, api_key,
             thinking_flag, **kw):
        return {**_hit(), "error": "boom"}
    chunks = build_steps(3, 1000, salt=1)
    calls, records = _run_perf_with_fake(fake, chunks)
    check("stops after the failed step", len(calls) == 1, f"{len(calls)}")
    check("failed step recorded with the error",
          len(records) == 1 and records[0]["error"] == "boom",
          f"{records}")


def test_fmt_level_table():
    recs = [
        {"step": 0, "run": 0, "prompt": 10000, "pp": 20000.0,
         "tg": 64.0, "error": None},
        {"step": 0, "run": 1, "prompt": 10100, "pp": 22000.0,
         "tg": 60.0, "error": None},
        {"step": 1, "run": 0, "prompt": 20000, "pp": 10000.0,
         "tg": 50.0, "error": None},
        {"step": 1, "run": 1, "prompt": 20000, "pp": None,
         "tg": None, "error": "boom"},
    ]
    lines = fmt_level_table(recs)
    check("header",
          lines[0] == "  level       ctx          pp          tg",
          f"{lines[0]!r}")
    check("level 1 averaged across runs",
          lines[1] == "      1       10k      21,000          62",
          f"{lines[1]!r}")
    check("level 2 averages only the completed runs",
          lines[2] == "      2       20k      10,000          50",
          f"{lines[2]!r}")
    check("empty records -> header only",
          fmt_level_table([]) == ["  level       ctx          pp          tg"])


def test_main_rejects_bad_flags():
    for bad in ("--step-tokens", "--output-tokens", "--runs"):
        try:
            perf_sim.main(["--base-url", "http://x/v1", "--model", "m",
                           "--max-context", "100000", bad, "0"])
            check(f"main rejects {bad} 0", False, "no SystemExit")
        except SystemExit as e:
            check(f"main rejects {bad} 0", e.code == 2, f"code={e.code}")


def test_main_rejects_empty_plan():
    try:
        perf_sim.main(["--base-url", "http://x/v1", "--model", "m",
                       "--max-context", "5000"])
        check("main exits when no step fits", False, "no SystemExit")
    except SystemExit as e:
        check("main exits when no step fits", e.code == 1, f"code={e.code}")


def main():
    test_make_lorem()
    test_plan_steps()
    test_plan_steps_never_overflows()
    test_build_steps()
    test_pp_speed()
    test_tg_speed()
    test_summarise()
    test_summarise_multi_run()
    test_fmt_row()
    test_fmt_level_table()
    test_run_perf_stable_prefix()
    test_run_perf_fallback_reply()
    test_run_perf_max_tokens_forwarded()
    test_run_perf_thinking_flag()
    test_run_perf_stops_on_error()
    test_main_rejects_bad_flags()
    test_main_rejects_empty_plan()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("all tests passed")


if __name__ == "__main__":
    main()
