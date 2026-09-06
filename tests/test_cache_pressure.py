#!/usr/bin/env python3
"""Tests for the cache-pressure benchmark (cache_pressure).

Verifies:
  - verify_phase stops after the first MISS: the reverse-order verify walks
    back from newest to oldest, and under LRU everything older than the first
    miss is evicted by construction — so remaining probes are skipped.
  - summarise reflects the truncated verify rows (retained hits + first miss).

Run:  python3 tests/test_cache_pressure.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
from cache_pressure import summarise, verify_phase

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


def main():
    test_verify_stops_after_first_miss()
    print(f"\n{len(FAILURES)} failure(s)")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
