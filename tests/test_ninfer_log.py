#!/usr/bin/env python3
"""Tests for ninfer request-log parsing in agent-sim.

The OpenAI API only exposes usage.prompt_tokens_details.cached_tokens, which
counts device cache hits and RAM-tier restores together. The ground truth —
computed_prefill_tokens, prefix_cache_hit_tokens and prefix_reuse_path — is
only in ninfer's box-side request log (--request-log-jsonl). These tests
verify the parser and the miss/hit/restore classification.

Run:  uv run python tests/test_ninfer_log.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cache_pressure.ninfer_log import (classify_reuse, parse_ninfer_log,
                                       reannotate_records)

FAILURES = []


def check(name, cond, detail=""):
    status = "OK  " if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def _entry(req_id, prompt, computed, cached, path):
    return {
        "event": "request_done",
        "request": {"request_id": req_id},
        "result": {
            "prompt_tokens": prompt,
            "computed_prefill_tokens": computed,
            "prefix_cache_hit_tokens": cached,
            "prefix_reuse_path": path,
        },
    }


def _throughput(host_kv_bytes=0):
    return {"event": "throughput",
            "context_cache": {"occupancy": {"host_kv_bytes": host_kv_bytes}}}


def test_parse_ninfer_log():
    lines = [
        json.dumps(_entry(2, 4643, 4643, 0, "root")),
        json.dumps(_throughput(500_000_000)),
        json.dumps(_entry(3, 9322, 4686, 4636, "private_turn_closure")),
        json.dumps({"event": "server_start"}),
    ]
    with open("/tmp/ninfer-log-test.jsonl", "w") as f:
        f.write("\n".join(lines) + "\n")
    entries = parse_ninfer_log("/tmp/ninfer-log-test.jsonl")
    check("only request_done entries parsed", len(entries) == 2,
          f"got {len(entries)}")
    check("request_id preserved", entries[0]["request_id"] == 2)
    check("fields extracted",
          entries[1]["computed_prefill_tokens"] == 4686 and
          entries[1]["prefix_reuse_path"] == "private_turn_closure")


def test_classify_reuse():
    # true miss: nothing reused, root path
    check("miss = reuse 0, root path",
          classify_reuse(0.0, "root") == "miss")
    # full continuation hit: whole previous prompt served from cache
    check("hit = reuse ~1.0, turn-closure path",
          classify_reuse(0.99, "private_turn_closure") == "hit")
    # offload hit looks identical to a device hit at the API level
    check("API cannot tell device-hit from offload-hit (same fields)",
          classify_reuse(0.99, "private_turn_closure")
          == classify_reuse(0.99, "private_turn_closure"))
    # partial reuse (some of the prefix recomputed) is neither
    check("partial reuse classified",
          classify_reuse(0.25, "private_turn_closure") == "partial")


def test_reannotate_records():
    records = [
        {"session": 0, "phase": "main", "step": 1, "prompt": 4643,
         "cached": 0, "reuse": 0.0},
        {"session": 0, "phase": "main", "step": 2, "prompt": 9322,
         "cached": 4636, "reuse": 0.99},
    ]
    entries = [
        {"request_id": 2, "prompt": 4643, "computed_prefill_tokens": 4643,
         "prefix_cache_hit_tokens": 0, "prefix_reuse_path": "root"},
        {"request_id": 3, "prompt": 9322, "computed_prefill_tokens": 4686,
         "prefix_cache_hit_tokens": 4636, "prefix_reuse_path": "private_turn_closure"},
    ]
    out, unmatched, extra = reannotate_records(records, entries)
    check("all records matched", len(out) == 2 and len(unmatched) == 0)
    check("record 0 flagged as miss",
          out[0]["reuse_class"] == "miss" and out[0]["reuse_path"] == "root")
    check("record 1 flagged as hit",
          out[1]["reuse_class"] == "hit")
    check("computed tokens recorded", out[1]["computed"] == 4686)

    # a log with fewer entries than records leaves the rest unmatched
    out2, unmatched2, extra2 = reannotate_records(records, entries[:1])
    check("short log leaves trailing unmatched",
          len(out2) == 1 and len(unmatched2) == 1)

    # a log with earlier (pre-run) traffic aligns records with the trailing
    # slice and reports the surplus as extra
    log = [{"request_id": 1, "prompt": 1000,
            "computed_prefill_tokens": 1000, "prefix_cache_hit_tokens": 0,
            "prefix_reuse_path": "root"}] + entries
    out3, unmatched3, extra3 = reannotate_records(records, log)
    check("earlier log traffic aligned to trailing slice",
          len(out3) == 2 and len(unmatched3) == 0 and len(extra3) == 1)
    check("trailing slice matched correct requests",
          out3[1]["reuse_path"] == "private_turn_closure")


def main():
    test_parse_ninfer_log()
    test_classify_reuse()
    test_reannotate_records()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        sys.exit(1)
    print("all tests passed")


if __name__ == "__main__":
    main()
