#!/usr/bin/env python3
"""Ninfer request-log parsing and cache-reuse classification.

The OpenAI-compatible API surface (`usage.prompt_tokens_details.cached_tokens`)
cannot distinguish a device cache hit from a RAM-tier restore — both report
the same cached token count. The ground truth for "did we really recompute
tokens" lives only in ninfer's box-side request log (`--request-log-jsonl`):

  - `result.computed_prefill_tokens` — tokens actually computed for this
    request (the true "re-prefill" cost)
  - `result.prefix_cache_hit_tokens` — tokens served from cache (device or RAM)
  - `result.prefix_reuse_path` — `root` = nothing reused (full re-prefill),
    `private_turn_closure`/`private_endpoint`/... = a cached prefix was used

This module parses that log and re-annotates agent-sim records with the true
reuse class, so the benchmark can verify the "never re-prefill a processed
prefix" requirement (except when the RAM tier itself evicts).
"""
import json


def parse_ninfer_log(path):
    """Parse a ninfer request-log JSONL; return request_done entries in order.

    Each entry keeps the request id plus the three reuse-relevant result
    fields. Non-request_done lines (throughput, server_start) are skipped.
    """
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") != "request_done":
                continue
            result = row.get("result") or {}
            entries.append({
                "request_id": (row.get("request") or {}).get("request_id"),
                "prompt": result.get("prompt_tokens"),
                "computed_prefill_tokens": result.get("computed_prefill_tokens"),
                "prefix_cache_hit_tokens": result.get("prefix_cache_hit_tokens"),
                "prefix_reuse_path": result.get("prefix_reuse_path"),
                "ts": row.get("timestamp_unix_ms"),
            })
    return entries


def classify_reuse(reuse, path):
    """Classify a request as miss / hit / partial from its reuse fraction.

    `reuse` is cached_tokens / previous-prompt-tokens (the fraction of the
    *already-processed* prefix that was served from cache — 1.0 = the whole
    prior context came back without recompute, whether from the device pool
    or the RAM tier). `path == "root"` marks a hard full re-prefill.
    """
    if reuse is None:
        return "unknown"
    if path == "root" or reuse < 0.10:
        return "miss"
    if reuse >= 0.70:
        return "hit"
    return "partial"


def reannotate_records(records, entries):
    """Zip agent-sim records with request-log entries by completion order.

    Ninfer appends request_done entries in completion order, matching agent-sim
    record order, but the log may also hold *earlier* traffic (other runs,
    probes). When the log has more entries than records, the run's requests are
    assumed to be the most recent traffic — the trailing slice is used, and the
    surplus is returned as `extra_entries`. Each annotated record gets
    `computed`, `reuse_path` and `reuse_class` from its log entry, plus a
    `reuse` fraction recomputed as cached / previous-record prompt when the
    record lacks one. Records beyond the log length are returned unmatched.
    """
    out = []
    unmatched = []
    extra = []
    if len(entries) > len(records):
        extra = entries[:len(entries) - len(records)]
        entries = entries[len(entries) - len(records):]
    for idx, rec in enumerate(records):
        if idx >= len(entries):
            unmatched.append(rec)
            continue
        e = entries[idx]
        prev_prompt = records[idx - 1]["prompt"] if idx > 0 else 0
        reuse = rec.get("reuse")
        if reuse is None and prev_prompt:
            reuse = (e.get("prefix_cache_hit_tokens") or 0) / prev_prompt
        out.append({
            **rec,
            "reuse": reuse if reuse is not None else 0.0,
            "computed": e.get("computed_prefill_tokens"),
            "reuse_path": e.get("prefix_reuse_path"),
            "reuse_class": classify_reuse(reuse, e.get("prefix_reuse_path")),
        })
    return out, unmatched, extra
