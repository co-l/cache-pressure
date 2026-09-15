#!/usr/bin/env python3
"""Tests for the needle-in-haystack harness (cache_pressure.needle).

Verifies:
  - chat_stream with thinking=False (the argparse default) must NOT send
    chat_template_kwargs: ninfer rejects any chat_template_kwargs.thinking
    with chat_template_option_not_supported, so a default-off run must not
    include it (regression: the guard was `is not None`, and argparse's
    store_true default is False, so every request shipped
    {"thinking": false} and 400'd).
  - thinking=True still forwards the kwarg.
  - content + usage are parsed from the streamed response.

Run:  uv run python tests/test_needle.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import cache_pressure.needle as needle

FAILURES = []


def check(name, cond, detail=""):
    status = "OK  " if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


class FakeResponse:
    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        pass

    def iter_lines(self):
        for line in self._lines:
            yield line


def fake_stream(payload_holder):
    def post(url, **kwargs):
        payload_holder.append(kwargs.get("json") or {})
        content = {"choices": [{"delta": {"content": "plover-1234"}}]}
        usage = {"usage": {"prompt_tokens": 50000, "completion_tokens": 4}}
        lines = [
            b"data: " + json.dumps(content).encode(),
            b"data: " + json.dumps(usage).encode(),
            b"data: [DONE]",
        ]
        return FakeResponse(lines)
    return post


def test_thinking_false_omits_chat_template_kwargs():
    seen = []
    original = needle.requests.post
    needle.requests.post = fake_stream(seen)
    try:
        text, full, prompt, comp, ttft, e2e, err = needle.chat_stream(
            "http://x/v1/chat/completions", "qwen3.8-27b",
            [{"role": "user", "content": "hi"}], 128, 60, None, False)
    finally:
        needle.requests.post = original
    check("thinking=False: request succeeds",
          err is None and "plover-1234" in text, f"err={err} text={text!r}")
    check("thinking=False: no chat_template_kwargs in payload",
          seen and "chat_template_kwargs" not in seen[0], f"payload={seen[0] if seen else None}")
    check("thinking=False: usage still parsed",
          prompt == 50000 and comp == 4, f"prompt={prompt} comp={comp}")


def test_thinking_true_forwards_kwarg():
    seen = []
    original = needle.requests.post
    needle.requests.post = fake_stream(seen)
    try:
        needle.chat_stream("http://x/v1/chat/completions", "qwen3.8-27b",
                           [{"role": "user", "content": "hi"}], 128, 60, None, True)
    finally:
        needle.requests.post = original
    check("thinking=True: chat_template_kwargs.thinking forwarded",
          seen and seen[0].get("chat_template_kwargs") == {"thinking": True},
          f"payload={seen[0] if seen else None}")


def test_argparse_default_is_false():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--thinking", action="store_true")
    args = p.parse_args([])
    check("--thinking default is False (sent by chat_stream when not None)",
          args.thinking is False, f"got {args.thinking!r}")


test_thinking_false_omits_chat_template_kwargs()
test_thinking_true_forwards_kwarg()
test_argparse_default_is_false()

print(f"\n{len(FAILURES)} failure(s)")
sys.exit(1 if FAILURES else 0)
