#!/usr/bin/env python3
"""Concurrent agent-session cache benchmark (deployment-agnostic).

Simulates real coding-agent sessions: X sessions run as concurrent workers,
each growing a unique main-agent context turn by turn (5K steps, assistant
replies replayed verbatim), then spawning `--sub-windows` cold sub-agent
contexts, then returning to the main context for a finalize turn. Sessions
issue requests concurrently (one thread each, barrier start), so the engine's
admission/queue and its cache retention are both exercised under real
contention.

The headline metric is finalize survival: after a session's sub-agents ran
(and other sessions kept hammering the cache), is the ~main-tokens main
context still served from cache? A healthy deployment answers yes for every
session, checked two independent ways: the engine's reported cached-token
reuse fraction (miss-threshold) and the finalize TTFT ratio against the
session's own warm-main baseline (ttft-ratio) — under pressure an engine can
claim a cache hit while doing real prefill work, and only the TTFT reveals
that.

Cached tokens are read from usage.prompt_tokens_details.cached_tokens (ninfer,
standard vLLM), falling back to llama.cpp-compatible timings.cache_n, and
optionally cross-checked against the engine's /metrics prefix-cache counters
(vLLM). Engines that report neither mark per-request reuse as unknown.

On a TTY the run paints a live progress view (one column per session:
main/sub bars, phase+step, reuse %, pp/tg tps), redrawn in place at 8 fps;
piped output and --no-viz keep the plain-text output, and --viz forces the
view on when the wrapper hides the TTY (e.g. uvx).

Run:
  uvx --from . agent-sim --base-url http://my-box:8000/v1 --sessions 3 \
      --main-tokens 150000 --sub-tokens 40000 --output run.json
  uvx --from . agent-sim --base-url http://my-box:8000/v1 --sessions 2 \
      --main-tokens 80000 --sub-tokens 20000 --salt 42   # deterministic A/B
"""
import argparse
import json
import math
import shutil
import signal
import statistics
import sys
import threading
import time
from pathlib import Path

import requests

from cache_pressure.core import make_body, resolve_model
from cache_pressure.ninfer_log import parse_ninfer_log, reannotate_records

SYSTEM = "You are a helpful assistant."
FALLBACK_REPLY = "[no output generated]"
DEFAULT_URL = "http://localhost:8000/v1"


def build_session_plan(salt, step_tokens, steps, sub_tokens, sub_windows,
                       finalize_new):
    """Deterministic per-session content plan.

    Returns {"main": [step-chunks...], "subs": [[window-chunks]...],
    "finalize": str}. Salts differ per session, so no two sessions share any
    prefix — nothing rides on another session's cached content.
    """
    main = [make_body(int(step_tokens / 0.25), salt=salt + i) for i in range(steps)]
    subs = [
        [make_body(int(sub_tokens / 0.25), salt=salt + 10_000 + w)]
        for w in range(sub_windows)
    ]
    finalize = make_body(int(finalize_new / 0.25), salt=salt + 20_000)
    return {"main": main, "subs": subs, "finalize": finalize}


def compute_reuse(cached, prev_prompt):
    """Fraction of the previous request's prompt served from cache."""
    if not cached or not prev_prompt:
        return 0.0
    return cached / prev_prompt


def finalize_ttft_verdict(records, ratio=1.75):
    """Per-session finalize TTFT verdict against a warm-main baseline.

    Baseline: median of the last 3 non-aborted main-continuation TTFTs (the
    warm-prefill speed of this session's own context, engine speed included).
    ok = finalize TTFT strictly below ratio x baseline. Sessions without a
    usable baseline are unverified (ratio None, ok False) — never a silent
    pass, since TTFT is the one signal the engine cannot fake.
    """
    base = {}
    fin = {}
    fin_reuse = {}
    fin_aborted = {}
    for r in records:
        sid = r["session"]
        if r["phase"] == "finalize":
            fin_aborted[sid] = bool(r.get("aborted"))
            if not r.get("aborted"):
                fin_reuse[sid] = r.get("reuse")
        if r.get("aborted") or r.get("ttft") is None:
            continue
        if r["phase"] == "main" and r["step"] > 0:
            base.setdefault(sid, []).append((r["step"], r["ttft"]))
        elif r["phase"] == "finalize":
            fin[sid] = r["ttft"]
    out = {}
    for sid in set(base) | set(fin) | set(fin_aborted):
        turns = [t for _, t in sorted(base.get(sid, []))][-3:]
        baseline = statistics.median(turns) if turns else None
        f = fin.get(sid)
        ratio_ = (f / baseline) if (baseline and f is not None) else None
        out[sid] = {"baseline_ttft": baseline, "finalize_ttft": f,
                    "ttft_ratio": ratio_,
                    "ttft_ok": ratio_ is not None and ratio_ < ratio,
                    "reuse": fin_reuse.get(sid),
                    "aborted": fin_aborted.get(sid, False)}
    return out


def label_finalize(v, threshold, ratio):
    """Verdict word for a session's finalize: ok | evicted | degraded
    (| 'evicted, degraded') | unverified | aborted.

    evicted: the engine reported the prefix as (mostly) uncached.
    degraded: the engine reported a hit but the finalize TTFT exceeds the
    warm-main baseline by the ratio factor — real prefill paid for a
    claimed cache hit.
    """
    if v.get("aborted"):
        return "aborted"
    if v.get("reuse") is None or v.get("ttft_ratio") is None:
        return "unverified"
    evicted = v["reuse"] < threshold
    degraded = v["ttft_ratio"] >= ratio
    if evicted and degraded:
        return "evicted, degraded"
    if evicted:
        return "evicted"
    if degraded:
        return "degraded"
    return "ok"


def validate_sizes(main_size, sub_tokens, finalize_new, max_context):
    """Return an error string if the workload would overflow max-context."""
    if main_size + finalize_new > max_context:
        return (f"main context ({main_size:,} tok) + finalize ({finalize_new:,} tok) "
                f"exceeds max-context ({max_context:,} tok)")
    if sub_tokens > max_context:
        return (f"sub-agent context ({sub_tokens:,} tok) exceeds "
                f"max-context ({max_context:,} tok)")
    return None


def fetch_max_context(base_url, api_key, timeout):
    """Advertised max_model_len via GET /models, or None."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = base_url.rstrip("/") + "/models"
    try:
        data = requests.get(url, headers=headers, timeout=timeout).json()
    except Exception:
        return None
    items = data.get("data") or data.get("models") or []
    lens = [m.get("max_model_len") for m in items if m.get("max_model_len")]
    return max(lens) if lens else None


def read_prefix_cache_metrics(base_url, api_key, timeout):
    """(queries, hits) token counters from the engine /metrics (vLLM), or None."""
    url = base_url.rstrip("/").replace("/v1", "") + "/metrics"
    try:
        text = requests.get(url, timeout=timeout).text
    except Exception:
        return None
    queries = hits = None
    for line in text.splitlines():
        if line.startswith("vllm:prefix_cache_queries_total"):
            queries = int(float(line.rsplit(" ", 1)[-1]))
        elif line.startswith("vllm:prefix_cache_hits_total"):
            hits = int(float(line.rsplit(" ", 1)[-1]))
    if queries is None or hits is None:
        return None
    return queries, hits


def chat_stream(url, model, messages, max_tokens, timeout, api_key, thinking,
                on_delta=None, abort_event=None):
    """Stream one completion; return prompt/cached/ttft/wall/content/error.

    on_delta, when given, is called with the char count of each content
    chunk as it arrives (drives the live progress view). abort_event, when
    given, is polled before posting and on every streamed line: once set,
    the connection is closed and the partial result is returned with
    aborted=True — a cache miss in one session cuts short the pending
    requests of every other session instead of letting them finish.
    """
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if not thinking:
        payload["reasoning_effort"] = "none"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    t0 = time.monotonic()
    ttft = None
    prompt = None
    cached_usage = cached_timings = None
    completion = None
    content = ""
    error = None
    resp = None
    aborted = False
    try:
        if abort_event is not None and abort_event.is_set():
            aborted = True
        else:
            resp = requests.post(url, json=payload, headers=headers,
                                 stream=True, timeout=timeout)
            resp.raise_for_status()
            for line in resp.iter_lines():
                if abort_event is not None and abort_event.is_set():
                    aborted = True
                    break
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
                    prompt = usage.get("prompt_tokens", prompt)
                    cached_usage = ((usage.get("prompt_tokens_details") or {})
                                    .get("cached_tokens"))
                    completion = usage.get("completion_tokens", completion)
                timings = chunk.get("timings")
                if timings:
                    cached_timings = timings.get("cache_n")
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {})
                    tok = (delta.get("content")
                           or delta.get("reasoning")
                           or delta.get("reasoning_content") or "")
                    if tok:
                        if ttft is None:
                            ttft = time.monotonic() - t0
                        c = delta.get("content")
                        if c:
                            content += c
                            if on_delta:
                                on_delta(len(c))
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
        if ttft is None:
            ttft = time.monotonic() - t0
    finally:
        if resp is not None:
            resp.close()
    if cached_usage is not None:
        cached, source = cached_usage, "usage"
    elif cached_timings is not None:
        cached, source = cached_timings, "timings"
    else:
        cached, source = None, "none"
    return {
        "prompt": prompt,
        "cached": cached,
        "cached_source": source,
        "ttft": ttft,
        "wall": time.monotonic() - t0,
        "content": content,
        "completion": completion,
        "error": error,
        "aborted": aborted,
    }


def run_session(base_url, model, api_key, timeout, plan, session_id,
                barrier, max_tokens, thinking, records, lock, viz=None,
                stop_event=None, miss_threshold=0.66):
    """One session worker: grow main -> subs -> finalize, recording each turn.

    Fails fast: the first turn that should hit the cache (a main continuation,
    step > 0, or the finalize) that comes back with reuse < miss_threshold
    trips the shared stop_event. Every session then winds down, and turns
    still in flight are aborted at the next streamed chunk (the stop_event
    doubles as chat_stream's abort_event), so only the turns that actually
    ran are recorded.
    """
    if stop_event is None:
        stop_event = threading.Event()
    try:
        barrier.wait()
    except threading.BrokenBarrierError:
        return
    session = requests.Session()
    main_messages = [{"role": "system", "content": SYSTEM}]
    prev_prompt = 0
    url = base_url.rstrip("/") + "/chat/completions"
    sid = session_id

    def _send(messages, phase, step, user_chars):
        """Run one turn, pushing live events into the view when attached."""
        if viz:
            viz.turn_start(sid, phase, step, user_chars)
        extra = {"on_delta": (lambda c: viz.delta(sid, c))} if viz else {}
        res = chat_stream(url, model, messages, max_tokens, timeout,
                          api_key, thinking, abort_event=stop_event, **extra)
        return res

    def _abort():
        if viz:
            viz.finish(sid, False)

    for step, chunk in enumerate(plan["main"]):
        if stop_event.is_set():
            _abort()
            break
        main_messages.append({"role": "user", "content": chunk})
        res = _send(main_messages, "main", step, len(chunk))
        reuse = compute_reuse(res["cached"], prev_prompt)
        aborted = bool(res.get("aborted"))
        record = {"session": session_id, "phase": "main", "step": step,
                  "prompt": res["prompt"],
                  "cached": res["cached"], "cached_source": res["cached_source"],
                  "reuse": reuse,
                  "ttft": res["ttft"], "wall": res["wall"],
                  "error": res["error"], "aborted": aborted}
        with lock:
            records.append(record)
        if viz is None:
            print(_fmt_record(record), flush=True)
        if viz:
            viz.turn_end(sid, res, reuse)
        if res["prompt"]:
            prev_prompt = res["prompt"]
        if not aborted:
            main_messages.append({"role": "assistant",
                                  "content": res["content"] or FALLBACK_REPLY})
        if aborted:
            stop_event.set()
            _abort()
            break
        if step > 0 and not res["error"] and reuse < miss_threshold:
            stop_event.set()
            _abort()
            break

    if not stop_event.is_set():
        for w, window in enumerate(plan["subs"]):
            if stop_event.is_set():
                _abort()
                break
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": window[0]}]
            res = _send(messages, "sub", w, len(window[0]))
            record = {"session": session_id, "phase": "sub", "step": w,
                      "prompt": res["prompt"],
                      "cached": res["cached"], "cached_source": res["cached_source"],
                      "reuse": 0.0,
                      "ttft": res["ttft"], "wall": res["wall"],
                      "error": res["error"],
                      "aborted": bool(res.get("aborted"))}
            with lock:
                records.append(record)
            if viz is None:
                print(_fmt_record(record), flush=True)
            if viz:
                viz.turn_end(sid, res, 0.0)
            if res.get("aborted"):
                stop_event.set()
                _abort()
                break

    if not stop_event.is_set():
        main_messages.append({"role": "user", "content": plan["finalize"]})
        res = _send(main_messages, "finalize", 0, len(plan["finalize"]))
        reuse = compute_reuse(res["cached"], prev_prompt)
        aborted = bool(res.get("aborted"))
        record = {"session": session_id, "phase": "finalize", "step": 0,
                  "prompt": res["prompt"],
                  "cached": res["cached"], "cached_source": res["cached_source"],
                  "reuse": reuse,
                  "ttft": res["ttft"], "wall": res["wall"],
                  "error": res["error"], "aborted": aborted}
        with lock:
            records.append(record)
        if viz is None:
            print(_fmt_record(record), flush=True)
        if viz:
            viz.turn_end(sid, res, reuse)
        miss = not res["error"] and not aborted and reuse < miss_threshold
        if miss or aborted:
            stop_event.set()
        if viz:
            viz.finish(sid, not miss and not aborted)
    session.close()


# ── live progress view ─────────────────────────────────────────────────
#
# Workers push turn events into a locked LiveView; a single renderer thread
# takes immutable snapshots and repaints the frame in place (ANSI cursor-up
# + clear-line). All formatting below is pure and ANSI-free unless color
# is requested, so frames are unit-testable.


def fmt_ktok(n):
    """135000 -> '135k', 1200000 -> '1.2M'."""
    if n is None:
        return "–"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1000:
        return f"{n / 1000:.0f}k"
    return str(int(n))


def fmt_tps(x):
    """12400 -> '12.4k', 46.4 -> '46'."""
    if x is None:
        return "–"
    if x >= 1000:
        return f"{x / 1000:.1f}k"
    return f"{x:.0f}"


def fmt_elapsed(x):
    """42.3 -> '42.3s', 83 -> '1m23s', 3661.7 -> '1h01m'."""
    if x is None:
        return "–"
    if x < 60:
        return f"{x:.1f}s"
    if x < 3600:
        return f"{int(x // 60)}m{int(x % 60):02d}s"
    return f"{int(x // 3600)}h{int((x % 3600) // 60):02d}m"


def _bar(frac, width):
    """Progress bar as (filled, partial, empty) strings."""
    frac = max(0.0, min(1.0, frac))
    cells = frac * width
    full = int(cells)
    if full >= width:
        return "█" * width, "", ""
    partial = "▌" if cells - full > 0.05 else ""
    return "█" * full, partial, "░" * (width - full - len(partial))


def _vlen(s):
    """Visible length: ANSI SGR escapes do not count."""
    n = i = 0
    while i < len(s):
        if s[i] == "\x1b" and i + 1 < len(s) and s[i + 1] == "[":
            j = i + 2
            while j < len(s) and (s[j].isdigit() or s[j] == ";"):
                j += 1
            i = j + 1
            continue
        n += 1
        i += 1
    return n


def _paint(s, code, color):
    return f"\x1b[{code}m{s}\x1b[0m" if color and s else s


def _bar_str(frac, width, color):
    full, partial, empty = _bar(frac, width)
    return _paint(full + partial, "32", color) + _paint(empty, "2", color)


def _session_lines(s, main_size, sub_tokens, sub_total, steps_total, color):
    l1 = (f"s{s['sid']}  main "
          f"{_bar_str(s['main_value'] / main_size if main_size else 0.0, 14, color)} "
          f"{fmt_ktok(s['main_value'])}/{fmt_ktok(main_size)}")
    sub_scale = sub_total * sub_tokens or 1
    cold = " · cold" if s["sub_done"] > 0 else ""
    l2 = (f"    sub  "
          f"{_bar_str(s['sub_value'] / sub_scale, 10, color)} "
          f"{s['sub_done']}/{sub_total}{cold}")
    if s["done"]:
        l3 = _paint(f"    done {'✓' if s['ok'] else '✗'}",
                    "32" if s["ok"] else "31", color)
    elif s["err"]:
        l3 = _paint("    err ✗", "31", color)
    elif s["phase"] == "sub":
        l3 = f"    sub {s['sub_done']}/{sub_total}" + cold
    elif s["phase"] == "finalize":
        l3 = "    final"
    elif s["turns"]:
        pct = 100.0 * s["sum_reuse"] / s["turns"]
        l3 = (f"    main {s['turns']}/{steps_total} · hit "
              f"{_paint(f'{pct:.0f}%', '32' if pct >= 95 else '31', color)}")
    else:
        l3 = f"    main 0/{steps_total} · hit –"
    l4 = f"    pp {fmt_tps(s['pp'])} tps · tg {fmt_tps(s['tg'])} tps"
    return [l1, l2, l3, l4]


def _pad(s, width):
    gap = width - _vlen(s)
    return s + " " * gap if gap > 0 else s


def render_frame(states, main_size, sub_tokens, sub_total, steps_total,
                 elapsed, term_w=160, color=False):
    """One frame from immutable session states: header + 4 rows.

    Columns sit side by side when they fit in term_w, otherwise session
    blocks stack vertically.
    """
    n = len(states)
    mm, ss = divmod(int(elapsed), 60)
    header = _paint(f"agent-sim · {n} session{'s' if n != 1 else ''} · "
                    f"{mm:02d}:{ss:02d}", "2", color)
    cols = [_session_lines(s, main_size, sub_tokens, sub_total, steps_total,
                           color) for s in states]
    width = max(_vlen(l) for col in cols for l in col)
    out = [header]
    if width * n + 2 * (n - 1) <= term_w:
        for r in range(4):
            out.append("  ".join(_pad(col[r], width) for col in cols))
    else:
        for i, col in enumerate(cols):
            if i:
                out.append("")
            out.extend(col)
    target = max(_vlen(l) for l in out)
    return "\n".join(_pad(l, target) for l in out)


def _fresh(sid):
    return {"sid": sid, "phase": "main", "step": 0, "main_value": 0.0,
            "sub_value": 0.0, "sub_done": 0, "turns": 0, "sum_reuse": 0.0,
            "pp": None, "tg": None, "done": False, "ok": None, "err": None,
            "_base": 0.0, "_user": 0, "_reply": 0, "_t0": None, "_t1": None}


def viz_enabled(is_tty, force=False, disable=False):
    """Whether the live progress view should be on.

    On by default when stdout is a TTY; --viz forces it on (wrappers like
    uvx pipe the child's stdout, so isatty() is False there); --no-viz
    always wins.
    """
    return (is_tty or force) and not disable


class LiveView:
    """Thread-shared live state: workers push events, renderer snapshots."""

    def __init__(self, *, sessions, main_size, sub_tokens, sub_total):
        self._lock = threading.Lock()
        self._main_size = main_size
        self._sub_tokens = sub_tokens
        self._sub_total = sub_total
        self._tpc = 4.0  # chars per token, calibrated from the cold turn
        self._calib = False
        self._s = [_fresh(i) for i in range(sessions)]

    @property
    def tpc(self):
        with self._lock:
            return self._tpc

    def turn_start(self, sid, phase, step, user_chars):
        with self._lock:
            s = self._s[sid]
            s["phase"] = phase
            s["step"] = step
            s["err"] = None
            s["_base"] = (s["sub_done"] * self._sub_tokens
                          if phase == "sub" else s["main_value"])
            s["_user"] = user_chars
            s["_reply"] = 0
            s["_t0"] = time.monotonic()
            s["_t1"] = None

    def delta(self, sid, chars):
        with self._lock:
            s = self._s[sid]
            s["_reply"] += chars
            if s["_t1"] is None:
                s["_t1"] = time.monotonic()

    def turn_end(self, sid, res, reuse):
        with self._lock:
            s = self._s[sid]
            prompt, cached = res["prompt"], res["cached"]
            ttft, wall, comp = res["ttft"], res["wall"], res.get("completion")
            if res.get("error"):
                s["err"] = res["error"]
            if s["phase"] in ("main", "finalize") and prompt:
                s["main_value"] = float(prompt)
            if s["phase"] == "sub":
                s["sub_done"] += 1
                s["sub_value"] = float(s["sub_done"] * self._sub_tokens)
            if s["phase"] == "main" and s["step"] > 0:
                s["turns"] += 1
                s["sum_reuse"] += reuse or 0.0
            if ttft and prompt and cached is not None and prompt - cached > 0:
                s["pp"] = (prompt - cached) / ttft
            if comp and wall and ttft and wall > ttft:
                s["tg"] = comp / (wall - ttft)
            if (not self._calib and s["phase"] == "main" and s["step"] == 0
                    and prompt and s["_user"] + s["_reply"] > 0):
                self._tpc = (s["_user"] + s["_reply"]) / prompt
                self._calib = True
            s["_t0"] = None
            s["_t1"] = None

    def finish(self, sid, ok):
        with self._lock:
            s = self._s[sid]
            s["done"] = True
            s["ok"] = ok
            s["_t0"] = None
            s["_t1"] = None

    def snapshot(self):
        with self._lock:
            now = time.monotonic()
            out = []
            for s in self._s:
                d = {k: v for k, v in s.items() if not k.startswith("_")}
                if s["_t0"] is not None and not s["done"]:
                    grown = (s["_user"] + s["_reply"]) / self._tpc
                    if s["phase"] == "sub":
                        d["sub_value"] = s["_base"] + grown
                    else:
                        d["main_value"] = s["_base"] + grown
                    if s["_t1"] is not None and s["_reply"]:
                        el = now - s["_t1"]
                        if el > 0.05:
                            d["tg"] = (s["_reply"] / self._tpc) / el
                out.append(d)
            return tuple(out)


class Viz:
    """Render loop: repaints the live frame in place until stop()."""

    def __init__(self, *, view, main_size, sub_tokens, sub_total, steps_total,
                 fps=8):
        self._view = view
        self._args = (main_size, sub_tokens, sub_total, steps_total)
        self._interval = 1.0 / fps
        self._stop = threading.Event()
        self._color = sys.stdout.isatty()
        self._width = shutil.get_terminal_size((160, 24)).columns
        self._t0 = time.monotonic()
        self._lines = 0
        self._th = None

    def refresh_size(self):
        """Re-read the terminal width (SIGWINCH handler)."""
        self._width = shutil.get_terminal_size((160, 24)).columns

    def start(self):
        sys.stdout.write("\x1b[?25l")
        sys.stdout.flush()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()
        if self._th:
            self._th.join()
        self._paint(self._render())
        sys.stdout.write("\x1b[?25h\n")
        sys.stdout.flush()

    def _render(self):
        return render_frame(self._view.snapshot(), *self._args,
                            elapsed=time.monotonic() - self._t0,
                            term_w=self._width, color=self._color)

    def _paint(self, frame):
        lines = frame.split("\n")
        out = f"\x1b[{self._lines}A" if self._lines else ""
        for line in lines:
            out += "\x1b[2K" + line + "\n"
        sys.stdout.write(out)
        sys.stdout.flush()
        self._lines = len(lines)

    def _loop(self):
        while not self._stop.wait(self._interval):
            self._paint(self._render())


def summarise(records, num_sessions, sub_windows, threshold=0.66,
              elapsed=None, ratio=1.75):
    """Headline metrics from the collected per-turn records.

    Turns aborted mid-flight (another session tripped the stop) are left out
    — they neither hit nor missed, so they would only skew the counts.

    Two independent finalize gates: reuse (the engine's reported
    cached_tokens fraction) and TTFT ratio (finalize ttft vs the warm-main
    baseline) — engines can report a cache hit while doing real prefill
    work under pressure, so both must pass.
    """
    mains = [r for r in records if r["phase"] == "main" and r["step"] > 0
             and not r.get("aborted")]
    finals = [r for r in records if r["phase"] == "finalize"
              and not r.get("aborted")]
    subs = [r for r in records if r["phase"] == "sub" and not r.get("aborted")]
    main_reused = sum(1 for r in mains if r["reuse"] >= threshold)
    finalize_ok = sum(1 for r in finals if r["reuse"] >= threshold)
    subs_cold = sum(1 for r in subs
                    if not r["cached"] or r["cached"] <= 0.05 * (r["prompt"] or 0))
    verdict = finalize_ttft_verdict(records, ratio=ratio)
    for v in verdict.values():
        v["verdict"] = label_finalize(v, threshold, ratio)
    ttft_ok = sum(1 for v in verdict.values() if v["ttft_ok"])
    ttft_verified = sum(1 for v in verdict.values()
                        if v["ttft_ratio"] is not None)
    return {
        "main_reused": main_reused,
        "main_total": len(mains),
        "finalize_ok": finalize_ok,
        "finalize_total": len(finals),
        "subs_cold": subs_cold,
        "subs_total": len(subs),
        "main_reuse_pct": (100.0 * main_reused / len(mains)) if mains else None,
        "finalize_survival_pct": (100.0 * finalize_ok / len(finals))
        if finals else None,
        "subs_cold_pct": (100.0 * subs_cold / len(subs)) if subs else None,
        "finalize_ttft_ok": ttft_ok,
        "finalize_ttft_verified": ttft_verified,
        "finalize_ttft_pct": (100.0 * ttft_ok / ttft_verified)
        if ttft_verified else None,
        "finalize_ttft": verdict,
        "elapsed": elapsed,
    }


def _fmt(v):
    return "–" if v is None else f"{v:.2f}s" if isinstance(v, float) and v < 100 else (f"{v:.1f}s" if isinstance(v, float) else str(v))


def _fmt_record(r):
    """One per-turn record as a plain-text line, shared by the live stream
    (non-viz runs) and the final per-turn dump."""
    cached = r["cached"]
    shown = "–" if cached is None else f"{cached:,}"
    reuse_class = r.get("reuse_class", "")
    prompt_shown = "–" if r["prompt"] is None else f"{r['prompt']:,}"
    aborted = "  ABORTED" if r.get("aborted") else ""
    return (f"  s{r['session']} {r['phase']:<9} step {r['step']:>2}  "
            f"prompt={prompt_shown:>8}  cached={shown:>8}  "
            f"reuse={r['reuse'] * 100:5.1f}%  "
            f"{reuse_class + '  ' if reuse_class else ''}"
            f"ttft={_fmt(r['ttft'])}  err={r['error'] or '–'}{aborted}")


def format_finalize(summary, ratio):
    """Render the finalize-survival block: one verdict headline, one line
    per session — the sub coldness sanity check is deliberately not shown,
    it only guards the test harness itself."""
    # Aborted sessions are our own short-circuit (a miss in one session
    # stops the others' pending turns) — not an engine finding, so they
    # stay out of the verdict block.
    per = {sid: v for sid, v in summary["finalize_ttft"].items()
           if not v["aborted"]}
    total = len(per)
    ok = [sid for sid in per if per[sid]["verdict"] == "ok"]
    if total == 0:
        lines = ["  FINALIZE survival: 0/0 (no finalize completed)"]
    elif len(ok) == total:
        lines = [f"  FINALIZE survival: {total}/{total} OK"]
    else:
        fails = ", ".join(f"s{sid} {per[sid]['verdict']}"
                          for sid in sorted(per)
                          if per[sid]["verdict"] != "ok")
        lines = [f"  FINALIZE survival: {len(ok)}/{total} — {fails}"]
    for sid in sorted(per):
        v = per[sid]
        ratio_txt = (f"{v['ttft_ratio']:.2f}x"
                     if v["ttft_ratio"] is not None else "–")
        lines.append(f"    s{sid}  base {_fmt(v['baseline_ttft'])}  "
                     f"finalize {_fmt(v['finalize_ttft'])}  {ratio_txt}  "
                     f"{v['verdict'].upper()}")
    return lines


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Concurrent agent-session cache benchmark")
    p.add_argument("--base-url", default=DEFAULT_URL)
    p.add_argument("--model", default=None,
                   help="Model id (default: auto-detected via GET /models)")
    p.add_argument("--api-key", default=None)
    p.add_argument("--sessions", type=int, default=3,
                   help="number of parallel agent sessions")
    p.add_argument("--miss-threshold", type=float, default=0.66,
                   help="min reuse fraction a main/finalize turn must hit "
                        "to avoid aborting the run (default: 0.66)")
    p.add_argument("--ttft-ratio", type=float, default=1.75,
                   help="max finalize TTFT as a multiple of the session's "
                        "warm-main baseline (median of last 3 main turns) "
                        "for a pass (default: 1.75)")
    p.add_argument("--main-tokens", type=int, default=150000,
                   help="main-agent context size in tokens")
    p.add_argument("--sub-tokens", type=int, default=40000,
                   help="sub-agent context size in tokens")
    p.add_argument("--sub-windows", type=int, default=2,
                   help="sub-agent windows per session")
    p.add_argument("--step-tokens", type=int, default=5000,
                   help="main-context growth per turn")
    p.add_argument("--finalize-tokens", type=int, default=2000,
                   help="back-to-main suffix size in tokens")
    p.add_argument("--max-context", type=int, default=None,
                   help="model max context (default: auto from GET /models)")
    p.add_argument("--max-concurrency", type=int, default=2,
                   help="engine max-concurrency hint (for the queue warning)")
    p.add_argument("--max-tokens", type=int, default=64,
                   help="generation budget per turn (replayed as the reply)")
    p.add_argument("--thinking", action="store_true",
                   help="enable reasoning/thinking (default: off — "
                        "reasoning_effort none, so replies are plain text "
                        "that replay verbatim into the next request)")
    p.add_argument("--timeout", type=int, default=1200)
    p.add_argument("--salt", type=int, default=None,
                   help="run salt (default: time-based, fresh contexts)")
    p.add_argument("--ninfer-log", default=None,
                   help="path to ninfer's request-log JSONL (--request-log-jsonl); "
                        "re-annotates records with the ground-truth reuse class "
                        "(device-hit vs offload-hit vs true-miss), which the API's "
                        "cached_tokens cannot distinguish")
    p.add_argument("--output", default=None)
    p.add_argument("--no-viz", action="store_true",
                   help="disable the live progress display")
    p.add_argument("--viz", action="store_true",
                   help="force the live progress display even when stdout "
                        "is not a TTY (e.g. when run via uvx)")
    args = p.parse_args(argv)

    if args.sessions < 1:
        p.error("--sessions must be >= 1")

    url = args.base_url.rstrip("/") + "/chat/completions"

    model = args.model
    if model is None:
        model = resolve_model(args.base_url, args.api_key, args.timeout)
        if model is None:
            print("  Could not auto-detect a model (GET /models failed or "
                  "returned none); pass --model")
            sys.exit(1)
        print(f"  Model: {model} (auto-detected)")

    max_context = args.max_context
    if max_context is None:
        max_context = fetch_max_context(args.base_url, args.api_key, args.timeout)
        if max_context is None:
            print("  Could not auto-detect max-context (GET /models failed); "
                  "pass --max-context")
            sys.exit(1)
        print(f"  Max context: {max_context:,} tokens (auto-detected)")

    steps = max(1, math.ceil(args.main_tokens / args.step_tokens))
    main_size = steps * args.step_tokens
    err = validate_sizes(main_size, args.sub_tokens, args.finalize_tokens,
                         max_context)
    if err:
        print(f"  Validation failed: {err}")
        sys.exit(1)

    salt = args.salt if args.salt is not None else int(time.time())
    print(f"Agent-sim: {args.sessions} session(s) x ~{main_size:,} tok main "
          f"(grown in {steps} x {args.step_tokens:,} steps) + "
          f"{args.sub_windows} x {args.sub_tokens:,} tok sub windows, "
          f"thinking={'on' if args.thinking else 'off'}, salt {salt}")

    metrics0 = read_prefix_cache_metrics(args.base_url, args.api_key, args.timeout)
    metrics1 = ratio = None
    if metrics0:
        print(f"  engine prefix-cache before: {metrics0[1]:,}/{metrics0[0]:,} tokens")

    records = []
    lock = threading.Lock()
    barrier = threading.Barrier(args.sessions)
    stop_event = threading.Event()
    view = viz = None
    if viz_enabled(sys.stdout.isatty(), args.viz, args.no_viz):
        view = LiveView(sessions=args.sessions, main_size=main_size,
                        sub_tokens=args.sub_tokens, sub_total=args.sub_windows)
        viz = Viz(view=view, main_size=main_size, sub_tokens=args.sub_tokens,
                  sub_total=args.sub_windows, steps_total=steps)
        try:
            signal.signal(signal.SIGWINCH, lambda *_: viz.refresh_size())
        except (ValueError, OSError, AttributeError):
            pass
        viz.start()

    t0 = time.monotonic()
    threads = []
    for s in range(args.sessions):
        plan = build_session_plan(salt + s * 100_000, args.step_tokens, steps,
                                  args.sub_tokens, args.sub_windows,
                                  args.finalize_tokens)
        t = threading.Thread(target=run_session, kwargs={
            "base_url": args.base_url, "model": model, "api_key": args.api_key,
            "timeout": args.timeout, "plan": plan, "session_id": s,
            "barrier": barrier, "max_tokens": args.max_tokens,
            "thinking": args.thinking, "records": records, "lock": lock,
            "viz": view, "stop_event": stop_event,
            "miss_threshold": args.miss_threshold,
        })
        threads.append(t)
        t.start()
    try:
        for t in threads:
            t.join()
    finally:
        elapsed = time.monotonic() - t0
        if viz:
            viz.stop()

    aborted = stop_event.is_set()
    if aborted:
        print("  ABORTED: a cache miss was caught — stopped early; "
              "turns that had not run yet were skipped")

    if args.ninfer_log:
        entries = parse_ninfer_log(args.ninfer_log)
        records, unmatched, _extra = reannotate_records(records, entries)
        print(f"  ninfer request-log: {len(entries)} requests, "
              f"{len(unmatched)} records without a log entry")
        n_miss = sum(1 for r in records if r.get("reuse_class") == "miss")
        n_hit = sum(1 for r in records if r.get("reuse_class") == "hit")
        n_part = sum(1 for r in records if r.get("reuse_class") == "partial")
        print(f"  ground-truth reuse classes (from the request log): "
              f"{n_hit} hit / {n_part} partial / {n_miss} miss")

    print(f"\n── per-turn records ({len(records)}) ──")
    for r in sorted(records, key=lambda x: (x["session"], x["phase"],
                                            x["step"])):
        print(_fmt_record(r))

    summary = summarise(records, args.sessions, args.sub_windows,
                        threshold=args.miss_threshold, elapsed=elapsed,
                        ratio=args.ttft_ratio)
    print("\n── summary ──")
    print(f"  run time: {fmt_elapsed(summary['elapsed'])}")
    print(f"  main continuation turns reused: "
          f"{summary['main_reused']}/{summary['main_total']} "
          f"({summary['main_reuse_pct'] and f'{summary['main_reuse_pct']:.0f}%' or 'n/a'})")
    for line in format_finalize(summary, args.ttft_ratio):
        print(line)

    if metrics0:
        metrics1 = read_prefix_cache_metrics(args.base_url, args.api_key,
                                             args.timeout)
        if metrics1:
            dh = metrics1[1] - metrics0[1]
            dq = metrics1[0] - metrics0[0]
            ratio = (dh / dq) if dq else None
            print(f"  engine prefix-cache delta over the run: {dh:,}/{dq:,} "
                  f"tokens (hits/queried)"
                  + (f" = {ratio * 100:.0f}%" if ratio is not None else ""))

    reported = any(r["cached_source"] != "none" for r in records)
    if reported:
        ok = (summary["finalize_ok"] == summary["finalize_total"] and
              summary["finalize_total"] > 0 and
              summary["finalize_ttft_verified"] > 0 and
              summary["finalize_ttft_ok"] == summary["finalize_ttft_verified"])
    elif metrics1 and ratio is not None:
        print("  engine does not report per-request cached tokens; "
              "verification falls back to the engine prefix-cache counters")
        ok = ratio >= 0.5
    else:
        print("  WARNING: no per-request cache signal and no /metrics "
              "counters — retention could not be verified")
        ok = False

    if aborted:
        ok = False

    result = {
        "salt": salt,
        "sessions": args.sessions,
        "main_tokens": main_size,
        "step_tokens": args.step_tokens,
        "sub_tokens": args.sub_tokens,
        "sub_windows": args.sub_windows,
        "finalize_tokens": args.finalize_tokens,
        "model": model,
        "summary": summary,
        "records": records,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"  Results written to {args.output}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
