# Agent notes

Rules and gotchas for AI agents working in this repo.

## Running the tools during development

- **Use `uv run <script> ...`, never `uvx --from .`, while developing.**
  `uvx` caches built wheels keyed by path + version. If the source changes
  but the version in `pyproject.toml` does not, uvx silently re-runs the
  **stale** wheel — `--refresh` may not even rebuild it. You will debug
  code that does not exist anymore. Bump the version when you want a
  fresh uvx wheel.
- `uv run agent-sim ...` always runs the current source.

## Tests

- Plain pytest-less python files: `uv run python tests/test_x.py`
  (each exits non-zero on failure). Run all six before calling anything done:
  `test_cache_pressure.py`, `test_needle.py`, `test_ninfer_log.py`,
  `test_agent_sim.py`, `test_abort_sim.py`, `test_perf_sim.py`.
- **TDD**: when fixing or refactoring, write/update the failing test first,
  then make it pass.
- `agent_sim.py` unit tests must stay network-free: `chat_stream` and
  `run_session` are faked via monkeypatching / keyword args (see
  `_run_session_with_fake`).
- `perf_sim.py` unit tests stay network-free too: `run_perf` is driven
  against a faked `chat_stream` (same monkeypatch pattern).
- For live-endpoint smoke tests, a stub SSE server in `/tmp/agent_sim_stub.py`
  (longest-prefix-match fake cache) + `script -qec "..." /dev/null` gives a
  pty so the live progress view actually renders. Check for ANSI frames with
  `grep -ac $'\x1b\['`.

## Live progress view (agent-sim)

- `agent_sim.py` contains pure, ANSI-free formatting (`render_frame`,
  `_bar`, `fmt_ktok`, `fmt_tps`) plus two classes: `LiveView` (locked,
  worker-facing event sink — the *only* thing workers touch) and `Viz`
  (renderer thread, 8 fps, in-place ANSI repaint). Keep rendering out of
  worker threads; keep `render_frame` pure so frames are unit-testable.
- `tpc` in `LiveView` is **chars per token** (calibrated from the cold
  first turn). Don't flip the units — a chars/token vs token/chars mixup
  inflated bars to 85k/15k in real testing.
- TTY detection: on by default when `sys.stdout.isatty()`; `--viz`
  forces on, `--no-viz` always wins (decided by `viz_enabled()`).
- When the live view is off (piped stdout or `--no-viz`), every completed
  turn prints one plain-text `_fmt_record` line immediately (`flush=True`) —
  so non-TTY runs stream progress instead of staying silent until the end.
  The final per-turn dump still prints (it carries the `--ninfer-log`
  reuse-class column); the live lines are the progress preview.

## Style

- No code comments in new code unless asked.
- `make_body(chars, salt=...)` is the deterministic context generator —
  ~4 chars/token target, `chars = int(tokens / 0.25)`.
- `make_lorem(chars, seed=...)` (perf-sim) is the true-lorem-ipsum variant:
  same 4 chars/token contract, seeded sentence order (unique per step).
- Per-turn record dict shape is a contract: tests, `summarise`, the
  `--output` JSON, and `ninfer_log.reannotate_records` all depend on it.
  Extend, don't rename.
