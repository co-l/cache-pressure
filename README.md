# cache-pressure

Measure how much context a prefix cache **actually retains** under overflow
pressure — not what the engine *advertises*.

The advertised KV cache size is an **active-request planning budget**, not a
ceiling on cacheable content. Hybrid layouts (MLA + sparse-attention +
recurrent-state groups) pack cached blocks denser than that reservation,
and cache-management defects can waste the headroom with duplicate blocks
and unreachable replay tails. This tool quantifies the real number, so you
can tell a healthy cache from a leaking one on any OpenAI-compatible
deployment.

## Benchmarks

| bench | what it measures | minimal command |
|---|---|---|
| `cache-pressure` | real retained capacity under overflow | `uvx --from cache-pressure cache-pressure --base-url … --kv-size N` |
| `needle-test` | whether long-context retrieval is intact | `uvx --from cache-pressure needle-test --base-url …` |
| `agent-sim` | session retention under concurrent load | `uvx --from cache-pressure agent-sim --base-url …` |
| `abort-sim` | prefix survival after a mid-thinking abort | `uvx --from cache-pressure abort-sim --base-url …` |
| `perf-sim` | prefill/generation speed vs context length | `uvx --from cache-pressure perf-sim --base-url …` |

All five talk to the same OpenAI-compatible endpoint (`--base-url`, default
`http://localhost:8000/v1`), auto-detect the model via `GET /models`
(override with `--model`), and take `--api-key` when the endpoint requires it.

## How it works

```
1. capacity   the engine's advertised KV size in tokens, provided manually
              with --kv-size (required)
2. calibrate  tokens/char for this tokenizer at the target length
              (length-dependent, iterates to convergence), then the
              hit/miss threshold: one probe context sent cold (miss
              baseline) and again from cache (hit baseline) — the
              threshold is their midpoint
3. hydrate    N unique ~8K-token contexts sequentially (max_tokens=1,
              prefill-only — the full context is committed regardless)
              N = ceil(capacity/8000)+5, so the cache overflows
4. verify     re-send each context in reverse order (newest first),
              classify hit/miss by TTFT against the calibrated threshold
              → the first miss is the oldest evicted context = the real
              retained capacity. Stops there: under LRU everything older
              is evicted by construction.
```

Contexts are seeded random word-orders (unique per salt within a run,
byte-identical across runs) so no context rides on another's cached
content, and two runs (e.g. before/after a fix) replay the same text.

The default 8K context size keeps the measurement granularity fine; pass
`--context-tokens` to coarsen or refine. The hit/miss threshold is
calibrated at runtime from measured cold-prefill and cache-hit TTFT, so it
stays valid regardless of engine speed — override with `--hit-threshold`
only if you have a reason.

## Usage

The latest release runs straight from PyPI — no install needed:

```bash
# point --base-url at your OpenAI-compatible endpoint; the model is
# auto-detected via GET /models (pass --model to override)
uvx --from cache-pressure cache-pressure --base-url http://my-server:8000/v1 \
    --kv-size <ADVERTISED_KV_CACHE> --output run.json

# A/B two runs
uvx --from cache-pressure cache-pressure --compare fix.json control.json
```

Small sanity check (3 contexts, all hits):

```bash
uvx --from cache-pressure cache-pressure --base-url http://my-server:8000/v1 \
    --kv-size <ADVERTISED_KV_CACHE> --num-contexts 3
```

No env vars are read for credentials — pass `--api-key` when the endpoint
requires it:

```bash
uvx --from cache-pressure cache-pressure --base-url http://my-server:8000/v1 \
    --api-key sk-... --kv-size <ADVERTISED_KV_CACHE>
```

Key flags: `--kv-size` (required), `--context-tokens` (8000),
`--num-contexts` (default: computed to overflow), `--margin` (5),
`--hit-threshold` (calibrated at runtime), `--timeout` (300),
`--compare` for A/B.

### `needle-test` — companion correctness check

The retention number means nothing if the engine is broken. This sweeps
increasing context lengths (default 50K/100K/200K/300K/450K), hides a
needle sentence at 0.8 depth in each unique haystack, and requires the
model to output the exact secret code — proving long-context retrieval is
intact end-to-end.

```bash
uvx --from cache-pressure needle-test --base-url http://my-server:8000/v1  # full sweep (50K -> 450K)
uvx --from cache-pressure needle-test --base-url http://my-server:8000/v1 \
    --lengths 50000,100000                                                  # subset
```

Key flags: `--lengths` (50000,100000,200000,300000,450000), `--needle-pos`
(0.8), `--max-tokens` (128), `--timeout` (1200), `--salt`.

`needle-test` calibrates against its own target lengths and does not need
the advertised capacity (it only needs to stay under the engine's
`max_model_len`). The model is auto-detected like in `cache-pressure`; pass
`--base-url`/`--model` to point it at your endpoint.

### `agent-sim` — concurrent agent-session workload

Simulates real coding-agent sessions to validate that an engine actually
retains session contexts under concurrency. `--sessions` workers run as
concurrent threads (barrier start), each growing a unique main-agent context
turn by turn (default 5K steps, assistant replies replayed verbatim), then
spawning `--sub-windows` cold sub-agent contexts, then returning to the main
context for a finalize turn.

The headline metric is **finalize survival**: after a session's sub-agents
ran (while other sessions kept hammering the cache), is the ~`--main-tokens`
main context still served from cache? A healthy deployment answers yes for
every session. Per-turn records report reused tokens (from
`usage.prompt_tokens_details.cached_tokens`, falling back to
`timings.cache_n`, cross-checked against vLLM `/metrics` prefix-cache
counters) plus TTFT.

```bash
uvx --from cache-pressure agent-sim --base-url http://my-server:8000/v1 \
    --sessions 3 --main-tokens 150000 --sub-tokens 40000 --output run.json

# deterministic A/B (same salt = identical planned inputs)
uvx --from cache-pressure agent-sim --base-url http://my-server:8000/v1 \
    --sessions 3 --main-tokens 150000 --sub-tokens 40000 --salt 42
```

Key flags: `--sessions` (3), `--main-tokens` (150000), `--sub-tokens`
(40000), `--sub-windows` (2), `--step-tokens` (5000), `--max-concurrency`
(2), `--timeout` (1200), `--salt`.

While the sessions run, the tool paints a live progress view (one column
per session, redrawn in place at 8 fps): the main-context bar grows with
each step, the sub-window bar tracks the cold windows (the main bar holds
while subs run, then climbs back at finalize), and every column reports
phase/step, prefix-reuse %, and prefill (`pp`) / generation (`tg`)
throughput:

```
agent-sim · 3 sessions · 01:12
s0  main ███████████▌░░░░ 135k/150k   s1  main ██████████▌░░░░░ 140k/150k   s2  main ████████▌░░░░░░░ 128k/150k
    sub  ██░░░░░░ 1/2 · cold            sub  ░░░░░░░░ 0/2                   sub  ░░░░░░░░ 0/2
    main 17/30 · hit 99%               sub 1/2 · cold                      main 16/30 · hit 99%
    pp 12.4k tps · tg 46 tps           pp 9.1k tps · tg 52 tps             pp 7.7k tps · tg 44 tps
```

Columns stack vertically when the terminal is too narrow; piped output
and `--no-viz` fall back to plain text. When the view is auto-disabled
(e.g. when run through `uvx`, which pipes the child's stdout), pass
`--viz` to force it on.

`--max-context` defaults to the engine's advertised `max_model_len` (auto via
`GET /models`) and the workload is validated against it. Reasoning/thinking
is off by default (`reasoning_effort: none`) so each generated reply is plain
text that replays verbatim into the next request — the resend is byte-exact
what the engine cached. Pass `--thinking` to re-enable reasoning (note: a
thinking model may emit no visible content, in which case the placeholder
`[no output generated]` is replayed instead).

Deterministic A/B with `--salt` guarantees identical *planned* inputs
(main steps, sub windows, finalize chunk) across runs — but the replayed
assistant replies are model-generated, so the full conversation prefix is
not byte-identical between two runs. The finalize-survival metric measures
within-run retention and stays valid; just don't expect cross-run prefixes
to be identical.

### `abort-sim` — mid-thinking-abort prefix retention

Reproduces the everyday "stop the model mid-thinking, then re-orient it"
event and measures whether the processed prefix survives the abort:

1. **prime** — a `--context-tokens` (default 40K) lorem context plus a
   hard multi-step problem, thinking on (`reasoning_effort: xhigh`),
   streamed with a big `--max-tokens` budget.
2. **abort** — the client counts streamed tokens (reasoning + content) and
   closes the connection at `--thinking-tokens` (default 500), keeping
   exactly the tokens already sent.
3. **re-prompt** — a new request re-sends the conversation with the
   captured partial assistant turn (thinking replayed verbatim via
   `reasoning_content`) plus a new user instruction
   (`--reprompt-mode branch`, the re-orientation; `continuation` instead
   ends the conversation on the partial turn), with `max_tokens: 0` — a
   pure prefill probe.

The headline metric is the re-prompt's reuse fraction: a healthy engine
reuses the whole processed prefix (context + captured thinking); a full
re-prefill is the "massive cache miss" this bench exists to catch. Cached
tokens come from `usage.prompt_tokens_details.cached_tokens` (falling back
to `timings.cache_n`); with `--ninfer-log` the records are re-annotated
with the ground truth from ninfer's request log
(`prefix_cache_hit_tokens`, `computed_prefill_tokens`,
`prefix_reuse_path` — `root` marks a hard full re-prefill). A run passes
at `--min-reuse` (default 0.95) and never at a `root` path; the exit code
is 0 only if every run passes.

```bash
uvx --from cache-pressure abort-sim --base-url http://my-server:8000/v1 \
    --context-tokens 40000 --thinking-tokens 500 --output run.json

# ground truth from the engine's request log
uvx --from cache-pressure abort-sim --base-url http://my-server:8000/v1 \
    --ninfer-log /path/to/requests.jsonl --salt 42
```

Key flags: `--context-tokens` (40000), `--thinking-tokens` (500), `--runs`
(2), `--min-reuse` (0.95), `--reprompt-mode` (branch), `--timeout` (600),
`--salt`, `--ninfer-log`.

### `perf-sim` — speed vs context length

A performance benchmark, not a retention one: it grows a single context
from 0 toward the model's max context in full `--step-tokens` increments
(default 10k tokens of true lorem ipsum) and measures prefill (`pp`) and
generation (`tg`) throughput at every context level. Each step generates a
fixed `--output-tokens` budget (default 256) and replays the reply into
the next request, so the prefix is stable — responses included — the same
organic growth `agent-sim` simulates, but sequential (concurrency 1) and
timed.

Per step, the tool reports:

- `pp` — incremental prefill speed `(prompt − cached) / ttft`: the first
  step is a full cold prefill, later steps prefill only the new increment,
  so the curve is prefill speed at each context level;
- `tg` — generation speed `completion / (wall − ttft)`.

```bash
uvx --from cache-pressure perf-sim --base-url http://my-server:8000/v1 --output run.json

# deterministic A/B (same salt = identical planned inputs)
uvx --from cache-pressure perf-sim --base-url http://my-server:8000/v1 --salt 42
```

Key flags: `--step-tokens` (10000), `--output-tokens` (256),
`--runs` (3), `--max-context` (auto from `GET /models` `max_model_len`),
`--timeout` (1200), `--salt`, `--thinking`. The step count is the number
of full increments that fit under the max context (replies count toward
the growth), so the last step lands just short of the limit — no partial
increment. Each `--runs` pass replays the same planned chunks with a
fresh conversation; the summary's per-level `context`/`pp`/`tg` arrays
average the runs, which smooths out single-shot timing noise. `--output`
writes the per-step records (with a `run` index) plus those arrays. The
exit code is 0 only if every step of every run completed.

## Development

```bash
git clone <repo> && cd cache-pressure
uv sync                        # create .venv with the project installed

# unit tests, no cluster needed (run all six before calling anything done)
uv run python tests/test_cache_pressure.py
uv run python tests/test_needle.py
uv run python tests/test_ninfer_log.py
uv run python tests/test_agent_sim.py
uv run python tests/test_abort_sim.py
uv run python tests/test_perf_sim.py

# exercise the console scripts against the local checkout
# (uv run, not uvx — uvx may re-run a stale cached wheel)
uv run cache-pressure --help
uv run needle-test --help
uv run agent-sim --help
uv run abort-sim --help
uv run perf-sim --help
```

## Interpreting results

`retained % capacity` can exceed 100% — that is **not a bug**. The
advertised capacity is a worst-case reservation per active context token;
cached content (constant-size recurrent state, bounded sliding-window
groups, deduplicated blocks) packs denser, so the cache can physically hold
more context than the budget implies. One fixed deployment retained ~147%
of advertised capacity with zero evictions at 39K granularity; the
pre-fix engine kept ~52-65%.
