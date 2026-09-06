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

```bash
pip install -r requirements.txt

# point --base-url at your OpenAI-compatible endpoint; the model is
# auto-detected via GET /models (pass --model to override)
python3 bench/cache_pressure.py --base-url http://my-server:8000/v1 \
    --kv-size <ADVERTISED_KV_CACHE> --output run.json

# A/B two runs
python3 bench/cache_pressure.py --compare fix.json control.json
```

Small sanity check (3 contexts, all hits):

```bash
python3 bench/cache_pressure.py --base-url http://my-server:8000/v1 \
    --kv-size <ADVERTISED_KV_CACHE> --num-contexts 3
```

### `needle_test.py` — companion correctness check

The retention number means nothing if the engine is broken. This sweeps
increasing context lengths (default 50K/100K/200K/300K/450K), hides a
needle sentence at 0.8 depth in each unique haystack, and requires the
model to output the exact secret code — proving long-context retrieval is
intact end-to-end.

```bash
python3 bench/needle_test.py --base-url http://my-server:8000/v1  # full sweep (50K -> 450K)
python3 bench/needle_test.py --base-url http://my-server:8000/v1 \
    --lengths 50000,100000                                        # subset
```

`needle_test.py` calibrates against its own target lengths and does not
need the advertised capacity (it only needs to stay under the engine's
`max_model_len`). The model is auto-detected like in `cache_pressure.py`;
pass `--base-url`/`--model` to point it at your endpoint.

## Tests

```bash
python3 tests/test_cache_pressure.py    # no cluster needed
```

## Interpreting results

`retained % capacity` can exceed 100% — that is **not a bug**. The
advertised capacity is a worst-case reservation per active context token;
cached content (constant-size recurrent state, bounded sliding-window
groups, deduplicated blocks) packs denser, so the cache can physically hold
more context than the budget implies. One fixed deployment retained ~147%
of advertised capacity with zero evictions at 39K granularity; the
pre-fix engine kept ~52-65%.
