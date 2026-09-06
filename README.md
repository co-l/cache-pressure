# cache-pressure

Measure how much context a vLLM prefix cache **actually retains** under
overflow pressure — not what the engine *advertises*.

The advertised "GPU KV cache size" (the `GPU KV cache size` log line /
`kv_cache_size_tokens`) is an **active-request planning budget**, not a
ceiling on cacheable content. Hybrid layouts (MLA + sparse-attention +
recurrent-state groups) pack cached blocks denser than that reservation,
and cache-management defects can waste the headroom with duplicate blocks
and unreachable replay tails. This tool quantifies the real number, so you
can tell a healthy cache from a leaking one on any deployment.

## How it works

```
1. capacity   advertised KV size (--kv-size, or auto-read from the engine
              log over ssh with --ssh-host)
2. calibrate  one probe request: real tokens/char for this tokenizer at
              the target length (length-dependent, iterates to convergence)
3. hydrate    N unique ~39K-token contexts sequentially (max_tokens=1,
              prefill-only — the full context is committed regardless)
              N = ceil(capacity/40000)+5, so the cache overflows
4. verify     re-send each context in reverse order (newest first),
              classify hit/miss by TTFT (cold prefill ~seconds, hit ~0.5s)
              → the first miss is the oldest evicted context = the real
              retained capacity. Stops there: under LRU everything older
              is evicted by construction.
```

Contexts are seeded random word-orders (unique per salt within a run,
byte-identical across runs) so no context rides on another's cached
content, and two runs (e.g. before/after a fix) replay the same text.

## Usage

```bash
pip install -r requirements.txt

# capacity auto-read from the engine log over ssh
python3 bench/cache_pressure.py --ssh-host mynode --output run.json

# capacity passed explicitly (no ssh needed)
python3 bench/cache_pressure.py --kv-size 2000000 --output run.json

# A/B two runs
python3 bench/cache_pressure.py --compare fix.json control.json
```

Small sanity check (3 contexts, all hits):

```bash
python3 bench/cache_pressure.py --kv-size 2000000 --num-contexts 3
```

### `needle_test.py` — companion correctness check

The retention number means nothing if the engine is broken. This sweeps
increasing context lengths (default 50K/100K/200K/300K/450K), hides a
needle sentence at 0.8 depth in each unique haystack, and requires the
model to output the exact secret code — proving long-context retrieval is
intact end-to-end.

```bash
python3 bench/needle_test.py                        # full sweep (50K -> 450K)
python3 bench/needle_test.py --lengths 50000,100000 # subset
```

`needle_test.py` calibrates against its own target lengths and does not
need the advertised capacity (it only needs to stay under the engine's
`max_model_len`). Pass `--base-url`/`--model` to point it at your endpoint.

## Tests

```bash
python3 tests/test_cache_pressure.py    # no cluster needed
```

## Interpreting results

`retained % capacity` can exceed 100% — that is **not a bug**. The
advertised capacity is a worst-case reservation per active context token;
cached content (constant-size recurrent state, bounded sliding-window
groups, deduplicated blocks) packs denser, so the cache can physically hold
more context than the budget implies. A healthy fixed engine retained
~147% of advertised capacity with zero evictions at 39K granularity; the
pre-fix engine kept ~52-65%. See the
[ds4-prefix-cache-fixes](https://github.com/co-l/ds4-prefix-cache-fixes)
write-up for the full investigation.
