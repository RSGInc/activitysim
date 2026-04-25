# RNG Investigation Benchmarks

This folder contains off-model experiments for ActivitySim random-number-generation questions.

## Why `eet_keyed_maz_test.py` exists

The explicit-error-term destination choice path currently protects chooser-alternative shock (aka random number) invariance by drawing a dense vector of EV1 shocks up to `max_alt_id + 1` and then gathering the sampled alternatives back out. That is simple and correct, but it becomes expensive when MAZ ids are sparse. The placeholder two-zone example is exactly the kind of case that makes this visible because the MAZ ids are sparse while the actual sampled set is small.

The benchmark in `benchmark_rng.py` is useful for broad RNG experimentation, but it is much wider than the question under discussion here. The newer `eet_keyed_maz_test.py` narrows the test down to the exact tradeoff we care about:

- Dense max-id gather: the current behavior, drawing over `max_alt_id + 1`
- Compressed alternative remap: remap actual MAZ ids to a compressed dense code space and still draw densely within that compressed space
- Keyed chooser-alt strategies: generate shocks directly from `(chooser_seed, maz_id)` so only sampled alternatives are touched

The keyed family currently includes several backend variants so the benchmark can separate access-pattern gains from generator-choice gains:

- keyed hash
- keyed Philox
- keyed PCG64

## What the new benchmark actually measures

This script intentionally does not try to reproduce the full ActivitySim destination workflow. It isolates the two pieces that matter for the design decision:

- `shock_lookup`: how expensive it is to generate or recover the EV1 shock for each sampled MAZ
- `final_choice`: the same shocks plus a simple argmax over deterministic utility plus shock

That keeps the benchmark small enough to reason about while still answering the practical question: is a keyed chooser-plus-MAZ design materially better than a compressed-id version of the current approach?

The benchmark also includes an offset sweep. That matters because the current ActivitySim RNG design advances each chooser stream by an offset as draws are consumed within a step. A replacement design is more attractive if it is not just faster for sparse ids, but also less sensitive to larger offsets.

## Inputs and invariants

Each benchmark scenario creates synthetic chooser seeds using the same ActivitySim-style row-seed formula and builds sampled MAZ choice sets with configurable:

- chooser count
- sample size
- number of actual MAZ alternatives
- maximum MAZ id
- overlap ratio between two sampled sets for the same choosers
- offset before the tested draw

The overlap ratio exists because the whole point of the current dense implementation is invariance. The script therefore checks two invariants before timing results are interpreted:

- batch invariance: the same chooser gets the same shocks regardless of batch membership or row order
- sampled-set invariance: a chooser-MAZ pair keeps the same shock when the sampled choice set changes
- offset sensitivity: nonzero offsets actually change the draw state while preserving the invariance guarantees above

If an option fails those checks, its timing is not enough to justify it.

## Output files

By default the script writes to `output/eet_keyed_maz_test/`:

- `eet_keyed_maz_test_results.csv`: one row per scenario, kernel, and strategy
- `eet_keyed_maz_test_invariance.csv`: pass/fail results for the invariance checks
- `eet_keyed_maz_test_summary.md`: compact interpretation of the metrics
- `plots/*.png`: runtime, memory, throughput, waste, and speedup plots

The most important output columns are:

- `generated_shocks_per_chooser`: how many shocks a strategy had to create
- `useful_shocks_per_chooser`: how many shocks were actually needed for the sampled alternatives
- `waste_factor`: generated divided by useful shocks
- `useful_shocks_per_sec`: comparable throughput across different strategies
- `offset`: the number of prior draws consumed before the benchmarked lookup
- `mean_seconds` and `peak_memory_mb`: direct runtime and memory cost

## How to run it

From the repository root:

```powershell
"c:/Users/david.hensle/OneDrive - Resource Systems Group, Inc/Documents/projects/activitysim/rsg_activitysim/.venv/Scripts/python.exe" other_resources/rng_investigation/eet_keyed_maz_test.py --profile fast
```

For a larger run:

```powershell
"c:/Users/david.hensle/OneDrive - Resource Systems Group, Inc/Documents/projects/activitysim/rsg_activitysim/.venv/Scripts/python.exe" other_resources/rng_investigation/eet_keyed_maz_test.py --profile full
```

## Current results

The current full-profile run says three things clearly.

- All tested strategies passed the benchmark invariance checks, including the offset checks at 0 and 16.
- The compressed alternative remap is the best practical result in the current code family. In the high-sparsity chooser and offset sweeps, it was typically about 18x to 23x faster than dense max-id gather while reducing peak memory by about 2.7x for `final_choice` and about 5x for `shock_lookup`.
- The keyed hash prototype is dramatically faster than everything else, but it is not a drop-in RNG replacement. In the same high-sparsity cases it was roughly 300x to 500x faster than dense max-id gather, while using somewhat more peak memory than dense max-id gather and much more than compressed alternative remap because the prototype vectorizes over the full sampled chooser-alt matrix.
- The keyed Philox and keyed PCG64 variants are not competitive in their current per-chooser-alt form. They preserved invariance, but were generally slower than dense max-id gather and much slower than compressed alternative remap, even though their peak memory was lower than dense max-id gather.

The bottom line from the current benchmark results is that the performance win is not coming from switching NumPy bit generators by itself. The win comes from changing the access pattern so the code stops drawing over the dense MAZ id span.

## Recommended approach

The recommended implementation path from these results is:

- First choice: implement the compressed alternative remap in production code.
- Do not pursue the current keyed Philox or keyed PCG64 designs as production candidates in their present form.
- Keep the keyed hash result as a research signal, not as the immediate implementation target.

The reason to recommend the compressed alternative remap first is that it already captures most of the practical benefit that the current model structure can use without needing a new RNG model. It preserves the current semantics, removes the dependence on the dense MAZ id span, and delivered consistent double-digit speedups in the sparse-id cases that motivated this investigation.

The keyed hash prototype shows that a custom stateless keyed design could be much faster still, but that result should be treated as a proof of concept for the access pattern, not as evidence that a conventional keyed `Philox` or `PCG64` implementation will work well. If there is still a need to go beyond the compressed remap after that change lands, the next investigation should be a custom keyed or counter-based design, not per-pair generator construction with standard NumPy bit generators.

## Estimating impact from TAZs, MAZs, and choosers

Use the following notation:

- `C`: number of choosers
- `T`: number of TAZs
- `M`: number of actual MAZs
- `I`: MAZ id span used by the current dense method, usually `max_maz_id + 1`
- `S`: sampled TAZs per chooser entering the MAZ expansion stage
- `K`: sampled MAZ alternatives per chooser at the EET choice step

If you do not already know `K`, a reasonable first estimate is:

`K ≈ min(M, S * (M / T))`

That is: sampled TAZs per chooser times the average number of MAZs per TAZ.

### Draw-count estimates

For the current dense max-id gather:

`generated_shocks_dense ≈ C * I`

For the compressed alternative remap:

`generated_shocks_compressed ≈ C * M`

For a keyed chooser-alt strategy:

`generated_shocks_keyed ≈ C * K`

From this, the exact draw-count reduction factors are:

- Dense to compressed: `I / M`
- Dense to keyed: `I / K`
- Compressed to keyed: `M / K ≈ T / S`

That means `T` affects the estimate mainly through `K`. If the average number of MAZs per TAZ is high or the model samples many TAZs per chooser, the keyed family gets less additional benefit over the compressed remap.

### Runtime estimates for the recommended approach

For the compressed alternative remap, the benchmark suggests a simple rule of thumb for sparse-id cases like the ones that motivated this work:

`expected_speedup_vs_dense ≈ 0.15 to 0.20 * (I / M)`

This rule is empirical, based on the full-profile benchmark, and is most useful when `I / M` is large.

Examples:

- If `I / M ≈ 16`, expect about 2x to 3x faster runtime.
- If `I / M ≈ 64`, expect about 10x to 13x faster runtime.
- If `I / M ≈ 128`, expect about 19x to 26x faster runtime.

That matches the current full-profile results reasonably well. At the benchmark's high-sparsity setting with `M = 128`, `I = 16385`, and `C = 250` to `2000`, the compressed remap was about 18x to 23x faster than dense max-id gather in the chooser and offset sweeps.

If `I / M` is near 1, the compressed remap should still reduce waste bookkeeping, but the runtime win will be small because there is little dense-id overhead to remove.

### Peak-memory estimates for the recommended approach

The temporary storage shape changes from depending on the dense id span to depending on the actual MAZ count:

- Dense max-id gather: roughly `O(I + C * K)`
- Compressed alternative remap: roughly `O(M + C * K)`

So the dense vector term shrinks by the same factor `I / M`.

In the current full-profile run, the observed peak-memory reduction for compressed remap versus dense max-id gather was:

- about 2.7x lower for `final_choice` in the high-sparsity chooser and offset sweeps
- about 5x lower for `shock_lookup` in the same cases

As a practical estimate, if your MAZ ids are very sparse, you should expect a several-fold peak-memory reduction from the compressed remap even when the chooser-output matrix still dominates part of the allocation.

### Worked example

Suppose you have:

- `C = 100000` choosers
- `T = 200` TAZs
- `M = 25000` actual MAZs
- sampled TAZs per chooser `S = 4`
- dense MAZ id span `I = 1000000`

Then:

- `K ≈ S * (M / T) ≈ 4 * 125 = 500`
- dense generated shocks `≈ 100000 * 1000000 = 100000000000`
- compressed generated shocks `≈ 100000 * 25000 = 2500000000`
- keyed generated shocks `≈ 100000 * 500 = 50000000`

So the compressed remap removes about `1000000 / 25000 = 40x` of the dense-id draw burden. Using the benchmark rule of thumb, a realistic runtime gain would be on the order of about `6x` to `8x` rather than the full `40x`, because fixed overheads remain. Peak memory should also drop materially because the dense temporary vector shrinks from about one million entries to about twenty-five thousand entries.

For that same example, a keyed chooser-alt strategy would reduce draw count by another factor of about `25000 / 500 = 50x` over the compressed remap, but the current results say that this only looks promising when the keyed implementation is a custom stateless design. The current Philox and PCG64 prototypes do not realize that theoretical gain.