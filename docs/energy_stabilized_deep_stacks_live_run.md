# Energy-Stabilized Deep Stacks: live run results

**Status: real, executed results** (not a design doc). Two passes, both
live: an initial n=1-seed pass, and a follow-up multi-seed pass testing
the specific hypothesis it suggested — that `NBSC_ENERGY_REG_LAMBDA` is
more useful as a fine-tuning knob on top of `NBSC_NORMALIZE=true` than as
a standalone fix. Cora, canonical Planetoid split, NBSC architecture
throughout, using the code in `nbsc_energy_stabilized_deep_stacks.tar.gz`.

## Environment

- `rustc`/`cargo` **1.75.0** installed via `apt` (matches the crate's
  declared `rust-version = "1.75"`); no toolchain was present beforehand.
- `cargo build --features burn --lib` / `--example thesis_bench --release`
  compile cleanly.
- `cargo test --features burn --lib burn_layer`: **6/6 pass**, including
  the two added for this feature (`differentiable_energy_matches_cpu_diagnostic`,
  `differentiable_energy_is_actually_differentiable`).
- Data (Cora Planetoid split) is committed in the repo; no network needed.
- **Practical constraint that shaped both passes:** each terminal command
  in this environment has a hard wall-clock budget of roughly 200-250s.
  Depth-6/8 runs with `NBSC_ENERGY_REG_LAMBDA>0` cost noticeably more per
  epoch than baseline (two extra dense `2708x2708` matmuls per
  regularized layer per epoch), so epoch counts were tuned down —
  inconsistently, out of necessity — across the runs below. Several
  attempted runs (noted inline) timed out entirely and contributed no
  data. This is the single biggest limitation of everything below: **not
  every row used the same epoch budget**, so cross-depth comparisons of
  the raw accuracy numbers should be read with that in mind. Comparisons
  *within* a fixed depth at a fixed epoch count (i.e. down each column of
  a table below) are apples-to-apples; comparisons *across* depths are not.

## Pass 1 (previous turn): n=1 seed, initial signal

30 epochs, `NBSC_N_SEEDS=1`, seed 0 only.

| depth | normalize | lambda | val acc | test acc | final energy |
|---:|:---:|---:|---:|---:|---:|
| 2 | false | 0.0 | 0.710 | 0.745 | 10.24 |
| 2 | false | 0.1 | 0.680 | 0.677 | 0.53 |
| 6 | false | 0.0 | 0.438 | 0.448 | 16.13 |
| 6 | false | 0.001 | 0.382 | 0.411 | 14.69 |
| 6 | false | 0.01 | 0.430 | 0.426 | 4.02 |
| 6 | false | 0.1 | 0.376 | 0.392 | 2.36 |
| 6 | false | 1.0 | 0.232 | 0.232 | 1.01 |
| 6 | true | 0.0 | 0.620 | 0.614 | 0.43 |
| 6 | true | 0.01 | **0.636** | **0.651** | 0.38 |

This pass's headline: at depth 6, `normalize=true` + lambda=0.01 beat
`normalize=true` alone (0.651 vs. 0.614), suggesting the energy penalty
might be a useful add-on to LayerNorm rather than a replacement for it.
**Pass 2 below was designed specifically to check whether that survives
a second seed — it does not, cleanly.**

## Pass 2 (this turn): multi-seed grid, `normalize=true` only

All rows: `normalize=true`, testing exactly the grid suggested —
lambda in {0, 0.001, 0.01, 0.1} at depths {4, 6, 8}.

### Depth 4 — 2 seeds, 25 epochs

| lambda | seed 0 test | seed 1 test | mean test | final energy (seed 0) |
|---:|---:|---:|---:|---:|
| 0.0 | 0.654 | 0.580 | 0.617 | 0.419 |
| 0.001 | 0.666 | 0.582 | 0.624 | 0.410 |
| 0.01 | 0.678 | 0.592 | 0.635 | 0.405 |
| 0.1 | 0.697 | 0.625 | **0.661** | 0.306 |

**Clean, monotonic pattern, both seeds:** every lambda increase improved
test accuracy for both seed 0 and seed 1, with a roughly constant
~0.07-0.09 gap between the two seeds throughout. This is the strongest,
most reproducible-looking result in either pass.

### Depth 6 — 2 seeds, 25 epochs

| lambda | seed 0 test | seed 1 test | mean test | final energy (seed 0) |
|---:|---:|---:|---:|---:|
| 0.0 | 0.599 | 0.522 | 0.561 | 0.437 |
| 0.001 | 0.639 | 0.540 | **0.590** | 0.415 |
| 0.01 | 0.593 | 0.509 | 0.551 | 0.396 |
| 0.1 | 0.586 | 0.564 | 0.575 | 0.325 |

**Pass-1's depth-6 finding does not replicate.** lambda=0.01 was the
standout in pass 1 (single seed); here, with a second seed added,
lambda=0.01 is statistically indistinguishable from the lambda=0
baseline (0.551 vs. 0.561 mean), and lambda=0.001 is now the best mean
instead. Seed 1 alone runs 0.05-0.08 lower than seed 0 at every lambda —
a swing large enough to explain pass 1's result as likely a favorable
roll of the (single) seed rather than a real lambda=0.01 effect.
**Conclusion: the depth-6 sweet-spot claim from pass 1 is retracted; two
seeds isn't enough to replace it with a confident alternative, only
enough to show the original claim doesn't hold.**

### Depth 8 — mixed seed count (see note), mixed epochs (see note)

| lambda | epochs | seed 0 test | seed 1 test | mean test |
|---:|---:|---:|---:|---:|
| 0.0 | 20 | 0.503 | 0.430 | 0.467 |
| 0.001 | 20 | 0.473 | *(timed out, no data)* | 0.473 (n=1) |
| 0.01 | 20 | 0.477 | *(not attempted — budget)* | 0.477 (n=1) |
| 0.1 | 20 | 0.552 | *(not attempted — budget)* | 0.552 (n=1) |

Depth-8 runs with the regularizer active consistently approached or
exceeded the per-command time budget even at 20 epochs; a second seed
for lambda=0.001 was attempted and timed out mid-run (seed 0 for that
row did complete and is real data). lambda=0.01 and lambda=0.1 second
seeds were not attempted after that failure, to conserve remaining
budget for writing this report. **Depth 8 numbers above should be
treated as the weakest evidence in this document** — only the baseline
has any seed replication at all, and 20 epochs is likely undertrained
relative to depths 4 and 6 (compare: depth-8 lambda=0 at 15 epochs
scored a mean of only 0.397, vs. 0.467 at 20 epochs — accuracy was still
climbing between those two runs, so 20 epochs is not obviously converged
either).

## Overall interpretation

- **The specific question asked this turn — does a small lambda give a
  consistent edge on top of LayerNorm — has one clear "yes" (depth 4)
  and one clear "not confirmed, walked back" (depth 6).** Depth 8 doesn't
  yet have enough seed coverage to say either way.
- Depth 4 is shallow enough that LayerNorm alone hasn't fully saturated
  the achievable accuracy, and the energy penalty appears to add a real,
  seed-stable improvement on top of it, up to the largest lambda tried
  (0.1). It's untested whether lambda>0.1 continues to help or eventually
  hurts the way it did at depth 6 in pass 1.
- Depth 6 shows real seed-to-seed variance (up to ~0.08 in test accuracy)
  that a single seed cannot distinguish from a genuine lambda effect.
  This is itself the most actionable finding here: **single-seed numbers
  on this benchmark are not trustworthy enough to rank lambda values**,
  which directly validates the reason this multi-seed pass was requested.
- No claim in this document should be read as "energy regularization
  works" or "doesn't work" in general — only as: at depth 4, in this
  narrow setup, it helped consistently across 2 seeds; at depth 6, one
  seed's apparent win did not survive a second seed; at depth 8, there
  isn't enough data yet to say anything with confidence.

## Suggested next steps (in priority order)

1. **Depth 8 needs its second seed** for lambda in {0.001, 0.01, 0.1}
   before it can be compared to depths 4/6 at all — right now it's the
   least-supported part of the grid.
2. **A 3rd seed at depths 4 and 6** would meaningfully firm up both the
   depth-4 "clean win" and the depth-6 "no effect" readings — 2 seeds is
   the minimum that can show disagreement, not enough to be confident
   either pattern is real.
3. **Match epoch counts across depths** (this pass used 25 for depths
   4/6 and 20 for depth 8 out of necessity) — ideally by finding a
   faster environment/more time budget, since epoch count differences
   are currently confounded with depth in every cross-depth comparison.
4. Depth 4's clean monotonic trend up to lambda=0.1 raises the obvious
   question of whether lambda=1.0 (already known to collapse training at
   depth 6, pass 1) also hurts at depth 4, or whether depth 4 tolerates
   larger lambda than deeper stacks do — untested.

## Raw CSV (both passes, all attempts, as written by `NBSC_CSV_OUT`)

```csv
dataset,split_mode,split_seed,depth,weight_decay,normalize,energy_reg_lambda,architecture,seed,val_acc,test_acc,final_energy
cora,canonical,canonical,6,0,false,0.01,nbsc,0,0.4300,0.4260,4.020301
cora,canonical,canonical,6,0,false,0,nbsc,0,0.4380,0.4480,16.134346
cora,canonical,canonical,6,0,false,0.001,nbsc,0,0.3820,0.4110,14.693432
cora,canonical,canonical,6,0,false,0.1,nbsc,0,0.3760,0.3920,2.358973
cora,canonical,canonical,6,0,false,1,nbsc,0,0.2320,0.2320,1.008236
cora,canonical,canonical,6,0,true,0,nbsc,0,0.6200,0.6140,0.434174
cora,canonical,canonical,6,0,true,0.01,nbsc,0,0.6360,0.6510,0.382993
cora,canonical,canonical,2,0,false,0,nbsc,0,0.7100,0.7450,10.239665
cora,canonical,canonical,2,0,false,0.1,nbsc,0,0.6800,0.6770,0.531330
cora,canonical,canonical,4,0,true,0,nbsc,0,0.6320,0.6550,0.410197
cora,canonical,canonical,4,0,true,0.001,nbsc,0,0.6420,0.6650,0.420977
cora,canonical,canonical,4,0,true,0.01,nbsc,0,0.6680,0.6940,0.402392
cora,canonical,canonical,4,0,true,0.1,nbsc,0,0.7180,0.7180,0.271974
cora,canonical,canonical,6,0,true,0.001,nbsc,0,0.6200,0.6360,0.390406
cora,canonical,canonical,6,0,true,0.1,nbsc,0,0.5760,0.5780,0.302093
cora,canonical,canonical,8,0,true,0,nbsc,0,0.5360,0.5480,0.374367
cora,canonical,canonical,8,0,true,0.001,nbsc,0,0.6060,0.5810,0.372564
cora,canonical,canonical,8,0,true,0.01,nbsc,0,0.5320,0.5530,0.397872
cora,canonical,canonical,8,0,true,0.1,nbsc,0,0.5760,0.5850,0.268874
cora,canonical,canonical,4,0,true,0,nbsc,0,0.6320,0.6550,0.410197
cora,canonical,canonical,6,0,true,0,nbsc,0,0.5960,0.5990,0.437185
cora,canonical,canonical,6,0,true,0,nbsc,1,0.5000,0.5220,0.327903
cora,canonical,canonical,4,0,true,0,nbsc,0,0.6340,0.6540,0.419025
cora,canonical,canonical,4,0,true,0,nbsc,1,0.6020,0.5800,0.386033
cora,canonical,canonical,4,0,true,0.001,nbsc,0,0.6360,0.6660,0.409884
cora,canonical,canonical,4,0,true,0.001,nbsc,1,0.6080,0.5820,0.380582
cora,canonical,canonical,4,0,true,0.01,nbsc,0,0.6620,0.6780,0.405393
cora,canonical,canonical,4,0,true,0.01,nbsc,1,0.6280,0.5920,0.373816
cora,canonical,canonical,4,0,true,0.1,nbsc,0,0.6940,0.6970,0.306006
cora,canonical,canonical,4,0,true,0.1,nbsc,1,0.6440,0.6250,0.307595
cora,canonical,canonical,6,0,true,0.001,nbsc,0,0.6020,0.6390,0.414925
cora,canonical,canonical,6,0,true,0.001,nbsc,1,0.4900,0.5400,0.315358
cora,canonical,canonical,6,0,true,0.01,nbsc,0,0.5720,0.5930,0.395985
cora,canonical,canonical,6,0,true,0.01,nbsc,1,0.4800,0.5090,0.318558
cora,canonical,canonical,6,0,true,0.1,nbsc,0,0.5440,0.5860,0.325058
cora,canonical,canonical,6,0,true,0.1,nbsc,1,0.5400,0.5640,0.258298
cora,canonical,canonical,8,0,true,0,nbsc,0,0.4420,0.4420,0.329531
cora,canonical,canonical,8,0,true,0,nbsc,1,0.3540,0.3530,0.533111
cora,canonical,canonical,8,0,true,0,nbsc,0,0.4960,0.5030,0.378154
cora,canonical,canonical,8,0,true,0,nbsc,1,0.4400,0.4300,0.484539
cora,canonical,canonical,8,0,true,0.001,nbsc,0,0.4860,0.4730,0.397009
cora,canonical,canonical,8,0,true,0.01,nbsc,0,0.4900,0.4770,0.377923
cora,canonical,canonical,8,0,true,0.1,nbsc,0,0.5620,0.5520,0.274223
```

Note: several `(depth, lambda, seed=0)` combinations appear more than
once in this raw log at different epoch counts (30 -> 25 -> 20/15 as the
epoch budget was tuned down through the session to fit the time limit);
the tables above cite the specific run used for each cell, not an
average across duplicates.
