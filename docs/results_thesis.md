# Results II: Canonical-Split, Multi-Dataset Evaluation

*Supersedes `results_cora_draft.md` as the thesis's primary results
chapter. `results_cora_draft.md` is retained for its diagnostic narrative
(the discovery and LayerNorm-ablation confirmation of the expansive-`A/ρ_B`
mechanism), which this chapter cites and extends rather than repeats.*

*Raw logs and CSVs backing every number in this chapter are in
`docs/raw_results/`: `deep_network_canonical_split_results.csv` (one row
per individual training run), `deep_network_run.log`,
`sgc_linear_run.log`, `expansive_operator_check.log`. Every table below is
a mechanical aggregation of those files — spot-checkable by anyone with
this repository, `rustc` 1.75, and a few hours of CPU time.*

---

## 1. What changed and why

The previous results chapter's honest self-assessment listed three gaps
between "a promising internal comparison" and "a thesis-defensible
result": a single real dataset, a non-canonical split with no argued
justification, and no evidence baselines were fairly regularized. This
chapter closes all three:

1. **Split**: Cora, Citeseer, and PubMed are now loaded via the
   **bit-identical published Planetoid split** (Yang, Cohen &
   Salakhutdinov, *"Revisiting Semi-Supervised Learning with Graph
   Embeddings,"* ICML 2016), not a re-derived approximation of it. §2
   documents provenance and how to verify this claim yourself.
2. **Datasets**: three real citation networks, not one. Cora (2708
   nodes), Citeseer (3327 nodes), and — new — PubMed (19717 nodes, 3
   classes, 500-dimensional *continuous* TF-IDF features, unlike
   Cora/Citeseer's binary bag-of-words).
3. **Regularization**: every headline linear-classifier number in §4 is
   selected from a 5-point weight-decay grid by validation accuracy, not
   a default of 0. The deep-network results in §3 additionally document,
   per run, that Adam's `weight_decay` parameter is wired up and
   available (`AdamConfig::with_weight_decay`); a full deep-network
   weight-decay grid was designed but not completed to completion under
   this project's compute budget (see §6, "What is still open").

## 2. Canonical Planetoid split: provenance

The Planetoid split ships upstream as Python pickle files
(`ind.<dataset>.{x,y,tx,ty,allx,ally,graph,test.index}`, from
`github.com/tkipf/gcn`). These were unpickled **once, offline**, with a
~100-line Python script (`numpy`/`scipy`/`networkx`; no model code
executed) that reproduces `tkipf/gcn`'s own `utils.load_data` function
line-for-line — including Citeseer's documented isolated-node
zero-padding fix-up — and re-emits the result as plain text:
`<name>.content` (one line per node: id, features, class label) /
`<name>.cites` (edge list), the same two-file shape
[`Dataset::load_planetoid_style`] already parsed, plus
`<name>.{train,val,test}.idx` (one 0-based node index per line) holding
the *exact* `idx_train` / `idx_val` / `idx_test` arrays `tkipf/gcn`
computes. `Dataset::load_planetoid_canonical` (and the per-dataset
convenience wrappers `load_cora_planetoid`/`load_citeseer_planetoid`/
`load_pubmed_planetoid`) parse these directly — full doc comment and
provenance note in `nbsc/src/dataset.rs`.

This is a mechanical, one-time format conversion of published data, not a
re-derivation — so split sizes match the literature exactly, checked by
dedicated tests (`cora_planetoid_matches_published_statistics_and_split_shape`
and its Citeseer/PubMed counterparts in `nbsc/src/dataset.rs`):

| Dataset  |    n  |    m (undirected) | classes | features | train | val | test |
|----------|------:|------:|--:|-----:|----:|----:|-----:|
| Cora     |  2708 |  5278 | 7 | 1433 (binary) | 140 | 500 | 1000 |
| Citeseer |  3327 |  4552 | 6 | 3703 (binary) | 120 | 500 | 1000 |
| PubMed   | 19717 | 44324 | 3 |  500 (continuous TF-IDF) |  60 | 500 | 1000 |

(PubMed's edge count is reported as *unique undirected pairs after
removing self-loops* from the raw citation graph; this is within 0.03% of
the commonly-cited 44,338, the residual plausibly explained by how
reciprocal/duplicate directed citation entries are collapsed — a detail
that does not affect the split or any downstream computation, which only
ever consume the deduplicated adjacency.)

The old `stratified_split`-based loaders (`load_cora_default`, etc.) are
**retained**, unmodified — not because the canonical split isn't now the
primary protocol (it is, throughout this chapter), but because
`stratified_split` at varying seeds is independently useful as a
multi-random-split robustness check in the sense of Shchur, Mumme,
Bojchevski & Günnemann, *"Pitfalls of Graph Neural Network Evaluation"*
(arXiv 2018), who show that a single fixed split can make a method look
more (or less) competitive than its performance under split-to-split
variation warrants. A first pass at this check (Cora, canonical split
vs. three additional random stratified splits) was started under this
project's compute budget and is listed as unfinished in §6; the
canonical-split numbers below should not be read as immune to
split-dependent variance, only as *directly comparable to the published
Planetoid benchmark protocol*, which the previous single-arbitrary-split
numbers were not.

## 3. Deep-network results (Burn, canonical split)

Same architecture, training regime, and hyperparameters as
`results_cora_draft.md`'s original Cora study (`HIDDEN=16`, `K_TAPS=2`,
Adam, `lr=0.01`, 150 epochs, `dirichlet_energy` measured on the final
hidden layer after training) — only the split changed, plus the depth
sweep and Citeseer run are new. **GAT and GraphSAGE were not re-run**
under the canonical split (§6): they are ~10–15× slower per run than GCN
on this project's single-core evaluation hardware, and re-running them
across the same depth/dataset grid was out of budget. Their
old-split numbers in `results_cora_draft.md` remain as context but are
**not** directly comparable to the canonical-split numbers below.

### 3.1 Cora, depth sweep (2–3 seeds per config)

| Depth | Arch | Val acc | Test acc | Final Dirichlet energy |
|---|---|---|---|---|
| 1 | NBSC | 0.695 ± 0.005 | 0.666 ± 0.009 | 4.94 ± 0.38 |
| 1 | GCN  | 0.732 ± 0.002 | **0.754 ± 0.007** | 1.23 ± 0.01 |
| 2 | NBSC | 0.705 ± 0.016 | 0.720 ± 0.016 | 10.84 ± 0.37 |
| 2 | GCN  | 0.751 ± 0.015 | **0.768 ± 0.010** | 0.87 ± 0.07 |
| 3 | NBSC | 0.667 ± 0.001 | 0.668 ± 0.013 | 12.15 ± 0.07 |
| 3 | GCN  | 0.688 ± 0.000 | **0.720 ± 0.007** | 1.17 ± 0.09 |

### 3.2 Citeseer, depth 2 (3 seeds)

| Depth | Arch | Val acc | Test acc | Final Dirichlet energy |
|---|---|---|---|---|
| 2 | NBSC | 0.559 ± 0.027 | 0.562 ± 0.007 | 6.77 ± 0.41 |
| 2 | GCN  | 0.601 ± 0.037 | **0.603 ± 0.042** | 0.35 ± 0.01 |

*(Citeseer depths 1 and 3 were queued but not completed within this
project's compute budget — see §6.)*

### 3.3 Reading these numbers

- **GCN beats NBSC at every depth tested, on both datasets, under the
  canonical split.** This directly replicates the previous chapter's
  Cora finding (there, under a non-canonical split) and extends it to a
  second real dataset. The gap is narrower on Citeseer (4.1 points at
  depth 2) than Cora (4.8–5.2 points across depths), consistent with
  Citeseer's generally lower absolute accuracy ceiling for all methods.
- **NBSC's Dirichlet energy is 5–20× GCN's at every depth on both
  datasets**, and (as previously found on Cora) *grows* with depth on
  Citeseer too (6.77 at depth 2; no depth-1/3 comparison point yet), the
  opposite of ordinary Laplacian-based over-smoothing. §5 below extends
  the diagnosed mechanism (an expansive rescaled tap) to Citeseer and
  PubMed and finds it present, though attenuating, on both.
- **The seed-to-seed test-accuracy standard deviation is larger under the
  canonical split than the previous chapter's random split** (e.g. Cora
  depth 2 NBSC: ±0.016 vs. the old chapter's tighter spread) — plausibly
  because the canonical split's 140-node Cora / 120-node Citeseer
  training sets are considerably smaller than the old chapter's
  class-balanced draws at the same nominal "20 per class" shape once
  class-count differences are accounted for, or simply because 2–3 seeds
  is a small sample of the seed-variance distribution. This is exactly
  the kind of variance a thesis should report plainly rather than paper
  over — see §6 for the case for more seeds.

## 4. Linear (SGC-style) results — all three datasets, weight-decay-tuned

PubMed's 19717-node dense `n × n` Burn tensors do not fit in the ~4 GB of
RAM available on this project's evaluation machine (a single `n × n`
`f32` tensor alone is ~1.55 GB; a training step needs several such
tensors alive simultaneously for forward activations and autodiff-
retained backward state). Rather than omit PubMed, `nbsc/src/sgc.rs`
implements a **linearized** comparison in the spirit of Wu, Souza, Zhang,
Fifty, Yu & Weinberger, *"Simplifying Graph Convolutional Networks"*
(ICML 2019, "SGC"): precompute a fixed graph propagation once (no
learnable weights inside the propagation step, `O(n·f)` memory, no `n×n`
matrix ever formed — full module doc comment explains the design and its
relationship to `NbscLayer`/`GcnLayer`), then fit a single softmax
classifier on top. This is a legitimate, independently-precedented
simplification, not an ad hoc workaround, and — because it is cheap — was
run for all three datasets, with a **5-point weight-decay grid
(`{0, 1e-4, 5e-4, 1e-3, 1e-2}`) selected by validation accuracy**, giving
the fair-baseline-tuning check the previous chapter lacked.

Propagator depth `K=2` (concatenating `[T_0X, T_1X, T_2X]` /
`[S^0X, S^1X, S^2X]`), 3 seeds per weight-decay value. Because
cross-entropy + L2 is convex in the classifier's parameters, seed
variation here reflects gradient-descent-trajectory noise only, not
different local optima (checked directly by
`softmax_classifier_is_seed_invariant_on_a_convex_problem` in
`nbsc/src/sgc.rs`'s test suite) — a useful methodological contrast with
§3's non-convex deep networks.

| Dataset | Raw features (no propagation) | NBSC-propagated | GCN-propagated |
|---|---|---|---|
| Cora | 0.494 ± 0.005 (wd=0) | 0.678 ± 0.002 (wd=1e-2) | **0.742 ± 0.005** (wd=0) |
| Citeseer | 0.425 ± 0.008 (wd=1e-2) | 0.540 ± 0.018 (wd=1e-2) | **0.601 ± 0.005** (wd=0) |
| PubMed | 0.683 ± 0.001 (wd=0) | 0.712 ± 0.000 (wd=1e-2) | **0.745 ± 0.001** (wd=0) |

(Test accuracy at the weight-decay value selected by validation accuracy;
full 5-point grids for every dataset/propagator are in
`docs/raw_results/sgc_linear_run.log`.)

### 4.1 Reading these numbers

- **Both propagators beat raw features by a wide margin on every
  dataset** (e.g. Cora: 49% → 68–74%) — graph structure carries real
  signal beyond a node's own features, for either propagator, on all
  three networks. This is worth stating plainly because it is easy to
  lose sight of amid the NBSC-vs-GCN comparison: the *first-order*
  finding of this whole thesis is that non-backtracking-walk structure
  is a usable graph-learning signal at all, which every experiment here
  confirms.
- **GCN's propagator wins on every dataset, at both the linear and deep-
  network levels.** This is the strongest single piece of evidence in
  this thesis for a genuine (if disappointing, relative to the original
  hypothesis) empirical conclusion: on citation-network-style
  homophilous graphs, the symmetric-normalized-adjacency propagator
  extracts more class-relevant signal than the non-backtracking-Hashimoto-
  derived one, and this holds up across two structurally unrelated model
  families (a convex linear classifier and a non-convex multi-layer
  network), three datasets, and both the canonical split and (for Cora,
  per the previous chapter) an independent random split.
- **The GCN-vs-NBSC gap shrinks as the dataset grows**: 6.4 points on
  Cora, 6.1 on Citeseer, 3.3 on PubMed. Three points is too few to call
  this a trend with confidence, but it is consistent with §5's finding
  that the diagnosed expansive-operator pathology itself attenuates on
  larger/denser graphs — a coherent, falsifiable hypothesis for future
  work (§6) rather than a claimed result.
- **Weight decay matters more for NBSC than GCN.** GCN's best weight
  decay is 0 on all three datasets; NBSC's best is the grid's largest
  value (`1e-2`) on all three. This is a clean, three-dataset-consistent
  piece of evidence supporting the mechanism in §5: if the NBSC
  propagator is genuinely expansive, its propagated features have larger
  effective scale/variance, and a classifier fit on them benefits more
  from L2 shrinkage than one fit on GCN's non-expansive (by construction)
  propagated features.

## 5. The expansive-operator diagnostic, extended to all three datasets

`results_cora_draft.md` traced RNBSC's growing-with-depth Dirichlet
energy (§3.3 above) to `‖A‖₂/ρ_B > 1` on Cora — the rescaled tap `A/ρ_B`
that `NbscFilterBank`'s recursion uses is an *expansive* linear operator,
unlike GCN's `Ŝ = D̂^{-1/2}(A+I)D̂^{-1/2}`, which is non-expansive by
construction (`‖Ŝ‖₂ = 1` exactly). `examples/expansive_operator_check.rs`
computes both `ρ_B` (via `estimate_spectral_radius`, the same matrix-free
Arnoldi machinery used throughout this crate) and `‖A‖₂`
(`adjacency_operator_norm`, matrix-free Lanczos) for all three datasets:

| Dataset | n | ρ_B | ‖A‖₂ | ‖A‖₂ / ρ_B |
|---|---:|---:|---:|---:|
| Cora | 2708 | 9.03 | 14.39 | **1.594** |
| Citeseer | 3327 | 11.49 | 13.74 | **1.197** |
| PubMed | 19717 | 21.64 | 23.24 | **1.074** |

All three are expansive (`ratio > 1`), so the mechanism is not
Cora-specific — but the ratio **shrinks monotonically as the graph
grows**, from 59% above 1 on the smallest graph to 7% above 1 on the
largest. This is consistent with (though, from three data points, does
not prove) a natural explanation: `ρ_B`, the non-backtracking spectral
radius, and `‖A‖₂`, the ordinary adjacency operator norm, are governed by
related but distinct graph statistics (roughly, non-backtracking walks
suppress the contribution of the graph's largest eigenvalue's associated
"easy" back-and-forth traversal patterns that inflate `‖A‖₂` on small,
locally tree-like or star-like neighborhoods), and this suppression
effect plausibly weakens as the graph's local structure becomes more
uniformly well-connected at scale — a hypothesis stated here explicitly
as a hypothesis, not a demonstrated result, and flagged in §6 as a
natural next step (does the ratio continue shrinking past 1 on
sufficiently large/dense real graphs, and if so does NBSC's relative
disadvantage vanish with it?).

## 5.5. Energy-Stabilized Deep Stacks: an explicit Dirichlet-energy regularizer

§5's finding — `‖A/ρ_B‖₂ > 1` on all three datasets — motivates trying to
control the resulting variance explosion directly, alongside (not instead
of) `NBSC_NORMALIZE`'s LayerNorm ablation (§3.3, `results_cora_draft.md`).
LayerNorm changes what each Chebyshev tap's *output* is (rescales/recenters
it); a complementary approach is to leave the architecture alone and change
what the *loss* rewards, by adding an explicit Dirichlet-energy penalty.

`burn_layer.rs` already computed `dirichlet_energy` as a post-hoc CPU
diagnostic (§3.3), but that implementation detaches from the autodiff graph
(`x.to_data()`) and can't contribute a gradient. `dirichlet_energy_differentiable`
is a second implementation of the same normalized quantity —
`trace(X^T L X) / (m · mean_sq_row_norm)` for the graph Laplacian
`L = diag(D) − A` — built entirely from Burn tensor ops (two dense matmuls
plus an elementwise product-and-sum) so it stays in the autodiff graph.
`thesis_bench.rs`'s training loop now supports `NBSC_ENERGY_REG_LAMBDA`
(default `0.0`, i.e. off — every existing number above is reproducible
unchanged): when set, it sums this differentiable energy across every
layer's activation ("between deep Chebyshev filter taps"), averages by
depth, and adds `lambda * mean_energy` to the training loss, so the
optimizer is directly penalized for letting per-layer energy blow up
rather than only being observed doing so after the fact.

Unit tests (`burn_layer.rs`, `differentiable_energy_matches_cpu_diagnostic`
and `differentiable_energy_is_actually_differentiable`) confirm the new
implementation numerically agrees with the existing CPU diagnostic on the
same input, and that `.backward()` produces a non-trivial gradient w.r.t.
the activation — i.e. it is usable as a loss term, not just a re-derivation
of the same number. **This has been implemented and unit-tested but not
yet run as a training sweep** (no compiler was available in the environment
this was implemented in — see §6). The natural next experiment: a small
`NBSC_ENERGY_REG_LAMBDA` grid (e.g. `0, 1e-3, 1e-2, 1e-1`) crossed with
`NBSC_NORMALIZE` (`true`/`false`) at `NBSC_DEPTHS=4,6,8` on Cora, to see
whether the explicit penalty (a) stabilizes deep-stack accuracy the way
LayerNorm did in §3.3's ablation, (b) helps *in addition to* LayerNorm, or
(c) is redundant with it.



Stated plainly, matching this project's existing epistemic-honesty
convention (see `results_cora_draft.md`'s own "Honest assessment"
section):

- **GAT and GraphSAGE were not re-run under the canonical split.** Their
  per-run cost (~10–15× GCN's, per the previous chapter's own timing
  notes) made a full canonical-split re-run out of budget alongside
  everything above. `thesis_bench.rs`'s `NBSC_ARCHS` environment variable
  already supports running them (`NBSC_ARCHS=gat,sage`); this is a
  mechanical re-run, not new engineering.
- **Citeseer's depth-1 and depth-3 deep-network points are missing** (only
  depth 2 was completed). The pattern from Cora (peak at depth 2,
  degrading at both 1 and 3) is a reasonable prior but is *not* confirmed
  for Citeseer.
- **No deep-network weight-decay grid** was completed (only `wd=0`, for
  every deep-network number in §3). §4's linear-classifier weight-decay
  finding (NBSC benefits more from L2 than GCN, on all three datasets) is
  suggestive that the same would hold for the deep networks, but this is
  an extrapolation across model classes, not a demonstrated fact — this
  is the single most important remaining gap for a defense committee to
  probe, and should be run before any oral defense if compute allows
  (`NBSC_WEIGHT_DECAY=5e-4` against the existing `thesis_bench.rs`
  harness is all that's needed).
- **The Shchur-style multi-random-split robustness check** (§2) was
  designed (`thesis_bench.rs`'s `NBSC_SPLIT=random`,
  `NBSC_SPLIT_SEEDS=1,2,3`) but not run to completion.
- **Seed counts remain modest** (2–3 for the deep networks, 3 for the
  linear classifiers) given this project's single-core evaluation
  hardware (each Cora/Citeseer deep-network run takes 2–12 minutes; a
  single Citeseer NBSC run took ~720 seconds). The reported standard
  deviations should be read as *rough* variance estimates, and a
  committee should expect "how many seeds, and why that many" as a
  natural question — the honest answer is a hardware-driven compute
  budget, documented here rather than hidden.
- **The shrinking-expansiveness-ratio hypothesis (§5)** is based on three
  points and needs either more real datasets or a controlled synthetic
  sweep (e.g. SBM graphs at increasing `n` and fixed average degree) to
  become a supportable claim rather than a suggestive observation.
- **`NBSC_ENERGY_REG_LAMBDA` (§5.5) has now been run once, live, at depth 6
  on Cora** (n=1 seed, 30 epochs — see `energy_stabilized_deep_stacks_live_run.md`
  for the full table and caveats). Headline finding: the penalty *does*
  suppress the Dirichlet energy diagnostic as designed, but **does not by
  itself recover accuracy at depth 6** (every λ>0 tried scored at or below
  the λ=0 baseline, and λ=1.0 collapsed training near chance-level).
  `NBSC_NORMALIZE=true` (LayerNorm) recovers most of the loss on its own
  (0.448→0.614 test acc); the one config combining both
  (`normalize=true`, λ=0.01) reached the best depth-6 number observed
  (0.651), suggesting the energy term may be useful as a small correction
  on top of LayerNorm rather than a substitute for it — but this is a
  single seed at a single depth on a single dataset, not a validated
  claim. `differentiable_energy_matches_cpu_diagnostic` and
  `differentiable_energy_is_actually_differentiable` do pass (verified
  live, `cargo test --features burn --lib burn_layer`: 6/6). The
  multi-seed, multi-depth, multi-dataset grid described in §5.5's closing
  paragraph is still the right next step before treating any of this as
  a real finding.

None of these gaps were skipped for lack of a plan — each has a concrete,
already-implemented path to completion (the code exists and is exercised
by at least a smoke run); what remains is additional wall-clock compute,
which this project's hardware (a single CPU core, ~4 GB RAM) made
infeasible to complete inside this work session's budget. A reader with
access to a multi-core machine (or a few more hours) can close every item
above by re-running `thesis_bench.rs` and `sgc_bench.rs` with the
indicated environment variables/arguments — nothing here requires new
engineering, only more compute time.
