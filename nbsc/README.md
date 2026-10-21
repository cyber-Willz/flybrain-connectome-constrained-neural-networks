# NBSC — Non-Backtracking Spectral Convolution

A graph convolution built from the poles of the **Ihara zeta function**
(equivalently, the eigenvalues of the **Hashimoto / non-backtracking
matrix** `B`) instead of the normalized graph Laplacian. This crate
implements the derivation in `ihara_zeta.rs` end to end: math → Rust →
a Burn-based learnable layer → a benchmark against a standard GCN.

## Why non-backtracking?

The normalized Laplacian used by GCN/ChebNet is symmetric and real — it
sees a graph's *local density* structure well, but is blind to *oriented
cycle* structure (directionality, girth, non-bipartiteness). The Hashimoto
matrix `B` is built on directed edges with a no-immediate-reversal rule, is
non-symmetric, and its spectrum genuinely encodes that oriented-cycle
information (complex eigenvalue pairs correspond to non-real poles of
`ζ_G`). NBSC uses a Chebyshev-style polynomial filter bank built from `B`'s
spectrum instead of the Laplacian's.

## Layout

```
krylov_ds/            your existing Arnoldi/Lanczos Krylov-subspace crate (unmodified, used as-is)
spectral_hypergraph/  vendored spectral-hypergraph crate (unmodified core; see hypergraph feature below)
nbsc/
  src/graph.rs         graph struct + synthetic generators (SBM, tree, near-regular expander)
  src/spectral.rs       §2-6: Bass reduction, matrix-free Hashimoto linearization,
                         rho_B via krylov_ds Arnoldi, sparse T_k feature recursion
  src/gcn.rs             baseline GCN propagation (same sparse style, for fair comparison)
  src/burn_layer.rs      §7-8: learnable NbscLayer / GcnLayer as burn::Module, + Dirichlet energy
  src/hypergraph_bridge.rs  (feature `hypergraph`) clique expansion + krylov_ds/spectral_hypergraph
                             LinearOperator adapter; see "Hypergraph integration" below
  examples/benchmark.rs  trains both at depths 1/2/4/8/16 on SBM + tree, reports acc + energy
  examples/hypergraph_bridge_demo.rs  (feature `hypergraph`) both bridge paths + cross-checks
  tests/                 9 spectral tests (Bass identity, dense-vs-Krylov cross-checks) +
                          4 burn-layer tests + 6 hypergraph_bridge tests, all currently passing
```

## Toolchain note

This environment has **rustc 1.75** available (matching the pin you've used
before for `causal_llm`). Burn's current release line needs 1.81+, so this
uses **Burn 0.13.2**, with a handful of transitive dependencies pinned back
to 1.75-compatible versions in `Cargo.lock` (`half`, `uuid`, `bincode`,
`rmp-serde`, `rmp`, `indexmap`, `rayon`/`rayon-core`). If you build this on
a newer rustc, you can safely `cargo update` to let those float back up and
probably move to Burn 0.15+ as well — I'd suggest revisiting the pins in
that case rather than assuming they're still needed.

The core `spectral`/`graph`/`gcn` modules have **no Burn dependency at
all** (`cargo build --no-default-features`) and only need `nalgebra`,
`rand`, `krylov_ds` — so the Ihara-zeta/Hashimoto machinery itself is
usable independent of Burn or of MSRV constraints.

## Running it

```bash
# pure-CPU spectral tests (no Burn needed)
cargo test -p nbsc --no-default-features

# full test suite, including the Burn layer
cargo test -p nbsc --features burn

# the benchmark (depth sweep, SBM vs. tree, NBSC vs. GCN)
cargo run --release --example benchmark --features burn

# hypergraph bridge tests (no Burn needed)
cargo test -p nbsc --features hypergraph hypergraph_bridge

# hypergraph bridge demo (clique expansion + matrix-free operator adapter,
# cross-checked against spectral_hypergraph's own spectral_cluster)
cargo run --release --example hypergraph_bridge_demo --features hypergraph
```

## What the benchmark actually shows

This is a **synthetic sanity-check benchmark**, not a citation-grade
replication of Planetoid/OGB results — there's no bundled real-world
labeled graph dataset here, and each configuration is a single run from a
fixed seed rather than an average over many seeds. Treat the numbers as
"does the theory's qualitative prediction show up," not as
publication-ready accuracy figures.

A representative run (`cargo run --release --example benchmark --features
burn`, seed 7):

- **Stochastic Block Model** (4 communities, 160 nodes, real triangle
  structure — the near-Ramanujan-expander regime §6 targets): NBSC and GCN
  are both trainable and reach comparable accuracy at moderate depth, but
  at depth 16 NBSC's Dirichlet energy (0.248) stays roughly 5-6x higher
  than GCN's (0.043) — GCN's representations have visibly collapsed toward
  a constant per connected component (classic over-smoothing) while NBSC's
  haven't, exactly the effect §6-8 predicts non-backtracking filters should
  resist. (Both models' *accuracy* also degrades by depth 16, which looks
  like a separate optimization-difficulty effect — no residual connections
  or normalization in either stack — rather than something the energy
  metric alone explains; worth digging into if you want to push this
  further.)
- **Random tree** (bipartite, no cycles — the negative control): the
  Hashimoto spectrum is numerically nilpotent here (`rho_B ≈ 0`, verified
  directly in `spectral::tests::tree_has_zero_nontrivial_hashimoto_spectrum`),
  so NBSC has no oriented-cycle structure to exploit. Its energy-retention
  edge over GCN, large at shallow depth, collapses by depth 16 (down to
  ~0.0) — consistent with the theory's own prediction that this
  construction's benefit is cycle-structure-dependent, not universal.

That second result is the more interesting one, honestly: a method whose
benefits *disappear on exactly the case the theory predicts they should*
is much more convincing than one that "wins everywhere," which usually
means the benchmark isn't discriminating.

## Hypergraph integration (`spectral_hypergraph` bridge) — status update

**Added and compiled/tested.** `nbsc` now optionally depends on
`spectral_hypergraph` (vendored into this workspace as a third member,
`spectral_hypergraph/`) behind a new `hypergraph` feature, wired up in
`nbsc/src/hypergraph_bridge.rs`. The two crates had independently grown
incompatible `LinearOperator` traits (this crate's `krylov_ds` version is
generic/slice-based/allocation-free; `spectral_hypergraph`'s is
`f64`-only/`DVector`-based); the bridge module is the translation seam
between them, not a rewrite of either.

Two integration paths, both exercised by tests and by the new example:

1. **Clique expansion → `Graph`** (`hypergraph_bridge::clique_expand`):
   flattens a `SpectralHypergraph` into this crate's plain unweighted
   `Graph`, so the entire existing pipeline — NBSC filter bank, GCN/GAT/
   GraphSAGE baselines, `rho_B`, the Burn layers — runs on hypergraph data
   completely unmodified. `hypergraph_bridge::nbsc_filter_bank_from_hypergraph`
   is a one-call convenience for "clique-expand, then build the filter
   bank."
2. **Matrix-free operator adapter** (`hypergraph_bridge::HypergraphLaplacianOperator`):
   wraps `spectral_hypergraph`'s matrix-free normalized hypergraph
   Laplacian operator as a `krylov_ds::LinearOperator<f64>`, so this
   crate's own Arnoldi/Lanczos engine (the one `rho_B` is built on) runs
   directly against the *true* hypergraph structure — no clique expansion,
   no `n x n` matrix ever formed.
   `hypergraph_bridge::hypergraph_laplacian_operator_norm` and
   `hypergraph_bridge::hypergraph_algebraic_connectivity` are built on top
   of it, mirroring `spectral::adjacency_operator_norm`'s existing pattern
   (deterministic seeded start vector, full-reorthogonalization Lanczos,
   `krylov_ds::eig::lanczos_ritz_pairs`) but pointed at the hypergraph
   Laplacian instead of the plain adjacency operator.

Also added: `hypergraph_bridge::hypergraph_stochastic_block_model`, a
community-structured synthetic hypergraph generator (the hyperedge-level
analogue of `graph::stochastic_block_model`) with a coverage pass
guaranteeing no isolated vertices regardless of density parameters.

**Verified, not just written blind**, on `rustc 1.75.0` (installed via
`apt-get install cargo rustc`, matching the pin this workspace already
uses): `cargo build --workspace` (with and without `--features
nbsc/hypergraph`), `cargo test --workspace --features nbsc/hypergraph` — 69
tests total across all three crates, all passing, including 6 new
`hypergraph_bridge` tests (clique-expansion nonzero-pattern parity against
`spectral_hypergraph::laplacian::clique_expansion_adjacency`, operator
adapter parity against the wrapped `HypergraphOperator`, both new
diagnostics cross-checked against dense ground truth via
`nalgebra::SymmetricEigen`, generator shape/connectivity, and an
end-to-end filter-bank-from-hypergraph smoke test) — plus doctests, plus
`cargo build --examples --features "burn,hypergraph"` to confirm the new
example coexists cleanly with the existing Burn-based ones.

New example: `cargo run --release --example hypergraph_bridge_demo
--features hypergraph` builds a 4-community, 100-vertex synthetic
hypergraph, runs both integration paths, and cross-checks community
recovery between `spectral_hypergraph::spectral_cluster` (native
hypergraph Laplacian) and a from-scratch spectral clustering pass on the
clique-expanded graph's own Laplacian. A representative run (seed 11):
`rho_B = 8.220`, plain-adjacency operator norm `9.338` (so `||A||_2 /
rho_B = 1.136` — the clique-expanded graph's `A / rho_B` tap is measurably
expansive here too, same diagnostic `spectral.rs` already flags on Cora),
hypergraph Laplacian operator norm `0.988`, algebraic connectivity
`0.0053`; native hypergraph spectral clustering purity `1.000` vs.
clique-expansion spectral clustering purity `0.750` on the same ground
truth — a concrete, reproducible illustration of the higher-order
structure the clique expansion collapses away.

**Design choice, stated plainly:** the bridge is additive and optional
(`hypergraph` feature, off by default) rather than a restructuring of
either crate's core types. `Graph` stays unweighted and hypergraph-unaware;
`SpectralHypergraph` stays free of any `krylov_ds`/NBSC dependency. Neither
crate's existing public API changed.

## Citeseer (second real dataset) + weight decay — status update

**Citeseer added and validated.** `nbsc/data/citeseer/{citeseer.content,citeseer.cites}`
— sourced from the `data/citeseer/` folder committed directly (plain text,
not a pickle) in `ialireza13/expanded_gcn` on GitHub, itself derived from
the original LINQS/Sen et al. 2008 release. Verified against published
statistics: 3327 papers (3312 with real bag-of-words features, 15
zero-padded — a documented property of this release, not a parsing bug),
3703-dimensional binary features, 6 classes, per-class counts matching
{264, 508, 590, 596, 668, 701}. `Dataset::load_citeseer_default` reuses the
exact same parser as Cora (`load_planetoid_style` was already
dataset-agnostic) plus the same stratified-split logic.
`nbsc/examples/benchmark_citeseer.rs` mirrors `benchmark_cora.rs` for the
full NBSC/GCN/GAT/GraphSAGE sweep.

**Weight decay added**, addressing the ~70% vs. literature's ~81.5% gap
flagged in `docs/results_cora_draft.md`. `benchmark_cora.rs` defaults
`WEIGHT_DECAY = 0.0` (preserves exact reproducibility of the already-locked
baseline table — rerunning that file unmodified should still match); change
to `5e-4` (the original GCN paper's value) to run the regularization
experiment as a new, separate result. `benchmark_citeseer.rs` defaults to
`5e-4` from the start, since there's no prior Citeseer baseline to protect.

*(This section describes `benchmark_citeseer.rs`'s original, non-canonical
`stratified_split`-based Citeseer baseline. It has since been superseded as
the thesis's primary result by the canonical-split, three-dataset study in
"Canonical-split, three-dataset results" below — kept here for its own
sake since `benchmark_citeseer.rs` still exists and is still runnable
unmodified.)*

## Canonical-split, three-dataset results — thesis-primary — status update

**All three gaps flagged at the end of the previous section (single
dataset, non-canonical split, untuned regularization) are now closed.**
Full write-up: **`docs/results_thesis.md`** — start there, not
`results_cora_draft.md`, for the thesis's actual headline numbers.
Summary:

- **Canonical split, not an approximation of it.** Cora, Citeseer, and
  PubMed are now loaded via `Dataset::load_cora_planetoid` /
  `load_citeseer_planetoid` / `load_pubmed_planetoid`
  (`nbsc/src/dataset.rs`), which parse the *bit-identical* published
  Yang/Cohen/Salakhutdinov 2016 Planetoid split — obtained by unpickling
  `tkipf/gcn`'s reference data files once, offline, and converting to
  plain text (full provenance note in the `load_planetoid_canonical` doc
  comment). Verified against published split sizes (140/500/1000,
  120/500/1000, 60/500/1000) by dedicated tests.
- **Three real datasets, not one.** PubMed (19717 nodes, 3 classes,
  500-dim *continuous* TF-IDF features) is new. Its dense `n×n` Burn
  tensors don't fit in ~4 GB RAM, so `nbsc/src/sgc.rs` adds a second,
  independent evaluation path — a matrix-free, `O(n·f)`-memory,
  SGC-style (Wu et al. 2019) "propagate once, then fit a linear
  classifier" pipeline — run on all three datasets, not just PubMed, as
  a same-methodology cross-check alongside the deep networks.
- **Regularization tuned, not defaulted.** Every linear-classifier
  headline number is chosen from a 5-point weight-decay grid by
  validation accuracy (`nbsc/examples/sgc_bench.rs`).
- **Headline finding, replicated across two structurally unrelated model
  classes (deep, non-convex Burn networks and a convex softmax
  classifier) and three real datasets**: GCN's symmetric-normalized-
  adjacency propagator beats NBSC's non-backtracking-Hashimoto-derived
  one, every time. The gap narrows as the dataset grows (Cora > Citeseer
  > PubMed), which `docs/results_thesis.md` §5 connects to a new,
  three-dataset extension of the `‖A‖₂/ρ_B` expansive-operator
  diagnostic (ratio 1.594 → 1.197 → 1.074 as `n` grows from 2708 to
  19717) — stated there explicitly as a suggestive three-point
  observation, not a proven trend.
- **What's still open** (GAT/GraphSAGE canonical-split re-run, Citeseer
  depth 1/3, a deep-network weight-decay grid, the Shchur-style
  multi-split robustness check, more seeds): listed with reasons and the
  exact commands to close each gap in `docs/results_thesis.md` §6. Every
  gap has working code behind it already; what's missing is additional
  wall-clock compute on faster/more-parallel hardware than this
  project's single-core, ~4 GB evaluation machine.

## Real-data + baseline work in progress (Cora, GAT, GraphSAGE) — original plan



**`benchmark_cora` has now been compiled and run successfully** (confirmed
by the person running it, not just written blind — see the compiler output
and results below). The Cora loader, GAT layer, GraphSAGE layer, and the
training harness all built and ran correctly on the first real attempt, on
`burn v0.13.2`. `n=2708, m=5278, train=140, val=500, test=1000` — all match
what the module docs predicted.

**Full depth 1-3 results obtained** (5 seeds each, 60 training runs total).
Headline: RNBSC does not beat GCN or GAT at any tested depth on Cora; all
four architectures degrade past depth 2; RNBSC's Dirichlet energy *grows*
with depth (the opposite of the other three architectures' over-smoothing
pattern) with sharply increasing cross-seed variance at depth 3. Full
numbers, analysis, and the likely mechanism (a hypothesis that `A / rho_B`
may be an expansive operator, unlike GCN's provably non-expansive
propagator) are written up in `docs/results_cora_draft.md`.

**The full diagnostic chain has now been run, end to end, with real results
at every step:**
1. Baseline depth 1-3 results showed RNBSC's Dirichlet energy growing with
   depth (opposite of GCN/GAT/GraphSAGE's normal over-smoothing pattern),
   with cross-seed variance exploding at depth 3 (&plusmn;2.8, vs. &plusmn;0.15 or
   tighter for the other three architectures).
2. `operator_norm_check` confirmed the suspected mechanism directly:
   `||A||_2 / rho_B = 1.594` on Cora — the `A / rho_B` tap is measurably
   expansive (operator norm ~59% above 1), unlike GCN's provably
   non-expansive propagator.
3. The `NBSC_NORMALIZE` (LayerNorm) ablation confirmed the fix: depth-3
   energy dropped from `10.4997 &plusmn; 2.8046` to `0.3180 &plusmn; 0.0423` — a
   >33x reduction in mean, >65x reduction in variance. Accuracy improved
   only modestly and not clearly beyond noise (0.672&rarr;0.687 test,
   0.652&rarr;0.681 val) — normalization fixes the *instability*, not the
   *competitiveness gap* against GCN/GAT. That distinction, not "fixed" or
   "didn't help", is the accurate conclusion.

Full numbers, tables, and analysis are in `docs/results_cora_draft.md`,
which is now essentially complete as a results-chapter draft (no
placeholders remain).

**Still not done (as of the original Cora-only chapter; see "Canonical-split,
three-dataset results" above for what has since closed most of this):**
Citeseer/PubMed as second/third real datasets, weight
decay/dropout (a likely explanation for the ~70% vs. literature's ~81.5%
gap, separate from the depth/normalization questions above), the sparse CSR
kernel, and the full thesis document outside of the results section draft.

## Real-data + baseline work in progress (Cora, GAT, GraphSAGE) — original plan

This section documents an in-progress push on the two biggest gaps
flagged below: real labeled data and additional baselines.

**What's added:**
- `nbsc/data/cora/{cora.content,cora.cites}` — the real Cora citation
  network (2708 papers, 5429 citation links, 1433-dim binary bag-of-words
  features, 7 classes), the plain-text release used by `tkipf/pygcn`.
  Verified against published statistics.
- `nbsc/src/dataset.rs` — parser, `Dataset` type (graph + features + labels
  + train/val/test masks), and a documented stratified split. **The split
  is not the literature's exact "Planetoid" split** (that ships as Python
  pickles, not parsed here) — it has the same shape (20 labeled/class
  train, 500 val, 1000 test) but is a different, independently-seeded
  sample. See the module docs in `dataset.rs` for the full rationale.
  **Do not present accuracy numbers from this split next to published
  Cora leaderboard entries as if directly comparable.**
- `nbsc/src/gat_layer.rs` — GAT baseline (Velickovic et al. 2018).
- `nbsc/src/sage_layer.rs` — GraphSAGE baseline, mean aggregator
  (Hamilton, Ying & Leskovec 2017), full-batch (no neighbor sampling).
- `nbsc/examples/benchmark_cora.rs` — multi-seed (default 5) harness
  training NBSC/GCN/GAT/GraphSAGE on Cora at several depths, reporting
  mean +/- std for val/test accuracy and final Dirichlet energy.

**Status: written but not compiled or run.** This was developed in an
environment with no Rust toolchain available (network access to
`static.rust-lang.org` was blocked, so `rustup`/`cargo` could not be
installed), so none of the four new/changed files above have been through
`cargo build` or `cargo test` yet. They were written closely against the
patterns already proven to compile elsewhere in this crate (`burn_layer.rs`,
`benchmark.rs`), but that is not a substitute for actually building them.

**First thing to do, before anything else:**
```bash
cargo test -p nbsc --no-default-features   # dataset.rs has no burn dependency;
                                             # this alone validates the Cora parser
                                             # and stratified split in isolation.
cargo test -p nbsc --features burn          # brings in gat_layer.rs / sage_layer.rs tests
cargo run --release --example benchmark_cora --features burn
```

**Specific spots most likely to need a small fix on first compile** (flagged
honestly rather than presented as done):
- `gat_layer.rs`: the attention-score broadcast (`[n,1] + [1,n] -> [n,n]`)
  and the `.sum_dim(1)` keepdim assumption are the two lines with the least
  certainty behind them — everything else in these files reuses API calls
  already proven to compile elsewhere in this crate.
- `dataset.rs`: `Graph::m()` after loading Cora is asserted to be `> 0` and
  `<= 5429` rather than an exact hardcoded number, deliberately — some
  `.cites` edges reference paper IDs absent from `.content` and get
  skipped; run it once and see what the real number is before tightening
  that test.
- `benchmark_cora.rs`: not yet timed. `N_SEEDS x depths x 4 architectures`
  full training runs on a 2708-node graph is meaningfully more compute
  than the synthetic benchmark; consider `N_SEEDS = 2` and `depths = [2]`
  for a first smoke test.

**Still not done** (the remaining items from "What would turn this into a
real paper" below): Citeseer/PubMed as second/third real datasets, the
sparse CSR kernel, and the written thesis document itself.

## What would turn this into a real paper

The proposal's "Expected Academic Contribution" (over-smoothing resistance
+ better classification on sparse/noisy real graphs) would need, at
minimum: real labeled benchmarks (Cora/Citeseer/PubMed for node
classification, plus a link-prediction and an anomaly-detection dataset
per the original ask), multiple seeds with confidence intervals, GAT and
GraphSAGE baselines (only GCN is implemented here), and — per §8's own
flagged TODO — a sparse (CSR) `T_k` kernel with a hand-written backward
in place of the current dense `n x n` Burn tensors, since dense adjacency
tensors won't scale past a few thousand nodes. The sparse *forward* pass
for that already exists and is tested (`spectral::NbscFilterBank`); what's
missing is the differentiable (Burn-autodiff-compatible) sparse version.


## flybrain (fruit-fly connectome networks)

See `flybrain/README.md` and `docs/flybrain_results.md` — a connectome-constrained network built on the crates above.
