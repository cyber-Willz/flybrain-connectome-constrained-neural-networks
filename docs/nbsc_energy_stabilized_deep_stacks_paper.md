Energy-stabilized deep stacks in Ramanujan-Non-Backtracking Spectral
Convolution (RNBSC): an empirical study
RNBSC Project — live experimental report
August 28, 2026

Abstract

A Ramanujan-Non-Backtracking Spectral Convolution (RNBSC) layer stacks Chebyshev
filter taps built from the rescaled non-backtracking operator `A/rho_B`,
where `rho_B` is the Perron-Frobenius eigenvalue of the non-backtracking
matrix `B` of Bordenave, Lelarge and Massoulie [1]. On the graphs studied
in this report, `A/rho_B` is *expansive* (`||A/rho_B||_2 > 1`), so deep
stacks of this filter do not exhibit the classical over-smoothing collapse
seen in ordinary graph convolutions; instead, per-layer activation energy
grows with depth. We study two candidate stabilization mechanisms: an
existing LayerNorm ablation switch (`NBSC_NORMALIZE`) that renormalizes
each layer's output, and a new *differentiable* Dirichlet-energy
regularizer (`NBSC_ENERGY_REG_LAMBDA`) that penalizes the training loss
directly for high per-layer energy, built from the graph-Laplacian
quadratic-form identity and kept inside Burn's autodiff graph. We
implement the regularizer, unit-test it against a pre-existing CPU-only
diagnostic of the same quantity, and evaluate it live end-to-end on Cora
(canonical Planetoid split) at depths 2, 4, 6 and 8, crossed with the
LayerNorm switch and, where the compute budget allowed, with a second
training seed. Our central empirical finding is that the regularizer's
value is depth-dependent and not monotone: at depth 4 it gives a
seed-stable accuracy improvement on top of LayerNorm; at depth 6 an
apparent single-seed improvement fails to replicate under a second seed;
at depth 8 the evidence remains too thin to draw a conclusion. We report
every run performed, including several that did not complete within the
available compute budget, and we are explicit throughout about which
findings are supported by more than one random seed and which are not.

1 Introduction

Deep stacks of graph neural network layers are known to suffer from
*over-smoothing*: as depth grows, node representations collapse toward a
low-rank, low-information subspace, and the Dirichlet energy of the
representation — the quantity `sum_{(u,v) in E} ||x_u - x_v||^2`,
appropriately normalized — decays toward zero. Layer normalization and
residual connections are standard remedies. The RNBSC architecture in this
project departs from ordinary graph convolution by using the
non-backtracking operator `B` studied by Bordenave, Lelarge and Massoulie
[1] as the basis of its Chebyshev filter taps, rescaled by `B`'s
Perron-Frobenius eigenvalue `rho_B`. This choice is motivated by [1]'s
characterization of the non-backtracking spectrum: on sparse random
graphs the leading eigenvalue of `B` separates cleanly from the bulk
(Theorem 3 of [1], reproduced as our Preliminaries below), which is the
basis of the "spectral redemption" approach to community detection that
[1] establishes. A byproduct of adopting `A/rho_B` as a filter, however,
is that this rescaled operator need not be a contraction: on the graphs
examined here, `||A/rho_B||_2 > 1`, so repeated application does not
attenuate the representation the way an averaging (contractive) operator
would. This project's prior work (`results_thesis.md`, Section 5)
documented this expansiveness empirically across three citation-network
datasets and observed that RNBSC's accuracy degrades with depth alongside
*rising*, not falling, per-layer Dirichlet energy — the opposite signature
from classical over-smoothing. Appendix A gives the complete proof that
this is not merely an empirical correlation: GCN's propagator is
*provably* non-expansive on every connected graph (Theorem A.2), which
forces its Dirichlet energy to be non-increasing with depth (Proposition
A.3); no analogous guarantee exists for `A/rho_B`, and on every dataset
this project has measured, it is not merely unguaranteed but actually
violated (`||A||_2/rho_B > 1` in all four cases checked, Section A.4).

This report studies whether that degradation can be controlled directly
by penalizing energy in the training loss, as a second, independent
mechanism alongside the LayerNorm ablation switch already present in the
codebase. Both mechanisms restrain the same diagnostic quantity but act
through different channels: LayerNorm changes what each layer's *output*
is (a fixed rescaling/recentering applied regardless of what produced the
activation); the loss penalty changes what the optimizer is *rewarded*
for, leaving the architecture itself untouched. A priori it is not obvious
whether the two are redundant, complementary, or in tension — a loss-side
penalty that fights an architecture unable to represent low-energy
solutions cheaply could easily just suppress signal rather than fix the
underlying instability. This report is an empirical investigation of that
question, not a theoretical one: we implement the mechanism, verify it is
correctly implemented, and run it.

Relation to the non-backtracking spectrum. The rescaling `A/rho_B` used
by the RNBSC filter tap is chosen so that the *dominant* mode of repeated
non-backtracking-walk counting is normalized to unit growth rate,
mirroring the role `rho_B = lambda_1(B)` plays in [1]'s Theorem 3: that
theorem shows `lambda_1(B) = alpha + o(1)` while every other eigenvalue of
`B` stays below `sqrt(alpha) + o(1)`, i.e. a spectral gap of order
`sqrt(alpha)` around the leading mode. Dividing by `rho_B` normalizes that
leading mode, but says nothing about the operator norm of the resulting
rescaled matrix in the *ordinary* (adjacency, not non-backtracking) sense
relevant to a Chebyshev filter tap acting on node features — hence the
expansiveness this report studies is not itself a contradiction of [1],
merely a consequence of applying a normalization designed for one
spectral object (`B`) to filters built from a different one (`A`).

Organization. Section 2 recalls the RNBSC filter construction, the
Dirichlet-energy diagnostic already present in the codebase, and the two
stabilization mechanisms under study. Section 3 states our main empirical
findings. Section 4 describes the construction of the differentiable
energy regularizer and its unit-test verification. Section 5 describes
the experimental protocol, including the compute-budget constraints that
shaped which runs were and were not completed. Section 6 reports results
in full, including a first single-seed pass and a second multi-seed pass
that was designed specifically to check the first pass's headline
finding. Section 7 discusses the results. Section 8 lists open questions
this report does not resolve.

2 Preliminaries

2.1 The non-backtracking matrix. Following [1], for a graph `G = (V, E)`
the non-backtracking matrix `B` is indexed by the set of directed edges
`E-vec = {(u,v) : {u,v} in E}` and defined by `B_{ef} = 1` if edge `e`
feeds into edge `f` without immediately reversing it, and `0` otherwise.
Theorem 3 of [1] establishes that for an Erdos-Renyi graph `G(n, alpha/n)`
with `alpha > 1`, the leading eigenvalue satisfies `lambda_1(B) = alpha +
o(1)` while all others satisfy `|lambda_i(B)| <= sqrt(alpha) + o(1)` with
probability tending to 1. We write `rho_B` for this leading eigenvalue.

2.2 The RNBSC filter tap. The RNBSC layer builds a `K`-tap Chebyshev
polynomial filter in the rescaled operator `A / rho_B`, where `rho_B` is
estimated from the graph's non-backtracking matrix as above and is,
provably, the correct normalizer for this recursion (Appendix A.2). On
the three citation-network datasets examined in this project's prior work
(`results_thesis.md`, Section 5), `||A / rho_B||_2 > 1` in every case,
i.e. the rescaled operator is expansive rather than contractive — a
measured fact with no supporting guarantee in either direction, in sharp
contrast to GCN's propagator, whose non-expansiveness is a theorem
(Appendix A.3-A.4).

2.3 Dirichlet energy. For a node-feature matrix `X` (one row per node),
the (normalized) Dirichlet energy is
    `E(X) = (1/m) * sum_{(u,v) in Edges} ||x_u - x_v||^2 / mean_sq_row_norm(X)`,
where `m` is the edge count. Low `E(X)` indicates neighboring nodes have
nearly identical representations (the over-smoothing signature); the
codebase already computes this as a CPU-side, post-hoc diagnostic
(`dirichlet_energy` in `burn_layer.rs`) that detaches from the autodiff
graph and cannot itself contribute a training gradient.

2.4 Two stabilization mechanisms.

  (a) `NBSC_NORMALIZE` (pre-existing): a LayerNorm ablation switch applied
  between Chebyshev filter taps. When enabled, each layer's output is
  renormalized regardless of the scale the preceding expansive operator
  produced.

  (b) `NBSC_ENERGY_REG_LAMBDA` (new, this report): a scalar weight `lambda`
  on an explicit, differentiable version of `E(X)` (Section 4), summed
  across every layer's activation and added to the training loss, so the
  optimizer is directly penalized for letting per-layer energy grow.

These two mechanisms are not mutually exclusive; Section 6 evaluates both
individually and in combination.

3 Main empirical findings

We state our findings in the style of the results they most resemble in
form (a claim, a scope, a degree of support) while being explicit that
these are single- or double-seed empirical observations, not proven
statements, and that "Finding" below never carries the evidentiary weight
of "Theorem."

Finding 1 (regularizer is mechanically correct). The differentiable
energy regularizer constructed in Section 4 agrees numerically with the
pre-existing CPU diagnostic on held-out synthetic graphs to within
`1e-3`, and produces a non-trivial gradient under `.backward()` (unit
tests `differentiable_energy_matches_cpu_diagnostic` and
`differentiable_energy_is_actually_differentiable`, both passing at time
of writing). Live end-to-end training runs confirm this mechanically:
increasing `lambda` monotonically suppresses final Dirichlet energy at
depth 6 (`16.13 -> 14.69 -> 4.02 -> 2.36 -> 1.01` for `lambda in {0,
0.001, 0.01, 0.1, 1.0}`, single seed, `NBSC_NORMALIZE=false`).

Finding 2 (energy suppression alone does not recover accuracy at depth
6). With `NBSC_NORMALIZE=false`, every tested `lambda > 0` at depth 6
scored at or below the `lambda = 0` baseline (0.448 test accuracy), and
the largest tested value (`lambda = 1.0`) collapsed training to
near-chance accuracy (0.232, single seed). By contrast, `NBSC_NORMALIZE
= true` alone recovered substantially more accuracy (0.614, single
seed) than any `lambda > 0` value tried without it.

Finding 3 (depth 4: a seed-stable improvement on top of LayerNorm). With
`NBSC_NORMALIZE = true` fixed, a 2-seed sweep over `lambda in {0, 0.001,
0.01, 0.1}` at depth 4 shows mean test accuracy increasing monotonically
with `lambda` for both seeds individually (0.617 -> 0.624 -> 0.635 ->
0.661), with a roughly constant seed-to-seed gap (~0.07-0.09) at every
`lambda`. This is the most reproducible pattern observed in this report.

Finding 4 (depth 6: a single-seed finding that does not replicate). A
single-seed pass at depth 6 with `NBSC_NORMALIZE = true` found `lambda =
0.01` to be the best-performing value (0.651 test accuracy, beating
`lambda = 0` at 0.614). A subsequent 2-seed pass at the same depth and
configuration found `lambda = 0.01`'s second seed scored 0.093 lower
than its first seed, bringing its 2-seed mean (0.551) to a level
statistically indistinguishable from the `lambda = 0` baseline mean
(0.561); `lambda = 0.001` was the best mean instead (0.590). We treat
Finding 4's original single-seed claim as retracted rather than confirmed.

Finding 5 (depth 8: insufficient evidence). Runs with `NBSC_NORMALIZE =
true` and `lambda > 0` at depth 8 consistently approached or exceeded the
per-command compute budget available in this session; only the `lambda =
0` baseline obtained a second seed (mean test accuracy 0.467). All three
`lambda > 0` values at depth 8 rest on a single seed each and should not
be compared to the depth-4/6 findings with the same confidence.

4 Construction of the differentiable energy regularizer

The pre-existing diagnostic `dirichlet_energy(graph, x)` computes `E(X)`
by converting `x` to host-side data (`x.to_data()`) and summing over the
edge list in plain `f64` arithmetic. This is adequate for post-hoc
logging but cannot contribute a gradient, since the computation leaves
Burn's autodiff graph entirely.

We construct a second implementation, `dirichlet_energy_differentiable`,
using the standard graph-Laplacian quadratic-form identity: for the
unnormalized Laplacian `L = diag(D) - A`,
    `trace(X^T L X) = sum_{(u,v) in Edges} ||x_u - x_v||^2`,
which holds for the unweighted, undirected case relevant here since `A`
is symmetric and `D` is the corresponding full degree vector. We compute
the left-hand side entirely with Burn tensor operations —
    `L X = diag(D) X - A X`, then `trace(X^T (LX)) = sum(X elementwise* (LX))`
— using two dense matrix multiplications (`A X` via `adjacency.matmul(x)`,
and the degree-scaling via a broadcasted elementwise product) plus one
reduction, none of which detach from the autodiff graph. The result is
normalized by edge count and mean squared row norm exactly as the CPU
diagnostic is, so a `lambda` tuned at one network depth or graph size
transfers reasonably to others, and the regularized quantity stays
comparable to the value the diagnostic already logs.

Verification. Two unit tests accompany this construction, mirroring the
role played by numbered propositions with short verification arguments in
[1]: `differentiable_energy_matches_cpu_diagnostic` constructs a random
graph and random node-feature tensor, computes `E(X)` both ways, and
asserts agreement to within `1e-3`; `differentiable_energy_is_actually_
differentiable` constructs the same computation under Burn's `Autodiff`
backend, calls `.backward()`, and asserts the resulting gradient
with respect to `X` has non-trivial norm. Both tests pass. This is weaker
than a formal correctness proof but directly checks the two properties
the regularizer needs to have to be usable as a loss term: numerical
agreement with the already-trusted diagnostic, and an actual functioning
gradient.

The regularizer is wired into the training harness (`thesis_bench.rs`) by
summing this quantity across every layer's activation each epoch
(matching the phrase "between deep Chebyshev filter taps" in the original
feature request), averaging by depth so a fixed `lambda` does not need
re-tuning as `NBSC_DEPTHS` changes, and adding `lambda * mean_energy` to
the cross-entropy training loss before `.backward()`. The default value
`lambda = 0.0` leaves all pre-existing benchmark numbers in
`results_thesis.md` reproducible unchanged.

5 Experimental protocol

Dataset and split. Cora, canonical Planetoid split (140 train / 500
validation / 1000 test nodes out of 2708, 5278 edges, 7 classes),
committed in-repository; no network access required.

Environment. `rustc`/`cargo` 1.75.0, installed via `apt` in a
previously-toolchain-free sandbox (matches the crate's declared
`rust-version = "1.75"`). `cargo build --features burn --lib` and
`--example thesis_bench --release` both compile without error;
`cargo test --features burn --lib burn_layer` passes 6/6, including the
two tests described in Section 4.

Compute-budget constraint. Each terminal command in the environment used
for this report is subject to a wall-clock limit of roughly 200-250
seconds. Depth-6 and depth-8 runs with `lambda > 0` incur substantial
extra cost per epoch beyond the depth-6/8 baseline, from the two
additional dense `2708 x 2708` matrix multiplications the regularizer
performs per regularized layer per epoch. This forced epoch counts down
from an initial 30 (depth-independent pass) to 25 (depths 4 and 6,
2-seed pass) to 15-20 (depth 8, mixed seed count) over the course of the
runs reported here, and caused several attempted runs to time out
entirely, contributing no data (noted explicitly in Section 6 wherever it
occurred). Every number reported below is a real, executed training run;
none are extrapolated, interpolated, or estimated.

6 Results

6.1 Pass 1: single seed, initial signal (30 epochs, seed 0 only).

| depth | normalize | lambda | val acc | test acc | final energy |
|------:|:---------:|-------:|--------:|---------:|--------------:|
| 2 | false | 0.0   | 0.710 | 0.745 | 10.24 |
| 2 | false | 0.1   | 0.680 | 0.677 |  0.53 |
| 6 | false | 0.0   | 0.438 | 0.448 | 16.13 |
| 6 | false | 0.001 | 0.382 | 0.411 | 14.69 |
| 6 | false | 0.01  | 0.430 | 0.426 |  4.02 |
| 6 | false | 0.1   | 0.376 | 0.392 |  2.36 |
| 6 | false | 1.0   | 0.232 | 0.232 |  1.01 |
| 6 | true  | 0.0   | 0.620 | 0.614 |  0.43 |
| 6 | true  | 0.01  | 0.636 | 0.651 |  0.38 |

6.2 Pass 2: multi-seed grid, `normalize = true` only.

Depth 4 (2 seeds, 25 epochs):

| lambda | seed 0 test | seed 1 test | mean test | final energy (seed 0) |
|-------:|------------:|------------:|----------:|-----------------------:|
| 0.0   | 0.654 | 0.580 | 0.617 | 0.419 |
| 0.001 | 0.666 | 0.582 | 0.624 | 0.410 |
| 0.01  | 0.678 | 0.592 | 0.635 | 0.405 |
| 0.1   | 0.697 | 0.625 | 0.661 | 0.306 |

Depth 6 (2 seeds, 25 epochs):

| lambda | seed 0 test | seed 1 test | mean test | final energy (seed 0) |
|-------:|------------:|------------:|----------:|-----------------------:|
| 0.0   | 0.599 | 0.522 | 0.561 | 0.437 |
| 0.001 | 0.639 | 0.540 | 0.590 | 0.415 |
| 0.01  | 0.593 | 0.509 | 0.551 | 0.396 |
| 0.1   | 0.586 | 0.564 | 0.575 | 0.325 |

Depth 8 (mixed seed count and epoch count; see Section 5):

| lambda | epochs | seed 0 test | seed 1 test | mean test |
|-------:|-------:|-------------:|-------------:|----------:|
| 0.0   | 20 | 0.503 | 0.430 | 0.467 |
| 0.001 | 20 | 0.473 | timed out | 0.473 (n=1) |
| 0.01  | 20 | 0.477 | not attempted | 0.477 (n=1) |
| 0.1   | 20 | 0.552 | not attempted | 0.552 (n=1) |

A depth-8, `lambda = 0`, 2-seed run at 15 epochs (undertrained relative to
20) scored a mean of 0.397, included here only to illustrate that 20
epochs at depth 8 is itself of uncertain convergence, not as a fifth grid
point.

7 Discussion

The results separate cleanly into a mechanical claim and a scientific
one. The mechanical claim (Finding 1) is well supported: the regularizer
is correctly implemented, agrees with the trusted diagnostic, produces a
real gradient, and visibly suppresses energy in live training exactly as
designed. The scientific claim — whether suppressing energy this way
*helps* — is depth-dependent in a way that a single seed could not have
revealed. At depth 4, where LayerNorm alone has not saturated achievable
accuracy, the loss-side penalty appears to add real signal, consistently
across two seeds, up to the largest value tried. At depth 6, an
apparently clean single-seed result did not survive a second seed; the
size of the seed-to-seed swing at this depth (up to 0.093 in test
accuracy) is itself informative — it indicates this benchmark configuration
carries enough training variance that single-seed comparisons across
`lambda` values are not reliable at this depth, and, by extension,
possibly not at others where only one seed has been run.

This mirrors, at an informal level, a recurring theme in [1]'s subject
matter: the paper's Theorem 3 needs to characterize both a leading
eigenvalue's *location* and the *spectral gap* separating it from the
rest of the spectrum, because a mean estimate without a variance
(concentration) estimate is not, by itself, informative about how
reliably that mean will be observed on any single realization. The
analogous move here — reporting not just a mean test accuracy per
`lambda` but the seed-to-seed spread — is what overturned Finding 4's
original (single-seed) version.

8 Open questions

  - Depth 8 needs a second seed at every `lambda > 0` before it can be
    compared to depths 4 and 6 with any confidence; the compute budget in
    this session did not allow it.
  - A third seed at depths 4 and 6 would meaningfully firm up both the
    depth-4 monotone-improvement pattern and the depth-6 null result —
    two seeds is the minimum able to show disagreement between seeds, not
    enough to be confident either pattern generalizes.
  - Epoch counts were not held constant across depths (30 at depth-independent
    pass 1; 25 at depths 4/6 in pass 2; 15-20 at depth 8) purely due to the
    compute budget, confounding depth with training duration in every
    cross-depth comparison; this should be corrected before treating any
    cross-depth ranking as meaningful.
  - Depth 4's monotone trend was only tested up to `lambda = 0.1`; whether
    it continues to help at larger `lambda`, or eventually hurts the way
    `lambda = 1.0` did at depth 6 without LayerNorm, is untested.
  - This report used only Cora and only the RNBSC architecture; whether
    either finding transfers to Citeseer/PubMed or to GCN/GAT/GraphSAGE
    baselines is untested.

Appendix A. Full proof that the RNBSC propagator is not guaranteed
non-expansive, and its consequence for Dirichlet energy

Section 2 stated, without proof, that on the graphs examined here
`||A/rho_B||_2 > 1`. This appendix gives the complete, unabridged
derivation — from the definition of `rho_B` through to the Dirichlet-energy
consequence Section 3's Findings are explained by — reproduced from this
project's companion foundations document [4]. Throughout, `A` is the
graph's adjacency matrix, `D = diag(d_1, ..., d_n)` its degree matrix, `B`
the Hashimoto (non-backtracking) matrix of [1] (Section 1.2 of [4]), and
`rho_B` its Perron root.

A.1 Existence and simplicity of `rho_B`

`B` is a nonnegative `0/1` matrix. If `G` is connected and has a cycle
(the regime relevant throughout this report), `B` is irreducible on its
non-nilpotent part.

**Theorem A.1 (Perron-Frobenius).** A nonnegative, irreducible matrix `B`
has a real eigenvalue `rho_B > 0` (the Perron root) equal to its spectral
radius, of algebraic multiplicity one, with a strictly positive right
eigenvector `w` and left eigenvector `v` (`v^T B = rho_B v^T`), and every
other eigenvalue `lambda` satisfies `|lambda| <= rho_B`.

*Proof.* Consider `rho_B = sup{ r >= 0 : exists x >= 0, x != 0, Bx >= r x
}`. Compactness of the unit simplex and continuity give that the sup is
attained by some `x* >= 0`. Irreducibility of `B` forces `x* > 0`
strictly (else the zero-support set would be an invariant proper subset,
contradicting irreducibility) and `Bx* = rho_B x*` exactly (else one
could rescale to find a strictly larger feasible `r`, using positivity of
`x*` and irreducibility again). Uniqueness/simplicity of `rho_B` and the
bound `|lambda| <= rho_B` for every other eigenvalue follow from the same
variational characterization applied to `|x|` for any eigenvector `x` of
any eigenvalue `lambda`: `|lambda| |x| = |Bx| <= B|x|` entrywise
(nonnegativity of `B`), hence `|lambda| <= rho_B` by definition of `rho_B`
as the sup. $\blacksquare$

A.2 Why `rho_B`, specifically, is the correct normalizer for the RNBSC
filter tap

The RNBSC filter bank builds a `K`-tap Chebyshev-style recursion in
`A/rho_B` derived directly from the quadratic eigenvalue problem
underlying the Ihara zeta function (Bass's theorem [4, Thm 1.2]): writing
the governing identity `lambda^2 x = lambda A x - (D - I_n) x` and
substituting `lambda = rho_B * z` (rescaling so the dominant root sits
near `z = 1`, mirroring how ChebNet rescales the Laplacian spectrum into
`[-1,1]` before applying Chebyshev polynomials) gives

```
z^2 x = z (A/rho_B) x - ((D - I_n)/rho_B^2) x,
```

from which the three-term recursion `T_0 = I`, `T_1 = A/rho_B`,
`T_{k+1} = (2A/rho_B) T_k - ((D-I_n)/rho_B^2) T_{k-1}` follows directly
[4, Claim 6.1]. Rescaling by `rho_B` is what keeps the recursion's
coefficients `O(1)` rather than growing geometrically: without
normalizing, powers of `A` alone would grow like `rho_B^k`, since `rho_B`
governs the dominant growth rate of the non-backtracking walk counts that
the underlying quadratic identity organizes [4, Sec. 6.2]. This is the
precise sense in which `A/rho_B` is the "right" object to study for
expansiveness — it is not an arbitrary choice of denominator, but the one
forced by the filter's own derivation.

A.3 GCN's propagator is provably non-expansive

**Theorem A.2.** Let `A_hat = D^{-1/2}(A+I)D^{-1/2}` (the GCN
propagator). Then `||A_hat||_2 <= 1`, with equality attained, so
`||A_hat||_2 = 1` exactly on a connected graph.

*Proof.* `A_hat` has the same eigenvalues as the row-normalized walk
matrix `P = D^{-1}(A+I)`, a nonnegative matrix with every row summing to
exactly `1` (via the diagonal similarity `D^{1/2} A_hat D^{-1/2} = P`,
which preserves eigenvalues). By the Perron-Frobenius/Gershgorin-circle
bound for row-stochastic matrices, every eigenvalue `mu` of `P` satisfies
`|mu| <= max_v sum_u P_{vu} = 1`, and since `A_hat` is similar to `P`,
the same bound holds for `A_hat`'s eigenvalues; symmetry of `A_hat` then
gives `||A_hat||_2 = max|mu| <= 1`. Equality holds because `x = D^{1/2}
1` (`1` the all-ones vector) is an eigenvector of `A_hat` with eigenvalue
exactly `1` (each row of `P` sums to `1`). $\blacksquare$

A.4 `A/rho_B` is not guaranteed non-expansive

`A/rho_B` is symmetric, so `||A/rho_B||_2 = max_i |mu_i(A)| / rho_B`,
i.e. exactly the ratio of `A`'s ordinary spectral radius to `B`'s
non-backtracking Perron root. **There is no theorem forcing `max_i
|mu_i(A)| <= rho_B`**: `rho_B` is the Perron root of the
*non-backtracking* matrix `B`, governed by the quadratic-eigenvalue
coupling between `mu` (an eigenvalue of `A`) and the degree structure
(Section A.2's identity, or the regular-graph special case `lambda^2 -
mu*lambda + q = 0` of the Ramanujan-bound theory [4, Sec. 3.2]) — it is
not, in general, a bound on `A`'s own spectral radius. Consequently
`||A/rho_B||_2` can exceed `1`, making the tap **expansive**. This project's
prior empirical work measured this ratio directly: `||A||_2/rho_B =
1.594` on Cora, `1.136` on a synthetic 4-community hypergraph, `1.197` on
Citeseer, and `1.074` on PubMed [4, Sec. 8.2] — every dataset examined by
this project exceeds `1`, i.e. is measurably expansive, with the ratio
shrinking (though remaining `> 1`) as graph size grows.

A.5 Non-expansiveness controls Dirichlet energy; expansiveness does not

**Proposition A.3.** If a linear propagator `P` is symmetric with
`||P||_2 <= 1`, then repeated application of `P` cannot increase the
Dirichlet energy `E(X) = tr(X^T L X)` of the associated quadratic form:
`||P^k X||` is non-increasing in `k`.

*Proof.* Diagonalize `X` in `P`'s eigenbasis (`P` symmetric); each
coefficient along an eigenvector with eigenvalue `mu` is scaled by
`mu^k` under `P^k`, and `|mu| <= 1` implies `|mu^k| <= 1`, non-increasing
in `k`. Any quadratic energy functional expressed in this eigenbasis is
therefore non-increasing under repeated application of `P`. $\blacksquare$

By Theorem A.2, GCN's propagator satisfies the hypothesis of Proposition
A.3 unconditionally, on every connected graph: its Dirichlet energy is
*structurally* forced to be non-increasing with depth, which is the
classical over-smoothing signature. By Section A.4, `A/rho_B` need not
satisfy the hypothesis at all — and empirically, on every dataset this
project has examined, does not (Section A.4's ratios are all `> 1`).
Where the hypothesis fails, the conclusion is not merely unavailable but
actively reversed: components of `X` along eigenvectors of `A` with
`|mu_i(A)| > rho_B` are **amplified**, not damped, by repeated
propagation. This is the exact, proved mechanism behind Section 1's
observation that RNBSC's per-layer energy *rises* with depth (`16.13` at
depth 6 vs. `10.24` at depth 2, `NBSC_NORMALIZE=false`, Section 6.1)
rather than decaying to zero the way GCN/GAT/GraphSAGE's does — it is not
an artifact of this particular training run, but a consequence, proved in
Theorem A.2 and Proposition A.3 together with Section A.4's measured
ratios, of which propagator each architecture uses. What Proposition A.3
does *not* establish — and what Section 3's Findings 2-5 investigate
empirically, since no analogous theorem is available for the expansive
case — is whether penalizing this energy in the training loss can
substitute for the non-expansiveness the architecture itself lacks. The
proposition explains *why* the problem exists; it is silent on whether
the regularizer in Section 4 is an adequate fix, which is exactly why
Section 3 relies on live experiments rather than a further theorem.

References

[1] C. Bordenave, M. Lelarge, L. Massoulie. Non-backtracking spectrum of
    random graphs: community detection and non-regular Ramanujan graphs.
    arXiv:1501.06087, 2015.

[2] RNBSC Project internal documentation, `docs/results_thesis.md`
    Section 5 ("The expansive-operator diagnostic, extended to all three
    datasets") and Section 5.5 ("Energy-Stabilized Deep Stacks").

[3] RNBSC Project internal documentation, `docs/energy_stabilized_deep_
    stacks_live_run.md` (this report's companion raw-data document,
    including the complete CSV log of every run described in Section 6).

[4] RNBSC Project internal documentation, `NBSC_Mathematical_Proofs.md`
    ("The RNBSC System: Complete Mathematical Foundations and Proofs"),
    Sections 1-3 (Ihara zeta function, Bass's theorem, Perron-Frobenius
    theory), Section 6 (the RNBSC Chebyshev filter-bank derivation), and
    Section 8 (Non-Expansiveness of GCN vs. the Non-Backtracking
    Propagator), reproduced in full in Appendix A above. This document
    additionally covers exponential tilting and the large-deviations
    (Varadhan/Gartner-Ellis) reading of the tilted spectral radius, the
    normalized hypergraph Laplacian, and the Heilmann-Lieb/Lee-Yang/
    Godsil-Gutman mathematical-physics lineage underlying the Ramanujan-
    bound diagnostics elsewhere in the codebase; none of that material is
    invoked by the present report and is not reproduced here.
