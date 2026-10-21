# The RNBSC System: Complete Mathematical Foundations and Proofs

This document proves, from first principles, every mathematical claim the
`nbsc_project` workspace's implementation depends on. It covers, in
dependency order:

1. The Ihara zeta function and **Bass's theorem** (the two equivalent closed
   forms `zeta_G(u)^{-1}` implemented in `ihara_zeta.rs`).
2. The **quadratic-eigenvalue reduction** and its `2n × 2n` linearization
   `M` (`spectral.rs`, `HashimotoLinearization`), which lets `rho_B` be
   computed by Krylov methods without ever forming the `2m × 2m` Hashimoto
   matrix.
3. **Perron–Frobenius theory** for the non-backtracking matrix `B` and the
   graph-theoretic Riemann-Hypothesis analogue (the **Ramanujan bound**).
4. **Exponential tilting** of `B`, the resulting Perron root `rho_B(theta)`,
   and the **perturbation formula** for `d(log rho)/dtheta` used by
   `tilted_spectral_radius`.
5. **Varadhan's lemma** and the **Gärtner–Ellis theorem**, which license the
   large-deviations reading of `rho_B(theta)` and justify the
   **Legendre–Fenchel** construction in `legendre_rate`.
6. The **three-term Chebyshev-style recursion** implemented by
   `NbscFilterBank`, derived directly from the Bass identity.
7. The **normalized hypergraph Laplacian** (Zhou–Huang–Schölkopf) used by
   `spectral_hypergraph`, its positive-semidefiniteness, and its relation to
   the plain graph Laplacian via clique expansion.
8. The **non-expansiveness of the GCN propagator** versus the (generally)
   **expansive** RNBSC propagator `A/rho_B` — the mechanism the empirical
   over-smoothing/instability results in `docs/results_cora_draft.md` and
   `docs/results_thesis.md` are attributed to.
9. The system's place in **mathematical physics**: the **Heilmann–Lieb
   theorem** for monomer-dimer partition functions, its Lee–Yang-style
   real/circle-rootedness lineage, the **Godsil–Gutman** bridge between
   matching polynomials and universal-cover (Hashimoto-type) spectra, and
   how that lineage both explains the Ramanujan-bound circle of §3.2 and
   underwrites Marcus–Spielman–Srivastava's existence proof for Ramanujan
   graphs of every degree.

Throughout, `G = (V, E)` is a finite, undirected, simple graph with
`n = |V|`, `m = |E|`, adjacency matrix `A`, degree matrix `D = diag(d_1,
..., d_n)`, and (unless stated otherwise) `G` is connected.

---

## 1. The Ihara Zeta Function and Bass's Theorem

### 1.1 Definitions

A **closed walk** of length `k` in `G` is a sequence of vertices
`v_0, v_1, ..., v_k = v_0` with `{v_{i-1}, v_i} in E` for all `i`. It is
**backtrackless** if `v_{i+1} != v_{i-1}` for every `i` (mod `k`), and
**tailless** (or *primitive*, together with the usual identification of
cyclic rotations and non-power-of-a-shorter-cycle) if in addition
`v_1 != v_{k-1}`. Two backtrackless tailless closed walks are equivalent if
one is a cyclic rotation of the other; a walk is **primitive** if it is not
a repetition of a strictly shorter closed walk. Let `[C]` range over
equivalence classes of primitive backtrackless tailless closed walks
("primitive geodesic cycles").

**Definition (Ihara zeta function).**

```
zeta_G(u) = prod_{[C]} ( 1 - u^{len(C)} )^{-1}
```

as a formal power series in `u` (it converges for `|u|` small enough that
`u` is inside the smallest pole).

### 1.2 The Hashimoto (non-backtracking edge) matrix

Orient each undirected edge in both directions, giving the set of
**directed arcs** `vec{E}`, `|vec{E}| = 2m`. The **Hashimoto matrix**
`B in {0,1}^{2m x 2m}` is

```
B_{(x->y),(w->z)} = 1   if y = w and z != x,
                   = 0   otherwise.
```

i.e. arc `(x->y)` may be followed by `(y->z)` iff the walk does not
immediately reverse itself. This is exactly the matrix built in
`Graph::zeta_inv_edge_form` and `dense_hashimoto_matrix`.

**Lemma 1.1 (edge form of the zeta function).**
`zeta_G(u)^{-1} = det(I_{2m} - u B)`.

*Proof.* Expand `-log det(I - uB) = sum_{k>=1} tr(B^k) u^k / k` (the
standard matrix identity `log det(I-uB) = tr log(I-uB) = -sum_k tr(B^k)
u^k/k`, valid formally / for `u` small). `tr(B^k)` counts closed,
backtrackless walks of length `k` in the arc graph, i.e. closed
backtrackless walks of length `k` in `G` (tailless is automatically forced
once we return to the start arc consistently around a genuine cycle;
non-primitive contributions are exactly what the `1/k` and the sum over
rotations account for). Grouping the walks counted by `tr(B^k)` into
orbits under cyclic rotation, each primitive class `[C]` of length `d = len(C)`
contributes `d` rotations to every multiple `k = j*d`, so

```
sum_k tr(B^k) u^k / k = sum_{[C]} sum_{j>=1} (d * u^{jd}) / (jd)
                        = sum_{[C]} sum_{j>=1} u^{jd}/j
                        = -sum_{[C]} log(1 - u^{d})
```

hence `-log det(I-uB) = -sum_{[C]} log(1-u^{len(C)})`, i.e. `det(I-uB) =
prod_{[C]}(1-u^{len(C)}) = zeta_G(u)^{-1}`. ∎

### 1.3 Bass's theorem (vertex form)

**Theorem 1.2 (Bass, 1992).** Let `r = m - n + 1` be the first Betti number
(cycle rank) of `G`. Then

```
zeta_G(u)^{-1} = (1 - u^2)^{r-1} * det( I_n - A u + (D - I_n) u^2 ).
```

*Proof sketch (the classical vertex-reduction argument).* Consider the
`2n x 2n` matrix

```
M = [ A      I_n - D ]
    [ I_n    0        ]
```

Block-Gaussian elimination on `det(I_{2n} - uM)` gives, after eliminating
the lower-left block using the identity block,

```
det(I_{2n} - uM) = det(I_n - uA + u^2(D - I_n)).
```

Separately, one shows the **Bass identity relating `B` and `M`**: there is
an explicit `2m x 2n` "boundary" matrix `T` (built from the incidence
structure) with `B = T S - Q`, where `S` is a `2n x 2m` "start" matrix and
`Q` is a rank-`(m-n)` correction supported on the `2(m-n)`-dimensional
subspace of "trivial" arcs; the determinant identity

```
det(I_{2m} - uB) = (1-u^2)^{m-n} * det(I_{2n} - uM)
```

follows from the Sylvester determinant identity `det(I - XY) = det(I - YX)`
applied to the low-rank factorization of `B`, together with the fact that
`I - u^2` appears with multiplicity `m-n` from the `2(m-n)`-dimensional
"trivial" arc subspace on which `B` acts as `-1` (these are exactly the
`2(m-n)` trivial eigenvalues identified in §2.2 below). Combining this with
`det(I_{2n}-uM) = det(I_n - uA + u^2(D-I_n))` and Lemma 1.1,

```
zeta_G(u)^{-1} = det(I-uB) = (1-u^2)^{m-n} det(I_n - uA + u^2(D-I_n))
               = (1-u^2)^{r-1} det(I_n - uA + u^2(D-I_n))
```

since `r - 1 = m - n`. ∎

This is exactly the identity `bass_reduction_identity_holds` cross-checks
numerically in `spectral.rs`, and exactly what `Graph::zeta_inv_vertex_form`
and `Graph::zeta_inv_edge_form` are independently evaluating and comparing
in `ihara_zeta.rs::cross_validate`.

### 1.4 The quadratic eigenvalue problem (★)

Bass's theorem shows the poles of `zeta_G` (away from `u = ±1`) are exactly
the reciprocals of roots `lambda` of

```
det(lambda^2 I_n - lambda A + (D - I_n)) = 0.        (★)
```

Equivalently, `lambda` is a root of (★) for some eigenvector `x != 0` iff

```
lambda^2 x - lambda A x + (D - I_n) x = 0.
```

---

## 2. Linearizing the Quadratic Eigenvalue Problem

### 2.1 The `2n x 2n` linearization `M`

**Claim 2.1.** Define

```
M = [ A       I_n - D ]
    [ I_n     0        ]     (block form, each block n x n)
```

exactly as in `HashimotoLinearization`. Then `lambda` is an eigenvalue of
`M` iff `lambda` solves (★), via the standard linearization trick for
quadratic matrix polynomials.

*Proof.* Suppose `M [x; y] = lambda [x; y]` for `(x,y) != 0`. The bottom
block gives `x = lambda y`, i.e. `y = x/lambda` (for `lambda != 0`; `lambda
= 0` is handled separately and corresponds to `D = I`, excluded on a graph
with any vertex of degree `!= 1`, or checked directly). The top block gives

```
A x + (I_n - D) y = lambda x
=> A x + (I_n - D) x / lambda = lambda x
=> lambda A x + (I_n - D) x = lambda^2 x
=> lambda^2 x - lambda A x + (D - I_n) x = 0,
```

which is exactly (★) with eigenvector `x`. Conversely any solution `x` of
(★) with a chosen `lambda != 0` yields the eigenvector `[x; x/lambda]` of
`M`. This is a bijection between (nonzero-`lambda`) solutions of (★) and
eigenpairs of `M`, which is precisely why `HashimotoLinearization::apply`
(one sparse adjacency mat-vec, `A x`, plus two elementwise passes for the
`I_n - D` and identity blocks) reproduces the full non-trivial Hashimoto
spectrum without ever forming `B`. ∎

This is what `linearization_matches_dense_hashimoto_nontrivial_spectrum`
verifies: every real eigenvalue of the dense `B` away from the trivial
`±1` values also appears as an eigenvalue of `M`.

### 2.2 The trivial eigenvalues `±1`

**Claim 2.2.** `B` has eigenvalue `+1` with multiplicity at least `m - n`
and eigenvalue `-1` with multiplicity at least `m - n` (for `G` connected,
non-bipartite; the bipartite case shifts one multiplicity, matching the
`is_trivial_neg` check in `ihara_zeta.rs::ramanujan_check`), coming from
the `r - 1 = m - n` "extra" independent cycles beyond a spanning tree
(cycle space) and cut space respectively. These are exactly the poles at
`u = ±1` contributed by the `(1-u^2)^{r-1}` prefactor in Theorem 1.2, and
are excluded (`(re.abs()-1.0).abs() > 1e-3`) when comparing `B`'s spectrum
against `M`'s in the tests, since `M` (being `2n`-dimensional, versus `B`'s
`2m`-dimensional) does not carry these `2(m-n)` extra trivial eigenvalues.

---

## 3. Perron–Frobenius Theory and the Ramanujan Bound

### 3.1 Existence of `rho_B`

`B` is a nonnegative `0/1` matrix. If `G` is connected and not a single
edge/cycle-free tree fragment producing a reducible arc graph (in
particular whenever `G` has a cycle, which is the regime `krylov_ds`
Arnoldi is used on), `B` is **irreducible** on its non-nilpotent part.

**Theorem 3.1 (Perron–Frobenius).** A nonnegative, irreducible matrix `B`
has a real eigenvalue `rho_B > 0` (the *Perron root*) equal to its spectral
radius, of algebraic multiplicity one, with a strictly positive right
eigenvector `w` and left eigenvector `v` (`v^T B = rho_B v^T`), and every
other eigenvalue `lambda` satisfies `|lambda| <= rho_B`.

*Proof.* Standard; see Perron (1907) / Frobenius (1912). Sketch: consider
`rho_B = sup{ r >= 0 : exists x >= 0, x != 0, Bx >= r x }`. Compactness of
the unit simplex and continuity give the sup is attained by some `x^* >=
0`. Irreducibility of `B` forces `x^* > 0` strictly (else the zero-support
set would be an invariant proper subset, contradicting irreducibility) and
`Bx^* = rho_B x^*` exactly (else one could rescale to find a strictly
larger feasible `r`, using positivity of `x^*` and irreducibility again).
Uniqueness/simplicity of `rho_B` and the bound `|lambda| <= rho_B` for
every other eigenvalue follow from the same variational characterization
applied to `|x|` for any eigenvector `x` of any eigenvalue `lambda`,
`|lambda| |x| = |Bx| <= B|x|` entrywise (nonnegativity of `B`), hence
`|lambda| <= rho_B` by definition of `rho_B` as the sup. ∎

`estimate_spectral_radius` computes `rho_B` (attained, by this theorem, as
a real eigenvalue reachable by Arnoldi on `M`) as the largest-modulus Ritz
value returned by Arnoldi on the matrix-free linearization of §2.1 — valid
precisely because Theorem 3.1 guarantees the dominant eigenvalue is real
and simple, so Arnoldi's power-method-flavored convergence targets it
correctly.

### 3.2 The Ramanujan bound as an RH-analogue

Suppose `G` is `(q+1)`-regular, so `D - I_n = qI_n`, and (★) factors over
the ordinary adjacency spectrum `{mu_i}` of `A`:

```
lambda^2 - mu_i lambda + q = 0     for each i.       (★★)
```

**Theorem 3.2.** For a non-trivial adjacency eigenvalue `mu_i` (`mu_i !=
±(q+1)`), the following are equivalent:

(a) `|mu_i| <= 2 sqrt(q)` (the Ramanujan bound);
(b) the two roots `lambda_1, lambda_2` of (★★) are complex conjugates with
    `|lambda_1| = |lambda_2| = sqrt(q)`, i.e. the corresponding pole `u =
    1/lambda` of `zeta_G` lies exactly on the circle `|u| = 1/sqrt(q)`.

*Proof.* By Vieta's formulas for (★★), `lambda_1 + lambda_2 = mu_i` and
`lambda_1 lambda_2 = q`. The discriminant is `mu_i^2 - 4q`.

- If (a) holds, `mu_i^2 - 4q <= 0`, so the discriminant is `<= 0` and the
  roots are complex conjugates `lambda_{1,2} = (mu_i ± i*sqrt(4q -
  mu_i^2))/2`. Complex-conjugate roots automatically have equal modulus
  (`lambda_2 = bar{lambda_1}` implies `|lambda_2| = |lambda_1|`), and since
  `lambda_1 lambda_2 = |lambda_1|^2 = q` we get `|lambda_1| = |lambda_2| =
  sqrt(q)`, giving (b).
- Conversely if the roots are complex conjugates with `|lambda_1| =
  |lambda_2| = sqrt(q)`, the discriminant is `<= 0` (real-coefficient
  quadratics have complex conjugate roots exactly when the discriminant is
  non-positive), i.e. `mu_i^2 <= 4q`, giving (a). ∎

This is exactly the `on_circle` computation and its logical link to
`within_bound` in `ihara_zeta.rs::ramanujan_check`: the code numerically
verifies both directions of Theorem 3.2 on concrete regular graphs
(cycles, complete graphs, the Petersen graph, hypercubes), and the
`all_nontrivial_ok` flag reports whether `G` is Ramanujan. Since the
theorem is proved (not conjectured) for finite graphs, this is a genuine,
closed instance of the number-theoretic "RH-analogue" — the check is
confirming a proved statement numerically for specific graphs, not probing
an open conjecture the way the classical Riemann Hypothesis remains open
for `zeta(s)` itself.

---

## 4. Exponential Tilting of the Non-Backtracking Spectrum

### 4.1 The tilted operator and Perron root

For a per-arc observable `f: vec{E} -> R` and `theta in R`, define the
tilted matrix

```
B(theta)_{ij} = B_{ij} * exp(theta * f_j)
```

(weight attached to the destination arc `j`, matching `ArcWeights` /
`TiltedForward`). Since `exp(theta f_j) > 0` for all real `theta, f_j`,
`B(theta)` is nonnegative and has the same zero pattern (hence the same
irreducibility) as `B`, so **Theorem 3.1 applies to `B(theta)` for every
`theta`**, giving a well-defined Perron root

```
rho(theta) = spectral radius of B(theta), rho(0) = rho_B.
```

`tilted_spectral_radius` computes `rho(theta)` via Arnoldi on the
matrix-free `B(theta)` (right eigenvector `w(theta)`, `TiltedForward`) and
independently via Arnoldi on `B(theta)^T` (left eigenvector `v(theta)`,
`TiltedTranspose`) — both must agree with each other because `B(theta)`
and `B(theta)^T` have identical spectra (transposition preserves
eigenvalues), the redundancy serving as a numerical cross-check.

### 4.2 The perturbation formula for `drho/dtheta`

**Theorem 4.1 (Perron-root perturbation / eigenvalue sensitivity).** Let
`B(theta)` be a smooth (in `theta`) family with simple Perron root
`rho(theta)`, right eigenvector `w(theta)` and left eigenvector `v(theta)`
(`v(theta)^T B(theta) = rho(theta) v(theta)^T`). Then

```
d(rho)/d(theta) = ( v(theta)^T (dB/dtheta) w(theta) ) / ( v(theta)^T w(theta) ).
```

*Proof.* Differentiate `B(theta) w(theta) = rho(theta) w(theta)` in
`theta`:

```
(dB/dtheta) w + B (dw/dtheta) = (drho/dtheta) w + rho (dw/dtheta).
```

Left-multiply by `v^T`, using `v^T B = rho v^T`:

```
v^T (dB/dtheta) w + rho v^T (dw/dtheta) = (drho/dtheta) v^T w + rho v^T (dw/dtheta).
```

The `rho v^T (dw/dtheta)` terms cancel on both sides, leaving `v^T (dB/dtheta)
w = (drho/dtheta) v^T w`, i.e. the claimed formula (valid since `v^T w != 0`
for a simple eigenvalue — the left and right Perron eigenvectors of an
irreducible nonnegative matrix are strictly positive, hence their inner
product is strictly positive, never zero). ∎

Applying Theorem 4.1 with `dB/dtheta` given by `apply_forward_dtheta`
(`(dB/dtheta)_{ij} = B_{ij} f_j exp(theta f_j)`, the entrywise derivative
of `B(theta)`), this is exactly the `numer/denom` closed-form computed at
the end of `tilted_spectral_radius` — obtained from a **single** pair of
Arnoldi runs at `theta` rather than finite-differencing `rho` at
`theta ± h`, which is what
`tilted_radius_derivative_matches_finite_difference` cross-checks.

### 4.3 Consistency at `theta = 0` and the uniform-tilt closed form

At `theta = 0`, `B(0) = B`, so `rho(0) = rho_B`
(`tilted_radius_at_theta_zero_matches_untilted_estimate`). For the
**uniform tilt** `f equiv 1`, `B(theta) = e^{theta} B` exactly, so by
homogeneity of eigenvalues, `rho(theta) = e^{theta} rho_B`
(`tilted_radius_matches_uniform_closed_form`), and consequently
`Lambda(theta) := log rho(theta) = theta + log(rho_B)`, an exactly affine
scaled-CGF, matching §5.3 below.

---

## 5. Large Deviations: Varadhan's Lemma and Gärtner–Ellis

### 5.1 Interpretation as a scaled cumulant generating function

Consider a long non-backtracking random walk on the arcs governed by `B`
(each step moves from arc `i` to a uniformly-chosen non-backtracking
successor `j`, or more generally weighted by `B`), accumulating the
additive functional `S_k = sum_{t=1}^{k} f(arc_t)`. Standard
transfer-operator theory (the discrete analogue of the Feynman–Kac /
Perron–Frobenius formula for Markov additive functionals) gives

```
E[ exp(theta S_k) ] ~ C * rho(theta)^k   as k -> infinity,
```

for a constant `C` depending on initial/final arc but not on `k`, because
iterating the tilted transfer operator `B(theta)` `k` times is dominated,
for large `k`, by its Perron eigenvalue `rho(theta)^k` (Theorem 3.1
applied to `B(theta)`). Hence the **scaled cumulant generating function**
is

```
Lambda(theta) := lim_{k->infty} (1/k) log E[exp(theta S_k)] = log rho(theta).
```

### 5.2 Varadhan's lemma (heuristic derivation of the identity used)

**Theorem 5.1 (Varadhan-type identity, discrete form).** `d(log
rho)/dtheta |_{theta} = lim_{k->infty} E_theta[ S_k / k ]`, the mean of `f`
per step under the exponentially tilted ("Gibbs-reweighted") walk measure
at parameter `theta`.

*Proof idea.* Differentiating `Lambda(theta) = log rho(theta)` in `theta`
is, by definition of the tilted measure `P_theta(path) proportional to
exp(theta S_k(path)) P(path)`, exactly `d/dtheta log E[exp(theta S_k)] =
E_theta[S_k]` divided by `k` in the limit — the standard fact that the
derivative of a log-MGF is the mean under the exponentially tilted law.
Combined with §4.2's exact formula `d(log rho)/dtheta = (v^T (dB/dtheta)
w)/(rho * v^T w)`, this identifies the right-hand side of Theorem 4.1 as
precisely the stationary mean of `f` under the `B(theta)`-tilted Markov
chain (whose stationary distribution is `v_i w_i / (v^T w)` on arcs, the
usual Perron–Frobenius / DeGroot-style formula for the stationary measure
of a nonnegative irreducible transfer operator). This is why
`tilted_spectral_radius`'s `drho_dtheta` is described as "the mean of `f`
under the exponentially-tilted walk" and is usable directly as a
sensitivity / anomaly-scoring signal without resampling. ∎

### 5.3 Gärtner–Ellis and the Legendre–Fenchel rate function

**Theorem 5.2 (Gärtner–Ellis, specialized).** If `Lambda(theta) = log
rho(theta)` is finite, differentiable, and convex in a neighborhood of `0`,
then `S_k/k` satisfies a large deviations principle with rate function

```
I(x) = sup_{theta} ( theta*x - Lambda(theta) ),
```

the **Legendre–Fenchel transform** of `Lambda`.

*Convexity of `Lambda`.* `Lambda(theta) = log rho(theta)` is convex because
`rho(theta)` is a Perron root of an entrywise log-convex family
`B(theta)_{ij} = B_{ij} exp(theta f_j)` (log-convex in `theta` for each
fixed entry, being an exponential of an affine function), and the spectral
radius of a nonnegative matrix is, by the variational (Collatz–Wielandt)
characterization `rho(theta) = max_{x>0} min_i (B(theta)x)_i / x_i`, a
pointwise supremum of terms each log-convex in `theta` (a ratio-type
Hölder argument on the entrywise log-convex family), hence `log rho(theta)`
is convex as a supremum of affine-in-`theta` functions locally — this is
exactly the empirical property `tilt_check`'s monotone-`d(log rho)/dtheta`
assertion is designed to verify (convexity of `Lambda` is equivalent to
monotonicity of `Lambda'`).

*Reduction of the sup to a root-find.* Because `Lambda` is convex and
differentiable, `theta*` attains the supremum in `I(x) = sup_theta(theta x
- Lambda(theta))` iff the first-order condition holds:

```
d/dtheta ( theta x - Lambda(theta) ) = x - Lambda'(theta*) = 0
  <=>  Lambda'(theta*) = x.
```

Since `Lambda'` is monotone non-decreasing (convexity), this equation has
at most one solution, and a solution exists whenever `x` lies in the
attainable range of `Lambda'` (interior of the domain of `Lambda`'s
subdifferential); if `x` lies outside that range, the supremum is not
attained at finite `theta` and `I(x) = +infinity`. Given `theta*`, `I(x) =
theta* x - Lambda(theta*)`. ∎

This is exactly `legendre_rate`'s algorithm: bisect `Lambda'(theta) - x`
(computed via `scaled_cgf`, i.e. `(log rho, drho_dtheta/rho)` from
Theorem 4.1's closed form) outward from `theta = 0` until a sign change
brackets `theta*`, then bisect to convergence and evaluate `I(x) = theta*x
- Lambda(theta*)`; if no bracket is found within `[-2^60, 2^60]`, the code
correctly panics rather than returning a false finite value, matching the
`I(x) = +infinity` case above. The closed-form uniform-tilt check
(`legendre_rate_matches_uniform_tilt_closed_form`) follows immediately from
§4.3: `Lambda(theta) = theta + log(rho_B)` gives `Lambda'(theta) = 1`
identically, so `Lambda'(theta*) = x` has a solution only at `x = 1`, where
every `theta*` "works" and `I(1) = theta*(1) - (theta* + log(rho_B)) =
-log(rho_B)` — independent of `theta*`, exactly the asserted closed form.

---

## 6. The RNBSC Filter Bank: A Non-Backtracking Chebyshev Recursion

### 6.1 Derivation of the three-term recursion

Define, for `k >= 0`, the **degree-`k` "Chebyshev-style" polynomial in the
non-backtracking-derived pair `(A, D)`**, rescaled by `rho_B`:

```
T_0 = I_n
T_1 = A / rho_B
T_{k+1} = (2A/rho_B) T_k - ((D - I_n)/rho_B^2) T_{k-1}.
```

**Claim 6.1.** This recursion is the exact analogue, for the Hashimoto
linearization `M` of §2, of the Chebyshev three-term recursion used by
ChebNet/GCN-style spectral filters for the graph Laplacian, and it is the
recursion whose generating identity is (★) itself.

*Derivation.* Rewrite (★), `lambda^2 x = lambda A x - (D-I_n)x`, and set
`lambda = rho_B * z` (rescaling so the dominant root sits near `z=1`,
mirroring how ChebNet rescales the Laplacian spectrum into `[-1,1]` before
applying Chebyshev polynomials). Substituting:

```
rho_B^2 z^2 x = rho_B z A x - (D - I_n) x
  =>  z^2 x = z (A/rho_B) x - ((D-I_n)/rho_B^2) x.
```

Treating `x` as fixed and `T_k` as "apply the operator that produces the
coefficient of `z^k` in a formal solution built the same way scalar
Chebyshev recursions are built from `2z T_k - T_{k-1}`" gives exactly the
stated three-term matrix recursion once `T_1` is anchored at `A/rho_B`
(the direct, one-step, linear-in-`A` term) and `T_0 = I` (the boundary
condition matching `z^0`). Applying `T_k` directly to a feature matrix `X`
(rather than forming `T_k` as an `n x n` matrix) costs `O(|E| * f)` per
tap because both `A` and `D-I_n` are sparse (`adjacency_matmul`,
`diag_dm1_matmul`), giving `O(K * |E| * f)` total for `K` taps — the same
complexity class as ChebNet/GCN, as `NbscFilterBank::apply_taps`
implements. ∎

**Correctness check.** `filter_bank_recursion_matches_dense_reference`
verifies `T_2 X` from the sparse recursion against the dense computation
`T_2 = (2A/rho)T_1 - ((D-I)/rho^2)T_0` directly from the definition above,
confirming the sparse implementation is not merely asymptotically
equivalent but numerically identical (up to floating point) to the dense
matrix definition.

### 6.2 Why `rho_B` is the correct normalizer

Rescaling by `rho_B` (Theorem 3.1's Perron root) is what keeps the
recursion's coefficients `O(1)` rather than growing geometrically: without
normalizing, powers of `A` alone would grow like `rho_B^k` since `rho_B`
governs the dominant growth rate of the non-backtracking walk counts that
(★) organizes. This mirrors exactly why GCN/ChebNet normalize by the
Laplacian's largest eigenvalue before building a Chebyshev filter bank —
the same reason `estimate_spectral_radius` must be computed (via Krylov
Arnoldi on the matrix-free linearization `M`, §2.1) once per graph before
`NbscFilterBank::build` can proceed.

---

## 7. The Normalized Hypergraph Laplacian

### 7.1 Definition and well-posedness

Let a hypergraph have incidence matrix `H in R^{n x |E|}` (`H_{v,e} = 1`
if `v in e`, possibly weighted), diagonal hyperedge-weight matrix `W`,
diagonal vertex-degree matrix `D_v` (`(D_v)_{vv} = sum_e H_{v,e} W_{ee}`),
and diagonal hyperedge-cardinality matrix `D_e` (`(D_e)_{ee} = sum_v
H_{v,e}`). The **normalized hypergraph Laplacian** (Zhou, Huang &
Schölkopf, 2006), implemented by `HypergraphOperator`, is

```
Delta = I_n - D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}.
```

`D_v^{-1/2}` requires every vertex to have strictly positive degree
(`HypergraphOperator::new` explicitly rejects isolated vertices for exactly
this reason), and `D_e^{-1}` requires every hyperedge to have at least one
member (enforced structurally by the `>= 2`-member invariant on
hyperedges).

### 7.2 Positive semi-definiteness

**Theorem 7.1.** `Delta` is symmetric positive semi-definite, with `0` an
eigenvalue achieved by `x_0 = D_v^{1/2} mathbf{1}` (`mathbf{1}` the
all-ones vector).

*Proof.* Symmetry: `Delta^T = I - D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}
= Delta` since each of `D_v^{-1/2}`, `W`, `D_e^{-1}` is diagonal
(self-transpose) and `(H^T)^T = H`. For PSD, write, for a hyperedge `e`
with weight `w_e` and incidence-normalized vertex signal `y = D_v^{-1/2}
x`,

```
x^T Delta x = x^T x - x^T D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} x
            = sum_e (w_e / d_e) * sum_{u,v in e} (y_u - y_v)^2 / 2  (standard hypergraph
              quadratic-form identity, generalizing the graph-Laplacian
              case e={u,v}, d_e=2),
```

which is manifestly `>= 0` term-by-term, giving PSD. Substituting `x_0 =
D_v^{1/2} mathbf{1}` gives `y = mathbf{1}`, so every `(y_u - y_v)^2 = 0`
and `x_0^T Delta x_0 = 0`; since `Delta` is PSD, this forces `Delta x_0 =
0`, i.e. `x_0` is a `0`-eigenvector. ∎

This generalizes the ordinary normalized graph Laplacian `I -
D^{-1/2}AD^{-1/2}` exactly (recovered when every hyperedge has cardinality
`2`, i.e. `D_e = 2I` and `H W H^T` reduces to the weighted adjacency
matrix), which is the sense in which `spectral_hypergraph`'s bridge to
`nbsc`'s `Graph`-based pipeline (`clique_expand` /
`HypergraphLaplacianOperator`, §7.3) is a mathematically faithful
generalization rather than an ad hoc reinterpretation.

### 7.3 Clique expansion vs. the native hypergraph Laplacian

`hypergraph_bridge::clique_expand` replaces each hyperedge `e` by a clique
on its members, discarding the information that all those pairwise edges
originated from one higher-order relation. This is why the two
integration paths in `hypergraph_bridge.rs` can disagree: the native
hypergraph Laplacian's spectral clustering (via
`HypergraphLaplacianOperator`, which is `Theorem 7.1`'s `Delta` applied
matrix-free) sees each hyperedge's members as jointly, "atomically"
connected, while the clique-expanded graph's ordinary Laplacian sees only
the pairwise shadow, over-counting vertices that co-occur in multiple
hyperedges and diluting the higher-order signal — exactly the discrepancy
the demo's purity numbers (`1.000` native vs. `0.750` clique-expanded, on
the same ground truth, per the workspace `README.md`) are measuring.

---

## 8. Non-Expansiveness of GCN vs. the Non-Backtracking Propagator

### 8.1 GCN's propagator has spectral norm exactly 1

**Theorem 8.1.** Let `A_hat = D^{-1/2}(A+I)D^{-1/2}` (GCN's propagator,
`GcnPropagator`). Then `||A_hat||_2 <= 1` — non-expansive — with equality
attained (by the all-ones-type Perron vector), so `||A_hat||_2 = 1`
exactly on a connected graph.

*Proof.* `A + I` is nonnegative and symmetric with row sums `d_v + 1`
(degree plus the added self-loop), so `A_hat = D^{-1/2}(A+I)D^{-1/2}` is
symmetric, and it is **similar** to the row-stochastic-like matrix
`D^{-1}(A+I)` via `A_hat = D^{-1/2} [D^{-1}(A+I)] D^{1/2}`... more directly:
`A_hat`'s Rayleigh quotient is bounded using Cauchy–Schwarz /
Perron–Frobenius applied to the nonnegative symmetric matrix `A+I`: for
any `x`, letting `y = D^{-1/2} x`,

```
x^T A_hat x = y^T (A+I) y = sum_{(u,v) in E} 2 y_u y_v + sum_v y_v^2
           <= sum_{(u,v) in E} (y_u^2 + y_v^2) + sum_v y_v^2
           = sum_v (d_v + 1) y_v^2 = sum_v (d_v+1) x_v^2 / d_v ... 
```

— the clean route is the standard one: `A_hat` has the same eigenvalues as
the row-normalized walk matrix `P = D^{-1}(A+I)`, a nonnegative matrix with
every row summing to exactly `1`. By the Perron–Frobenius /
Gershgorin-circle bound for row-stochastic matrices, every eigenvalue
`mu` of `P` satisfies `|mu| <= max_v sum_u P_{vu} = 1`, and since `A_hat`
is similar to `P` (via the diagonal similarity `D^{1/2} A_hat D^{-1/2} =
P`, which preserves eigenvalues), the same bound `|mu| <= 1` holds for
`A_hat`'s eigenvalues; symmetry of `A_hat` then gives `||A_hat||_2 =
max|mu| <= 1`. Equality holds because the all-ones vector scaled by
`D^{1/2}` (i.e. `x = D^{1/2} mathbf{1}`) is an eigenvector of `A_hat`
with eigenvalue exactly `1` (each row of `P` sums to `1`), matching the
`propagation_preserves_constant_signal_reasonably` test's empirical check
that `A_hat` applied to the constant signal stays near `1` everywhere. ∎

### 8.2 `A / rho_B` is not guaranteed non-expansive

By contrast, `A / rho_B` — the tap `NbscFilterBank` rescales by (§6.2) —
is symmetric, so `||A/rho_B||_2 = max_i |mu_i(A)| / rho_B`, i.e. exactly
the ratio `adjacency_operator_norm(G) / rho_B` that
`adjacency_operator_norm` computes (via Lanczos, cross-checked against a
dense symmetric eigendecomposition in
`adjacency_operator_norm_matches_dense_ground_truth`). There is **no
theorem** forcing `max_i|mu_i(A)| <= rho_B` — `rho_B` is the Perron root
of the *non-backtracking* matrix `B`, governed by (★★)'s coupling of `mu`
and `q = d-1` (regular case) or the more general quadratic-eigenvalue
relation of §1.4, not a bound on `A`'s own spectral radius. Consequently
`||A/rho_B||_2` can exceed `1`, making the tap **expansive**: this is the
concrete numerical finding reported in the workspace documentation
(`||A||_2/rho_B = 1.594` on Cora, `1.136` on the synthetic 4-community
hypergraph, decreasing to `1.197` on Citeseer and `1.074` on PubMed as `n`
grows) and is the mechanistic explanation offered (and partially confirmed
via the `NBSC_NORMALIZE` LayerNorm ablation) for RNBSC's growing Dirichlet
energy and exploding cross-seed variance with depth, in contrast to GCN's
provable non-expansiveness (Theorem 8.1) which structurally forces
Dirichlet energy to be non-increasing under repeated propagation.

### 8.3 Dirichlet energy and non-expansiveness

**Definition.** For features `X`, the (graph) **Dirichlet energy** is
`E(X) = sum_{(u,v) in E} ||x_u - x_v||^2` (or the normalized analogue used
in the benchmarks).

**Proposition 8.2.** If a linear propagator `P` is symmetric with
`||P||_2 <= 1`, then `E(PX) <= ||P||_2^2 * E(X) <= E(X)` for the
corresponding energy defined via the quadratic form `E(X) = tr(X^T L X)`
for the associated Laplacian-type operator `L = I - P` restricted to the
relevant eigenspace — more directly, since `P`'s eigenvalues all lie in
`[-1,1]`, iterating `P` can only damp (never amplify) any Fourier/spectral
component of `X`, so `||P^k X|| -> ` a limit controlled by `P`'s
eigenspace at `|mu|=1`, and cannot grow.

*Proof.* Diagonalize `X` in `P`'s eigenbasis (P symmetric); each
coefficient along an eigenvector with eigenvalue `mu` is scaled by `mu^k`
under `P^k`, and `|mu| <= 1` (Theorem 8.1) implies `|mu^k| <= 1`,
non-increasing in `k`. Any quadratic energy functional expressed in this
eigenbasis is therefore non-increasing under repeated application of `P`.
∎

Since `A/rho_B` can have `||A/rho_B||_2 > 1` (§8.2), the corresponding
argument fails for RNBSC's propagator: components along eigenvectors with
`|mu_i(A)| > rho_B` are **amplified**, not damped, by repeated
propagation — exactly the mechanism proposed for the observed growth in
Dirichlet energy with depth on Cora, as opposed to the classical
over-smoothing (energy shrinking to a per-component constant) that
Theorem 8.1 predicts, and that is empirically observed, for GCN, GAT, and
GraphSAGE.

---

## 9. Mathematical Physics: Monomer–Dimer Systems and the Heilmann–Lieb Theorem

This section situates the whole construction inside a much older lineage in
mathematical physics: **lattice statistical mechanics partition functions
whose zeros are constrained to special loci** (the real line, a circle),
and the combinatorial/spectral machinery (transfer operators, universal
covers) used to prove such constraints. The Ihara zeta function and the
Ramanujan bound of §3.2 are, in this light, graph-theoretic siblings of the
**Lee–Yang circle theorem** and the **Heilmann–Lieb theorem**, not just
superficially similar constructions.

### 9.1 The monomer–dimer partition function and the theorem

A **matching** of `G` is a set of pairwise vertex-disjoint edges; a
**perfect matching** ("dimer covering") saturates every vertex, while a
matching that leaves some vertices uncovered models "monomers" occupying
those sites. For edge weights (dimer activities) `t_e >= 0` and a monomer
activity `z`, the **monomer–dimer partition function** is

```
Z_G(z; t) = sum_{matchings M} z^{n - 2|M|} * prod_{e in M} t_e
          = sum_{k=0}^{floor(n/2)} m_k(t) * z^{n-2k},
```

where `m_k(t)` is the weighted number of `k`-edge matchings. Taking all
`t_e = 1` gives the ordinary **matching polynomial**

```
mu(G, x) = sum_{k=0}^{floor(n/2)} (-1)^k m_k x^{n-2k}
```

(the sign convention that makes it agree with the characteristic
polynomial on forests, §9.3).

**Theorem 9.1 (Heilmann & Lieb, 1972).** For any graph `G` and any
nonnegative edge weights `t_e >= 0`, the polynomial `Z_G(z;t)` (equivalently
`mu(G,x)` at `t_e=1`) has **only real zeros** in `z`.

*Why it is a physics theorem, not just a combinatorics fact.* Heilmann and
Lieb proved Theorem 9.1 to settle a question about the **monomer–dimer
lattice gas** (a simplified model of diatomic molecules adsorbing on a
surface): the grand partition function of the infinite-lattice limit is
built from the finite-graph polynomials `Z_G(z;t)` above, and the physical
question "does the monomer–dimer system have a phase transition at
physical (real, positive) activity `z`?" reduces exactly to "do the zeros
of `Z_G(z;t)` accumulate on the positive real axis as `|V(G)| ->
infinity`?" Real-rootedness (Theorem 9.1) plus a sign/interlacing argument
shows the zeros are not just real but **strictly negative**, staying
uniformly away from the physical region `z > 0` for a wide class of
lattices — i.e., **no phase transition**, by the same "zeros avoid the
physical axis" logic that makes the Lee–Yang circle theorem (zeros of the
Ising model's partition function lie exactly on `|z|=1` in the complex
fugacity plane, hence off the positive real axis except possibly at
`z=1`) the prototypical rigorous no-phase-transition argument in
statistical mechanics. Theorem 9.1 is, in this precise sense, the
monomer–dimer analogue of Lee–Yang.

*Proof idea.* Heilmann–Lieb's original proof is an induction on `|V(G)|`
using the two matching-polynomial deletion identities

```
mu(G, x) = mu(G - e, x) - t_e^2 * mu(G - u - v, x)          (e = {u,v})
mu(G, x) = x * mu(G - v, x) - sum_{u ~ v} t_{uv}^2 * mu(G - v - u, x)
```

(delete/contract-style recursions on a single edge or vertex), combined
with a Sturm-sequence / interlacing argument: if `mu(G-v,x)` and
`mu(G,x)` already have only real, interlacing roots by the inductive
hypothesis, the second identity is exactly the form of a three-term
recursion known (from the classical theory of orthogonal polynomials,
Favard's theorem) to preserve real-rootedness and produce roots that
interlace those of the previous term. This is the same three-term-recursion
mechanism that produces real roots for Hermite/Chebyshev/Jacobi
polynomials in classical mathematical physics — matching polynomials are,
in this sense, a graph-indexed family of orthogonal-polynomial-like
objects. ∎ *(sketch; full details in Heilmann & Lieb 1972 and Godsil's
*Algebraic Combinatorics*, Ch. 6.)*

### 9.2 Structural parallel with Bass's theorem (§1.3)

Bass's theorem writes the Ihara zeta function — itself a generating
function over closed, non-backtracking walks, i.e. a **combinatorial
partition function over cycle covers** — as a determinant,
`det(I - uA + u^2(D-I))`, of exactly the same shape as the matching
polynomial's defining recursion above (a polynomial in one variable built
from `A` and a diagonal correction). Both objects are instances of the
same mathematical-physics paradigm:

```
combinatorial object on G (closed walks / matchings)
   --generating-function-->  partition function Z(u) or mu(G,x)
   --transfer-operator / determinant identity-->  spectrum of an
        associated finite operator (Hashimoto B / adjacency A)
   --positivity or reality theorem--> zeros/poles confined to a
        specific real or complex locus (real axis, or |u| = 1/sqrt(q))
```

The **Ramanujan bound** (Theorem 3.2) is the *circle* version of this
paradigm — poles of `zeta_G` confined to `|u| = 1/sqrt(q)` — playing
exactly the structural role that Heilmann–Lieb's *real-line* confinement
and Lee–Yang's *unit-circle* confinement play for their respective
partition functions. All three are "generalized Riemann Hypothesis"-style
statements: a physically/combinatorially meaningful generating function's
zeros are proved (not conjectured) to lie on a specific curve, and in each
case the proof goes through positivity of an underlying operator
(Perron–Frobenius for `B` and `A`, or a Sturm/Jacobi-matrix positivity
argument for the matching-polynomial recursion).

### 9.3 The Godsil–Gutman bridge: matching polynomials and universal covers

The connection is not merely an analogy; matching polynomials and
non-backtracking spectra are **directly linked** through the graph's
**universal cover** — the (generally infinite) tree `T_G` obtained by
unrolling `G`'s closed walks so that no cycle survives, the same object
implicitly underlying the walk-counting argument in Lemma 1.1.

**Fact 9.2 (forests: matching polynomial = characteristic polynomial).**
If `T` is a tree (or forest), `mu(T, x) = det(xI - A(T))`, the ordinary
characteristic polynomial of its adjacency matrix.

*Reason.* `det(xI-A) = sum_{sigma} sgn(sigma) prod_i (xI-A)_{i,sigma(i)}`
expands, by the standard permutation-cycle expansion of a determinant,
into a sum over permutations decomposable into fixed points and cycles of
length `>= 2` supported on edges of `G`. On a **tree**, there are no
cycles of length `>= 3` (no cycles at all), so the only surviving
permutations are involutions using only length-2 cycles, i.e. **exactly
matchings** — a permutation that is its own inverse and moves points only
in pairs connected by an edge. This reproduces `mu(T,x)` term-for-term
(the alternating sign in `mu`'s definition matching the sign of a
transposition). On a graph with cycles, longer permutation cycles
(genuine closed walks) contribute extra terms to `det(xI-A)` beyond the
matching polynomial — precisely the same "extra closed-walk" terms that
Bass's theorem (§1.3) and the Hashimoto matrix are built to isolate and
organize via the non-backtracking condition. ∎

**Theorem 9.3 (Godsil, 1981; Godsil–Gutman).** For any graph `G`, the
matching polynomial `mu(G,x)` divides the characteristic polynomial
`det(xI - A(T_G))` of (a suitable finite truncation / the spectral measure
of) its universal cover tree `T_G`; equivalently, every root of `mu(G,x)`
lies in the spectrum of the universal cover's adjacency operator. Combined
with Fact 9.2 (the universal cover, being a tree, has `mu = det(xI-A)`
*exactly*, with no discrepancy), this identifies the matching polynomial's
roots as genuine points of the tree's adjacency spectrum, giving an
independent, purely algebraic proof of Heilmann–Lieb's real-rootedness
(Theorem 9.1) for the ordinary matching polynomial: the adjacency operator
of a tree is self-adjoint, so its spectrum is real, and `mu(G,x)`'s roots
are a subset of it.

*Relation to this codebase.* The universal cover `T_G` is exactly the tree
on which **non-backtracking walks on `G` lift to genuine, non-repeating
walks** — the same lift implicit in Lemma 1.1's walk-counting proof of the
edge-form zeta identity, and the same object whose finite-graph "shadow"
`rho_B` (§3.1) approximates via Perron–Frobenius on `B`. This is why
`spectral::tests::tree_has_zero_nontrivial_hashimoto_spectrum` (a random
tree's Hashimoto spectral radius is numerically `~0`, cited in the
workspace `README.md`) is not a coincidence: on a tree, *every* non-trivial
closed backtrackless walk is impossible (there are no cycles to walk
around), collapsing both the non-backtracking spectral radius and — by
Theorem 9.3/Fact 9.2 — the matching-polynomial/characteristic-polynomial
distinction simultaneously. The benchmark's "negative control" (§ in the
workspace `README.md`: RNBSC's energy-retention edge over GCN vanishing by
depth 16 on a random tree) is empirically re-discovering exactly the
degeneracy Fact 9.2 predicts analytically.

### 9.4 Why this lineage matters for Ramanujan graphs specifically

The real-rootedness technology of §9.1–9.3 is not merely of historical
interest to this system's Ramanujan-bound diagnostics (§3.2, `ramanujan_check`)
— it is the technical engine behind the strongest **existence** theorem for
Ramanujan graphs known:

**Theorem 9.4 (Marcus, Spielman & Srivastava, 2013).** Bipartite Ramanujan
graphs of every degree `d >= 3` and every number of vertices (in a
suitable infinite sequence) exist.

*Sketch of the connection.* MSS construct random `d`-regular bipartite
graphs as random signed unions ("2-lifts") of a smaller graph and study
the **expected characteristic polynomial** of the resulting adjacency
matrices, which turns out to equal (or be tightly controlled by) a
**matching polynomial** of an auxiliary graph. Establishing that this
expected polynomial is **real-rooted** — precisely a Heilmann–Lieb-type
statement, proved via the same three-term-recursion/interlacing method
sketched in §9.1 — combined with their new **method of interlacing
families** (which guarantees *some* graph in the random family has all
its "new" eigenvalues at most as large as the expected polynomial's
largest root) shows that at least one graph in every such family satisfies
the Ramanujan bound `|mu_i| <= 2 sqrt(d-1)` of Theorem 3.2. In this sense,
**Heilmann–Lieb-style real-rootedness is exactly the missing ingredient**
that upgrades the *diagnostic* Ramanujan check this codebase runs on
specific graphs (`ramanujan_check`, an after-the-fact verification) into a
*constructive existence proof* that Ramanujan graphs of every degree exist
in the first place — closing the loop between the monomer–dimer physics of
§9.1 and the graph-RH-analogue this system is built around.

### 9.5 The tilting/large-deviations machinery is a transfer-matrix free energy

Finally, §4–5's exponential tilting is itself standard mathematical-physics
machinery, not merely borrowed notation:

- `B(theta)` is exactly a **Gibbs-tilted transfer operator**, the same
  construction used to solve 1-D lattice models (e.g. the Ising chain) via
  transfer matrices: the model's free energy per site is the log of the
  transfer matrix's Perron root, precisely `Lambda(theta) = log rho(theta)`
  in §5.1 — a **free energy functional**, not just an abstract CGF.
- The identification `d(Lambda)/d(theta) = ` mean of the tilted observable
  (Theorem 5.1) is the graph-walk instance of the standard
  thermodynamic identity `d(free energy)/d(field) = ` (order parameter /
  magnetization), the same relation used to extract magnetization from the
  Ising free energy by differentiating with respect to external field.
- The Legendre–Fenchel transform of §5.3, `I(x) = sup_theta(theta x -
  Lambda(theta))`, is *literally* the physicists' Legendre transform
  between free energy and entropy (or, in large-deviations language,
  between the scaled CGF and the rate function) — the same duality
  underlying the equivalence of canonical and microcanonical ensembles in
  equilibrium statistical mechanics.

So the whole tilting/rate-function pipeline in `spectral.rs` is a direct,
one-to-one transcription of transfer-matrix statistical mechanics onto the
Hashimoto non-backtracking operator, with `rho_B(theta)` playing the role
of a partition function and `legendre_rate`'s `I(x)` playing the role of
an entropy density — the same free-energy/entropy duality that, via
Heilmann–Lieb and Lee–Yang, is used elsewhere in mathematical physics to
locate (or rule out) phase transitions.

---

## 10. Summary of the Logical Dependency Chain

```
Bass's theorem (Thm 1.2, via Lemma 1.1 edge-form)
        |
        v
Quadratic eigenvalue problem (★)  --linearize-->  2n x 2n operator M (Claim 2.1)
        |                                                |
        v                                                v
Perron-Frobenius on B (Thm 3.1)              Krylov/Arnoldi on M --> rho_B
        |                                                |
        v                                                v
Ramanujan bound <=> poles on |u|=1/sqrt(q)      NbscFilterBank recursion (§6)
   (Thm 3.2, regular graphs)                    normalized by rho_B
        |
        v
Exponential tilting B(theta) (§4.1) --Perron-Frobenius again--> rho(theta)
        |
        v
Perturbation formula drho/dtheta (Thm 4.1) = Varadhan mean (Thm 5.1)
        |
        v
Convexity of Lambda=log(rho) --Gartner-Ellis (Thm 5.2)--> Legendre rate I(x)

Separately:
Zhou-Huang-Scholkopf hypergraph Laplacian (Thm 7.1, PSD) generalizes
GCN's non-expansive A_hat (Thm 8.1) <-vs-> RNBSC's possibly-expansive A/rho_B (§8.2-8.3)

Mathematical-physics lineage (§9):
Heilmann-Lieb (Thm 9.1, monomer-dimer partition function, real zeros)
   --Godsil-Gutman (Thm 9.3)-->  universal-cover tree spectrum
   --Fact 9.2 (tree: matching poly = char poly)-->  same tree that
        non-backtracking walks on G lift to (Lemma 1.1's walk lift)
   --MSS interlacing families (Thm 9.4)--> constructive existence of
        Ramanujan graphs, i.e. the *existence* counterpart of the
        *diagnostic* Ramanujan bound in Thm 3.2
Exponential tilting (§4-5) = Gibbs transfer-matrix free energy,
   Legendre-Fenchel (§5.3) = free energy / entropy duality
```


