// ihara_zeta.rs
//
// Implements the Ihara zeta function of a finite graph via two independent,
// algebraically equivalent routes, cross-validates them against each other,
// and (for regular graphs) checks the graph-theoretic Riemann-Hypothesis
// analogue: is the graph Ramanujan?
//
// Background:
//   zeta_G(u) = prod over primitive closed backtrackless tailless cycles [C]
//               of (1 - u^len(C))^{-1}
//
//   Bass's theorem gives a closed rational form purely from ordinary spectral
//   data (n = |V|, m = |E|, A = adjacency matrix, D = degree matrix,
//   r = m - n + 1 = first Betti number):
//
//     zeta_G(u)^{-1} = (1-u^2)^{r-1} * det(I_n - A u + (D-I) u^2)      [vertex form]
//
//   Equivalently, via the Hashimoto/non-backtracking edge matrix B
//   (size 2m x 2m, one row/column per directed arc):
//
//     zeta_G(u)^{-1} = det(I_{2m} - u B)                              [edge form]
//
//   These two formulas are proven identical; we implement both independently
//   and cross-check them numerically, which is a much stronger correctness
//   demonstration than trusting either formula's transcription by itself.
//
//   For a (q+1)-regular graph, D-I = q*I, so the vertex form factors over the
//   *adjacency* spectrum alone:
//     zeta_G(u)^{-1} = (1-u^2)^{r-1} * prod_i (1 - mu_i u + q u^2)
//   and each factor's roots (poles of zeta_G) satisfy lambda^2 - mu_i lambda + q = 0.
//   The graph is Ramanujan iff every non-trivial mu_i satisfies |mu_i| <= 2*sqrt(q),
//   which is *exactly* the condition that both roots lambda of that quadratic
//   are complex with |lambda| = sqrt(q) -- i.e. the corresponding poles of
//   zeta_G lie exactly on the circle |u| = 1/sqrt(q), the graph analogue of
//   the critical line Re(s) = 1/2. This is a genuine, solved RH-analogue,
//   not a numerical illustration of an open problem.
//
// Run: rustc -O ihara_zeta.rs -o ihara_zeta && ./ihara_zeta

use std::f64::consts::PI;

// ==========================================================================
// Complex arithmetic
// ==========================================================================

#[derive(Debug, Clone, Copy)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    fn zero() -> Self {
        Self::new(0.0, 0.0)
    }
    fn one() -> Self {
        Self::new(1.0, 0.0)
    }
    fn abs(&self) -> f64 {
        self.re.hypot(self.im)
    }
    fn add(self, o: Complex) -> Complex {
        Complex::new(self.re + o.re, self.im + o.im)
    }
    fn sub(self, o: Complex) -> Complex {
        Complex::new(self.re - o.re, self.im - o.im)
    }
    fn mul(self, o: Complex) -> Complex {
        Complex::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
    fn scale(self, r: f64) -> Complex {
        Complex::new(self.re * r, self.im * r)
    }
    fn div(self, o: Complex) -> Complex {
        let d = o.re * o.re + o.im * o.im;
        Complex::new((self.re * o.re + self.im * o.im) / d, (self.im * o.re - self.re * o.im) / d)
    }
    fn sqrt(self) -> Complex {
        // Standard complex square root via polar form.
        let r = self.abs();
        let re = ((r + self.re) / 2.0).sqrt();
        let mut im = ((r - self.re) / 2.0).sqrt();
        if self.im < 0.0 {
            im = -im;
        }
        Complex::new(re, im)
    }
}

/// Determinant of an n x n complex matrix via Gaussian elimination with
/// partial pivoting. Consumes (overwrites) the input matrix.
fn complex_det(mat: &mut [Vec<Complex>]) -> Complex {
    let n = mat.len();
    let mut det = Complex::one();
    for col in 0..n {
        let mut pivot_row = col;
        let mut max_abs = mat[col][col].abs();
        for r in (col + 1)..n {
            let a = mat[r][col].abs();
            if a > max_abs {
                max_abs = a;
                pivot_row = r;
            }
        }
        if max_abs < 1e-13 {
            return Complex::zero();
        }
        if pivot_row != col {
            mat.swap(pivot_row, col);
            det = det.mul(Complex::new(-1.0, 0.0));
        }
        let pivot = mat[col][col];
        det = det.mul(pivot);
        for r in (col + 1)..n {
            let factor = mat[r][col].div(pivot);
            if factor.abs() == 0.0 {
                continue;
            }
            for c in col..n {
                let sub = factor.mul(mat[col][c]);
                mat[r][c] = mat[r][c].sub(sub);
            }
        }
    }
    det
}

// ==========================================================================
// Graph representation
// ==========================================================================

struct Graph {
    n: usize,
    edges: Vec<(usize, usize)>, // undirected, simple, u < v
}

impl Graph {
    fn new(n: usize, edges: Vec<(usize, usize)>) -> Self {
        for &(u, v) in &edges {
            assert!(u < n && v < n && u != v, "invalid edge ({u},{v})");
        }
        Graph { n, edges }
    }

    fn degrees(&self) -> Vec<usize> {
        let mut d = vec![0usize; self.n];
        for &(u, v) in &self.edges {
            d[u] += 1;
            d[v] += 1;
        }
        d
    }

    fn is_regular(&self) -> Option<usize> {
        let d = self.degrees();
        let first = d[0];
        if d.iter().all(|&x| x == first) {
            Some(first)
        } else {
            None
        }
    }

    fn adjacency_f64(&self) -> Vec<Vec<f64>> {
        let mut a = vec![vec![0.0; self.n]; self.n];
        for &(u, v) in &self.edges {
            a[u][v] = 1.0;
            a[v][u] = 1.0;
        }
        a
    }

    fn betti_number(&self) -> i64 {
        self.edges.len() as i64 - self.n as i64 + 1
    }

    /// Simple BFS 2-coloring to test bipartiteness (needed to know whether
    /// -d is also a "trivial" eigenvalue for a d-regular graph).
    fn is_bipartite(&self) -> bool {
        let mut adj = vec![Vec::new(); self.n];
        for &(u, v) in &self.edges {
            adj[u].push(v);
            adj[v].push(u);
        }
        let mut color = vec![-1i8; self.n];
        for start in 0..self.n {
            if color[start] != -1 {
                continue;
            }
            color[start] = 0;
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(start);
            while let Some(u) = queue.pop_front() {
                for &v in &adj[u] {
                    if color[v] == -1 {
                        color[v] = 1 - color[u];
                        queue.push_back(v);
                    } else if color[v] == color[u] {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Vertex-form Bass evaluation of zeta_G(u)^{-1} at a complex point u.
    fn zeta_inv_vertex_form(&self, u: Complex) -> Complex {
        let a = self.adjacency_f64();
        let d = self.degrees();
        let n = self.n;
        let mut m: Vec<Vec<Complex>> = vec![vec![Complex::zero(); n]; n];
        let u2 = u.mul(u);
        for i in 0..n {
            for j in 0..n {
                let mut val = if i == j { Complex::one() } else { Complex::zero() };
                val = val.sub(u.scale(a[i][j]));
                if i == j {
                    let dmin1 = (d[i] as f64) - 1.0;
                    val = val.add(u2.scale(dmin1));
                }
                m[i][j] = val;
            }
        }
        let det = complex_det(&mut m);
        let r = self.betti_number();
        let prefactor_base = Complex::one().sub(u2); // (1-u^2)
        let prefactor = complex_powi(prefactor_base, r - 1);
        prefactor.mul(det)
    }

    /// Edge-form (Hashimoto non-backtracking operator) evaluation of
    /// zeta_G(u)^{-1} at a complex point u.
    fn zeta_inv_edge_form(&self, u: Complex) -> Complex {
        // Build directed arcs: each undirected edge (u,v) becomes arcs u->v and v->u.
        let arcs: Vec<(usize, usize)> = self
            .edges
            .iter()
            .flat_map(|&(u, v)| vec![(u, v), (v, u)])
            .collect();
        let m2 = arcs.len();
        let mut b: Vec<Vec<f64>> = vec![vec![0.0; m2]; m2];
        for (i, &(x, y)) in arcs.iter().enumerate() {
            for (j, &(w, z)) in arcs.iter().enumerate() {
                // Non-backtracking: arc (x->y) is followed by arc (y->z) as long
                // as we don't immediately reverse (z != x).
                if y == w && z != x {
                    b[i][j] = 1.0;
                }
            }
        }
        let mut mat: Vec<Vec<Complex>> = vec![vec![Complex::zero(); m2]; m2];
        for i in 0..m2 {
            for j in 0..m2 {
                let mut val = if i == j { Complex::one() } else { Complex::zero() };
                val = val.sub(u.scale(b[i][j]));
                mat[i][j] = val;
            }
        }
        complex_det(&mut mat)
    }
}

fn complex_powi(base: Complex, mut p: i64) -> Complex {
    if p == 0 {
        return Complex::one();
    }
    let inverse = p < 0;
    if inverse {
        p = -p;
    }
    let mut result = Complex::one();
    for _ in 0..p {
        result = result.mul(base);
    }
    if inverse {
        Complex::one().div(result)
    } else {
        result
    }
}

// ==========================================================================
// Jacobi eigenvalue algorithm for real symmetric matrices
// ==========================================================================

fn jacobi_eigenvalues(a_in: &[Vec<f64>], max_sweeps: usize, tol: f64) -> Vec<f64> {
    let n = a_in.len();
    let mut a = a_in.to_vec();
    for _sweep in 0..max_sweeps {
        let mut off_diag_norm = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off_diag_norm += a[p][q] * a[p][q];
            }
        }
        if off_diag_norm.sqrt() < tol {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                if a[p][q].abs() < 1e-15 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (theta * theta + 1.0).sqrt())
                } else {
                    -1.0 / (-theta + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                let app = a[p][p];
                let aqq = a[q][q];
                let apq = a[p][q];
                a[p][p] = app - t * apq;
                a[q][q] = aqq + t * apq;
                a[p][q] = 0.0;
                a[q][p] = 0.0;
                for i in 0..n {
                    if i != p && i != q {
                        let aip = a[i][p];
                        let aiq = a[i][q];
                        a[i][p] = c * aip - s * aiq;
                        a[p][i] = a[i][p];
                        a[i][q] = s * aip + c * aiq;
                        a[q][i] = a[i][q];
                    }
                }
            }
        }
    }
    let mut eigs: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    eigs.sort_by(|x, y| y.partial_cmp(x).unwrap()); // descending
    eigs
}

// ==========================================================================
// Test graph constructors
// ==========================================================================

fn cycle_graph(n: usize) -> Graph {
    let edges = (0..n).map(|i| (i, (i + 1) % n)).map(|(a, b)| if a < b { (a, b) } else { (b, a) }).collect();
    Graph::new(n, edges)
}

fn complete_graph(n: usize) -> Graph {
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            edges.push((i, j));
        }
    }
    Graph::new(n, edges)
}

fn petersen_graph() -> Graph {
    // Outer 5-cycle: 0-1-2-3-4-0
    // Inner 5-cycle (pentagram, step 2): 5-7-9-6-8-5
    // Spokes: i -- i+5
    let mut edges = Vec::new();
    for i in 0..5 {
        edges.push((i, (i + 1) % 5));
    }
    for i in 0..5 {
        let a = 5 + i;
        let b = 5 + (i + 2) % 5;
        edges.push((a.min(b), a.max(b)));
    }
    for i in 0..5 {
        edges.push((i, i + 5));
    }
    edges.sort();
    edges.dedup();
    Graph::new(10, edges)
}

fn hypercube_graph(d: usize) -> Graph {
    let n = 1usize << d;
    let mut edges = Vec::new();
    for v in 0..n {
        for bit in 0..d {
            let w = v ^ (1 << bit);
            if v < w {
                edges.push((v, w));
            }
        }
    }
    Graph::new(n, edges)
}

/// A small, deliberately non-regular graph (for cross-validating the two
/// zeta formulas on a case where the elegant regular-graph shortcut doesn't
/// apply): the "bull" graph plus a couple of extra edges, min degree 2.
fn irregular_test_graph() -> Graph {
    // 6 vertices, mixed degrees, connected, min degree 2.
    let edges = vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 5)];
    Graph::new(6, edges)
}

// ==========================================================================
// Cross-validation: vertex-form vs edge-form, at several sample points
// ==========================================================================

fn cross_validate(name: &str, g: &Graph) {
    println!("--- Cross-validation: {name} (n={}, m={}) ---", g.n, g.edges.len());
    let sample_points = [
        Complex::new(0.05, 0.0),
        Complex::new(0.1, 0.05),
        Complex::new(-0.07, 0.03),
        Complex::new(0.0, 0.08),
    ];
    for &u in &sample_points {
        let v_form = g.zeta_inv_vertex_form(u);
        let e_form = g.zeta_inv_edge_form(u);
        let diff = v_form.sub(e_form).abs();
        let scale = v_form.abs().max(e_form.abs()).max(1e-30);
        println!(
            "  u = {:>+.3}{:>+.3}i   vertex-form = {:>+.6}{:>+.6}i   edge-form = {:>+.6}{:>+.6}i   rel.diff = {:.2e}",
            u.re, u.im, v_form.re, v_form.im, e_form.re, e_form.im, diff / scale
        );
    }
    println!();
}

// ==========================================================================
// Ramanujan / graph-RH-analogue check for regular graphs
// ==========================================================================

fn ramanujan_check(name: &str, g: &Graph) {
    let d = match g.is_regular() {
        Some(d) => d,
        None => {
            println!("--- {name}: not regular, skipping Ramanujan check ---\n");
            return;
        }
    };
    let q = (d as f64) - 1.0;
    let bound = 2.0 * q.sqrt();
    let bipartite = g.is_bipartite();
    let a = g.adjacency_f64();
    let mu = jacobi_eigenvalues(&a, 200, 1e-12);

    println!("--- Ramanujan check: {name} ({d}-regular, n={}, bipartite={bipartite}) ---", g.n);
    println!("  Ramanujan bound: |mu| <= 2*sqrt(d-1) = {:.6}", bound);
    println!(
        "  {:>10} {:>10} {:>10} {:>28} {:>12} {:>12} {:>10}",
        "mu_i", "trivial?", "|mu_i|<=bd", "poles lambda (quadratic)", "|lambda_1|", "|lambda_2|", "on circle?"
    );

    let mut all_nontrivial_ok = true;
    for &m in &mu {
        let is_trivial_pos = (m - d as f64).abs() < 1e-6;
        let is_trivial_neg = bipartite && (m + d as f64).abs() < 1e-6;
        let trivial = is_trivial_pos || is_trivial_neg;

        // Solve lambda^2 - mu*lambda + q = 0 via complex quadratic formula.
        let disc = Complex::new(m * m - 4.0 * q, 0.0);
        let sqrt_disc = disc.sqrt();
        let lambda1 = Complex::new(m, 0.0).add(sqrt_disc).scale(0.5);
        let lambda2 = Complex::new(m, 0.0).sub(sqrt_disc).scale(0.5);
        let sqrt_q = q.sqrt();
        let on_circle = (lambda1.abs() - sqrt_q).abs() < 1e-6 && (lambda2.abs() - sqrt_q).abs() < 1e-6;

        let within_bound = m.abs() <= bound + 1e-9;
        if !trivial && !within_bound {
            all_nontrivial_ok = false;
        }

        println!(
            "  {:>10.4} {:>10} {:>10} lambda1={:>+.3}{:>+.3}i lambda2={:>+.3}{:>+.3}i {:>12.4} {:>12.4} {:>10}",
            m,
            if trivial { "yes" } else { "no" },
            if trivial { "n/a".to_string() } else { within_bound.to_string() },
            lambda1.re,
            lambda1.im,
            lambda2.re,
            lambda2.im,
            lambda1.abs(),
            lambda2.abs(),
            if trivial { "n/a".to_string() } else { on_circle.to_string() }
        );
    }
    println!(
        "  => {name} is {}Ramanujan (every non-trivial adjacency eigenvalue {} the bound).\n",
        if all_nontrivial_ok { "" } else { "NOT " },
        if all_nontrivial_ok { "respects" } else { "violates" }
    );
}

fn main() {
    println!("=== Ihara zeta function: Bass's formula vs the Hashimoto operator ===\n");

    cross_validate("cycle C_5", &cycle_graph(5));
    cross_validate("complete graph K_5", &complete_graph(5));
    cross_validate("Petersen graph", &petersen_graph());
    cross_validate("hypercube Q_3", &hypercube_graph(3));
    cross_validate("irregular test graph", &irregular_test_graph());

    println!("=== Graph-RH-analogue (Ramanujan) checks, regular graphs only ===\n");
    ramanujan_check("cycle C_5", &cycle_graph(5));
    ramanujan_check("cycle C_12", &cycle_graph(12));
    ramanujan_check("complete graph K_5", &complete_graph(5));
    ramanujan_check("complete graph K_10", &complete_graph(10));
    ramanujan_check("Petersen graph", &petersen_graph());
    ramanujan_check("hypercube Q_3", &hypercube_graph(3));
    ramanujan_check("irregular test graph", &irregular_test_graph());

    println!(
        "Interpretation: when a non-trivial adjacency eigenvalue mu satisfies the bound, \
the quadratic lambda^2 - mu*lambda + q = 0 necessarily has complex-conjugate roots of equal \
modulus sqrt(q) (Vieta: lambda1*lambda2 = q, and complex-conjugate roots always have equal \
modulus) -- i.e. that pole of the Ihara zeta function sits exactly on the circle |u| = 1/sqrt(q), \
the graph analogue of the critical line. This is a solved theorem for Ramanujan graphs, not a \
conjecture: the check above is verifying a known-true statement numerically, exactly the \
distinction that matters when comparing this to the (still open) Riemann Hypothesis itself."
    );

    // Silence unused-import warning if PI is not otherwise used in the final trimmed version.
    let _ = PI;
}
