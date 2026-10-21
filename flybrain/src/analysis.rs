//! Spectral and structural diagnostics of a connectome, reusing the
//! workspace's own machinery:
//!
//! * `nbsc::spectral` — Hashimoto/non-backtracking radius `rho_B` and the
//!   expansivity ratio `||A||_2 / rho_B` of the symmetrized wiring graph
//!   (the quantity the energy-stabilized-deep-stacks report shows decides
//!   whether deep propagation over this graph smooths or amplifies);
//! * `krylov_ds` Arnoldi — spectral radius of the **directed, signed**
//!   weight matrix `W` (the stability-relevant quantity for a recurrent
//!   network; symmetrization throws it away);
//! * `shpinn::embedding` — Laplacian eigenmap of the axonal-arbor
//!   hypergraph, and how well its low modes recover neuropil modules.
//!
//! Every statistic can be computed for the connectome and for a null model
//! so "what does the real wiring buy you" is a measured difference.

use krylov_ds::{Arnoldi, DenseMatrix};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::connectome::Connectome;
use crate::error::Result;

#[derive(Clone, Debug)]
pub struct WiringStats {
    pub n: usize,
    pub connected_pairs: usize,
    pub total_synapses: f64,
    pub density: f64,
    pub reciprocity: f64,
    pub mean_out_degree: f64,
    pub max_out_degree: usize,
    pub max_in_degree: usize,
    pub clustering: f64,
    pub frac_inhibitory: f64,
}

#[derive(Clone, Debug)]
pub struct SpectralStats {
    /// Spectral radius of the signed (Dale's-law) count matrix.
    pub rho_signed: f64,
    /// Perron radius of the unsigned count matrix.
    pub rho_abs: f64,
    /// Non-backtracking radius of the symmetrized graph's largest component.
    pub rho_b: f64,
    /// `||A||_2` of that component.
    pub adj_norm: f64,
    /// `||A||_2 / rho_B`: `> 1` means the rescaled operator `A/rho_B` is expansive.
    pub expansivity: f64,
    pub lcc_size: usize,
}

#[derive(Clone, Debug)]
pub struct ConnectomeReport {
    pub wiring: WiringStats,
    pub spectral: SpectralStats,
}

pub fn wiring_stats(c: &Connectome) -> WiringStats {
    let out = c.out_degrees();
    let inn = c.in_degrees();
    let pairs = c.n_pairs();
    let inter_or_sens: Vec<usize> = (0..c.n).filter(|&i| out[i] > 0).collect();
    WiringStats {
        n: c.n,
        connected_pairs: pairs,
        total_synapses: c.total_synapses(),
        density: pairs as f64 / (c.n * (c.n - 1)) as f64,
        reciprocity: c.reciprocity(),
        mean_out_degree: pairs as f64 / c.n as f64,
        max_out_degree: out.iter().copied().max().unwrap_or(0),
        max_in_degree: inn.iter().copied().max().unwrap_or(0),
        clustering: c.clustering_coefficient(),
        frac_inhibitory: inter_or_sens.iter().filter(|&&i| c.sign[i] < 0.0).count() as f64 / inter_or_sens.len().max(1) as f64,
    }
}

/// Largest-modulus Ritz value of a dense row-major `n x n` matrix via Arnoldi.
pub fn dense_spectral_radius(n: usize, a: &[f64], krylov_dim: usize, seed: u64) -> f64 {
    let op = DenseMatrix::new(n, a.to_vec());
    let mut rng = StdRng::seed_from_u64(seed);
    let v0: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() - 0.5).collect();
    let m = krylov_dim.min(n).max(1);
    let res = Arnoldi::new(m, 1e-12).run(&op, &v0).expect("Arnoldi failed");
    krylov_ds::eig::arnoldi_ritz_values(&res)
        .iter()
        .map(|z| (z.re * z.re + z.im * z.im).sqrt())
        .fold(0.0, f64::max)
}

pub fn spectral_stats(c: &Connectome, seed: u64) -> SpectralStats {
    let n = c.n;
    let signed: Vec<f64> = c.signed_count_matrix().iter().map(|&x| x as f64).collect();
    let unsigned: Vec<f64> = c.count_matrix().iter().map(|&x| x as f64).collect();
    let kd = 80;
    let rho_signed = dense_spectral_radius(n, &signed, kd, seed);
    let rho_abs = dense_spectral_radius(n, &unsigned, kd, seed);

    let (g, idx) = c.largest_component_graph();
    let rho_b = nbsc::spectral::estimate_spectral_radius(&g, g.n.min(60), seed);
    let adj_norm = nbsc::spectral::adjacency_operator_norm(&g, g.n.min(60), seed);
    SpectralStats { rho_signed, rho_abs, rho_b, adj_norm, expansivity: adj_norm / rho_b.max(1e-12), lcc_size: idx.len() }
}

pub fn report(c: &Connectome, seed: u64) -> ConnectomeReport {
    ConnectomeReport { wiring: wiring_stats(c), spectral: spectral_stats(c, seed) }
}

// ----------------------------------------------------------------------
// Module recovery from the hypergraph Laplacian eigenmap
// ----------------------------------------------------------------------

/// Plain Lloyd k-means (k-means++-style spread init), returns assignments.
pub fn kmeans(points: &[Vec<f64>], k: usize, iters: usize, seed: u64) -> Vec<usize> {
    let n = points.len();
    let d = points[0].len();
    let mut rng = StdRng::seed_from_u64(seed);
    let dist2 = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>();
    let mut centers: Vec<Vec<f64>> = vec![points[rng.gen_range(0..n)].clone()];
    while centers.len() < k {
        let ds: Vec<f64> = points.iter().map(|p| centers.iter().map(|c| dist2(p, c)).fold(f64::MAX, f64::min)).collect();
        let total: f64 = ds.iter().sum();
        let mut r = rng.gen::<f64>() * total;
        let mut pick = n - 1;
        for (i, &x) in ds.iter().enumerate() {
            if r <= x {
                pick = i;
                break;
            }
            r -= x;
        }
        centers.push(points[pick].clone());
    }
    let mut assign = vec![0usize; n];
    for _ in 0..iters {
        for (i, p) in points.iter().enumerate() {
            assign[i] = (0..k).min_by(|&a, &b| dist2(p, &centers[a]).partial_cmp(&dist2(p, &centers[b])).unwrap()).unwrap();
        }
        let mut sums = vec![vec![0.0; d]; k];
        let mut cnt = vec![0usize; k];
        for (i, p) in points.iter().enumerate() {
            cnt[assign[i]] += 1;
            for j in 0..d {
                sums[assign[i]][j] += p[j];
            }
        }
        for c in 0..k {
            if cnt[c] > 0 {
                for j in 0..d {
                    centers[c][j] = sums[c][j] / cnt[c] as f64;
                }
            }
        }
    }
    assign
}

/// Cluster purity of `assign` against ground-truth `labels`.
pub fn purity(assign: &[usize], labels: &[usize]) -> f64 {
    use std::collections::HashMap;
    let mut table: HashMap<usize, HashMap<usize, usize>> = HashMap::new();
    for (&a, &l) in assign.iter().zip(labels) {
        *table.entry(a).or_default().entry(l).or_insert(0) += 1;
    }
    let good: usize = table.values().map(|row| row.values().copied().max().unwrap_or(0)).sum();
    good as f64 / assign.len() as f64
}

/// Embed the arbor hypergraph with `k` Laplacian-eigenmap coordinates, run
/// k-means with `n_modules` clusters, and report purity against the
/// neuropil labels. Chance level is about `1 / n_modules` to the largest
/// module share; `1.0` is perfect recovery.
pub fn module_recovery_purity(c: &Connectome, k: usize, seed: u64) -> Result<f64> {
    let hg = c.arbor_hypergraph()?;
    let emb = shpinn::embedding::laplacian_eigenmap(&hg, k)?;
    let pts: Vec<Vec<f64>> = (0..emb.n()).map(|i| emb.row(i)).collect();
    let assign = kmeans(&pts, c.n_modules, 50, seed);
    Ok(purity(&assign, &c.module))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connectome::SyntheticConfig;

    #[test]
    fn dense_radius_matches_known_matrix() {
        // diag(3, -4, 1) has spectral radius 4
        let a = vec![3.0, 0.0, 0.0, 0.0, -4.0, 0.0, 0.0, 0.0, 1.0];
        let r = dense_spectral_radius(3, &a, 3, 1);
        assert!((r - 4.0).abs() < 1e-8, "got {r}");
    }

    #[test]
    fn perron_radius_dominates_signed_radius() {
        let c = Connectome::synthetic(&SyntheticConfig { n: 160, seed: 5, ..Default::default() });
        let s = spectral_stats(&c, 0);
        assert!(s.rho_abs >= s.rho_signed - 1e-6, "|W| radius {} must bound signed radius {}", s.rho_abs, s.rho_signed);
        assert!(s.rho_b > 1.0);
        assert!(s.lcc_size == c.n);
    }

    #[test]
    fn spectral_embedding_recovers_modules_far_above_chance() {
        let c = Connectome::synthetic(&SyntheticConfig { n: 512, seed: 2, ..Default::default() });
        let p = module_recovery_purity(&c, c.n_modules, 0).unwrap();
        let z = c.degree_preserving_null(0);
        let pz = module_recovery_purity(&z, z.n_modules, 0).unwrap();
        // chance for 8 equal modules is ~0.125-0.2; the modular wiring must be
        // recoverable and clearly beat the module-destroying null.
        assert!(p > 0.3, "purity {p} should be well above chance");
        assert!(p > pz + 0.15, "real wiring purity {p} should beat degree-preserving null {pz}");
    }

    #[test]
    fn kmeans_separates_two_blobs() {
        let mut pts = Vec::new();
        for i in 0..20 {
            pts.push(vec![i as f64 * 0.01, 0.0]);
            pts.push(vec![10.0 + i as f64 * 0.01, 10.0]);
        }
        let a = kmeans(&pts, 2, 20, 0);
        let labels: Vec<usize> = (0..40).map(|i| i % 2).collect();
        assert_eq!(purity(&a, &labels), 1.0);
    }
}
