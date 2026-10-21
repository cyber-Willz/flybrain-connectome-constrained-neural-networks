//! Connectome data model: a **directed, weighted, sign-annotated** neuron
//! graph, which is what a fly-brain wiring diagram (FlyWire / hemibrain) is
//! and what the undirected, unsigned `nbsc::graph::Graph` is not.
//!
//! Two ways to get one:
//!
//! * [`Connectome::synthetic`] — a *fly-like* generator (neuropil modules,
//!   sensory -> interneuron -> output hierarchy, heavy-tailed degree
//!   propensities, log-normal synapse counts, Dale's-law E/I split,
//!   boosted reciprocity). This is **not** real fly data; it exists so the
//!   whole pipeline can be built, tested and benchmarked offline.
//! * [`Connectome::from_edge_csv`] — loader for a FlyWire-style edge list
//!   (`pre_id,post_id,syn_count[,nt_type[,neuropil]]`), so the same code
//!   runs unchanged on a real (sub)connectome you export from Codex.

use std::collections::{HashMap, HashSet, VecDeque};

use nbsc::graph::Graph;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, LogNormal};
use spectral_hypergraph::hypergraph::{HypergraphBuilder, SpectralHypergraph};

use crate::error::{FlyError, Result};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellClass {
    Sensory,
    Inter,
    Output,
}

/// One directed neuron-to-neuron connection (all synapses between the pair
/// pooled into `count`, as connectome releases report them).
#[derive(Clone, Copy, Debug)]
pub struct Synapse {
    pub pre: usize,
    pub post: usize,
    pub count: f32,
}

#[derive(Clone, Debug)]
pub struct Connectome {
    pub n: usize,
    pub class: Vec<CellClass>,
    /// Neuropil / module id per neuron (`0..n_modules`).
    pub module: Vec<usize>,
    pub n_modules: usize,
    /// Dale's-law sign of each neuron's *outgoing* synapses: `+1` excitatory,
    /// `-1` inhibitory.
    pub sign: Vec<f32>,
    /// Unique directed pairs, no self-loops.
    pub synapses: Vec<Synapse>,
}

/// Parameters of the synthetic fly-like generator.
#[derive(Clone, Debug)]
pub struct SyntheticConfig {
    pub n: usize,
    pub n_modules: usize,
    /// Fraction of neurons in the first two ("input") modules that are sensory.
    pub frac_sensory_in_input_modules: f64,
    /// Fraction of neurons in the last ("output") module that are output neurons.
    pub frac_output_in_output_module: f64,
    /// Fraction of interneurons that are inhibitory (Dale's law).
    pub frac_inhibitory: f64,
    pub p_same_module: f64,
    pub p_adjacent_module: f64,
    pub p_far_module: f64,
    /// Multiplier on connection probability for the feed-forward module direction.
    pub feedforward_boost: f64,
    /// Probability that an interneuron->interneuron edge gets a reciprocal partner.
    pub reciprocity: f64,
    /// `sigma` of the log-normal degree propensity (heavy-tailed hubs).
    pub propensity_sigma: f64,
    pub seed: u64,
}

impl Default for SyntheticConfig {
    fn default() -> Self {
        Self {
            n: 512,
            n_modules: 8,
            frac_sensory_in_input_modules: 0.6,
            frac_output_in_output_module: 0.5,
            frac_inhibitory: 0.30,
            p_same_module: 0.06,
            p_adjacent_module: 0.025,
            p_far_module: 0.005,
            feedforward_boost: 2.5,
            reciprocity: 0.30,
            propensity_sigma: 0.6,
            seed: 0,
        }
    }
}

impl Connectome {
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    pub fn synthetic(cfg: &SyntheticConfig) -> Self {
        assert!(cfg.n >= 2 * cfg.n_modules && cfg.n_modules >= 3, "need >= 3 modules and >= 2 neurons/module");
        let mut rng = StdRng::seed_from_u64(cfg.seed);
        let n = cfg.n;
        let nm = cfg.n_modules;

        // Modules: contiguous blocks of near-equal size.
        let module: Vec<usize> = (0..n).map(|i| (i * nm) / n).collect();

        // Cell classes.
        let mut class = vec![CellClass::Inter; n];
        for i in 0..n {
            let m = module[i];
            if m < 2 && rng.gen::<f64>() < cfg.frac_sensory_in_input_modules {
                class[i] = CellClass::Sensory;
            } else if m == nm - 1 && rng.gen::<f64>() < cfg.frac_output_in_output_module {
                class[i] = CellClass::Output;
            }
        }

        // Dale's-law sign per presynaptic neuron.
        let sign: Vec<f32> = (0..n)
            .map(|i| match class[i] {
                CellClass::Inter if rng.gen::<f64>() < cfg.frac_inhibitory => -1.0,
                _ => 1.0,
            })
            .collect();

        // Degree-corrected propensities (heavy tails -> hubs).
        let ln = LogNormal::new(0.0, cfg.propensity_sigma).unwrap();
        let out_prop: Vec<f64> = (0..n).map(|_| ln.sample(&mut rng)).collect();
        let in_prop: Vec<f64> = (0..n).map(|_| ln.sample(&mut rng)).collect();

        let mut set: HashSet<(usize, usize)> = HashSet::new();
        let allowed = |a: usize, b: usize, class: &[CellClass]| -> bool {
            // sensory neurons take no recurrent input; output neurons send none.
            class[b] != CellClass::Sensory && class[a] != CellClass::Output
        };
        for a in 0..n {
            for b in 0..n {
                if a == b || !allowed(a, b, &class) {
                    continue;
                }
                let (ma, mb) = (module[a], module[b]);
                let mut p = if ma == mb {
                    cfg.p_same_module
                } else if ma.abs_diff(mb) == 1 {
                    cfg.p_adjacent_module
                } else {
                    cfg.p_far_module
                };
                if mb > ma {
                    p *= cfg.feedforward_boost;
                }
                // Sensory neurons project forward only; they synapse mostly onto the next modules.
                let p = (p * out_prop[a] * in_prop[b]).min(0.95);
                if rng.gen::<f64>() < p {
                    set.insert((a, b));
                }
            }
        }
        // Reciprocity among interneurons.
        let mut snapshot: Vec<(usize, usize)> = set.iter().copied().collect();
        snapshot.sort_unstable(); // HashSet order is randomised per process; sort for reproducibility
        for (a, b) in snapshot {
            if class[a] == CellClass::Inter && class[b] == CellClass::Inter && rng.gen::<f64>() < cfg.reciprocity {
                set.insert((b, a));
            }
        }
        // Make sure every neuron participates and the graph is weakly connected.
        let mut edges: Vec<(usize, usize)> = set.into_iter().collect();
        edges.sort_unstable();
        let mut es: HashSet<(usize, usize)> = edges.iter().copied().collect();
        // isolated neurons: attach to a random same-module (or neighbouring) interneuron
        let mut deg = vec![0usize; n];
        for &(a, b) in &es {
            deg[a] += 1;
            deg[b] += 1;
        }
        let inters: Vec<usize> = (0..n).filter(|&i| class[i] == CellClass::Inter).collect();
        for i in 0..n {
            if deg[i] == 0 {
                let partner = loop {
                    let j = inters[rng.gen_range(0..inters.len())];
                    if j != i && module[j].abs_diff(module[i]) <= 1 {
                        break j;
                    }
                };
                let e = match class[i] {
                    CellClass::Sensory => (i, partner),
                    CellClass::Output => (partner, i),
                    CellClass::Inter => (i, partner),
                };
                es.insert(e);
            }
        }
        // weak connectivity: bridge components to the giant one
        loop {
            let comps = weak_components(n, &es);
            if comps.len() <= 1 {
                break;
            }
            let gi = (0..comps.len()).max_by_key(|&i| comps[i].len()).unwrap();
            let mut giant_inters: Vec<usize> = comps[gi].iter().copied().filter(|&i| class[i] == CellClass::Inter).collect();
            giant_inters.sort_unstable();
            for (ci, c) in comps.iter().enumerate().filter(|(ci, _)| *ci != gi) {
                let _ = ci;
                let u = c[0];
                let v = giant_inters[rng.gen_range(0..giant_inters.len())];
                let e = match class[u] {
                    CellClass::Output => (v, u),
                    _ => (u, v),
                };
                es.insert(e);
            }
        }

        // Synapse counts: 1 + heavy-tailed.
        let cnt = LogNormal::new(0.9, 0.9).unwrap();
        let mut edges: Vec<(usize, usize)> = es.into_iter().collect();
        edges.sort_unstable();
        let synapses = edges
            .into_iter()
            .map(|(pre, post)| Synapse { pre, post, count: (1.0f64 + cnt.sample(&mut rng)).min(80.0).floor() as f32 })
            .collect();

        Connectome { n, class, module, n_modules: nm, sign, synapses }
    }

    /// Loader for a FlyWire/Codex-style connection table.
    ///
    /// Columns: `pre_id,post_id,syn_count[,nt_type[,neuropil]]` (header row
    /// optional; `#` comment lines skipped). Neuron ids are arbitrary strings
    /// or integers and are re-indexed densely. Conventions (adjust to taste
    /// for your release):
    /// * a neuron is **inhibitory** if the majority of its outgoing rows have
    ///   `nt_type` in {`GABA`, `GLUT`} (glutamate is largely inhibitory in
    ///   the fly central brain); everything else is treated as excitatory;
    /// * class is inferred from topology: no inputs in the table -> sensory,
    ///   no outputs -> output, otherwise interneuron;
    /// * module = majority `neuropil` label of the neuron's outgoing rows
    ///   (all `0` if the column is absent).
    pub fn from_edge_csv(text: &str) -> Result<Self> {
        let mut ids: HashMap<String, usize> = HashMap::new();
        let mut rows: Vec<(usize, usize, f32)> = Vec::new();
        let mut inhib_votes: HashMap<usize, (u32, u32)> = HashMap::new();
        let mut np_votes: HashMap<usize, HashMap<String, u32>> = HashMap::new();
        let intern = |s: &str, ids: &mut HashMap<String, usize>| -> usize {
            let next = ids.len();
            *ids.entry(s.to_string()).or_insert(next)
        };
        for (ln, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if f.len() < 3 {
                return Err(FlyError::Parse(format!("line {}: expected >= 3 columns", ln + 1)));
            }
            let count: f32 = match f[2].parse() {
                Ok(c) => c,
                Err(_) if ln == 0 => continue, // header
                Err(_) => return Err(FlyError::Parse(format!("line {}: bad syn_count '{}'", ln + 1, f[2]))),
            };
            let a = intern(f[0], &mut ids);
            let b = intern(f[1], &mut ids);
            if a == b || count <= 0.0 {
                continue;
            }
            rows.push((a, b, count));
            if let Some(nt) = f.get(3) {
                let inh = matches!(nt.to_ascii_uppercase().as_str(), "GABA" | "GLUT");
                let e = inhib_votes.entry(a).or_insert((0, 0));
                if inh {
                    e.1 += 1
                } else {
                    e.0 += 1
                }
            }
            if let Some(np) = f.get(4) {
                *np_votes.entry(a).or_default().entry((*np).to_string()).or_insert(0) += 1;
            }
        }
        if rows.is_empty() {
            return Err(FlyError::Parse("no synapse rows".into()));
        }
        let n = ids.len();
        // pool duplicate pairs
        let mut pooled: HashMap<(usize, usize), f32> = HashMap::new();
        for (a, b, c) in rows {
            *pooled.entry((a, b)).or_insert(0.0) += c;
        }
        let mut synapses: Vec<Synapse> = pooled.into_iter().map(|((pre, post), count)| Synapse { pre, post, count }).collect();
        synapses.sort_by_key(|s| (s.pre, s.post));
        let mut indeg = vec![0usize; n];
        let mut outdeg = vec![0usize; n];
        for s in &synapses {
            outdeg[s.pre] += 1;
            indeg[s.post] += 1;
        }
        let class = (0..n)
            .map(|i| {
                if indeg[i] == 0 {
                    CellClass::Sensory
                } else if outdeg[i] == 0 {
                    CellClass::Output
                } else {
                    CellClass::Inter
                }
            })
            .collect();
        let sign = (0..n)
            .map(|i| match inhib_votes.get(&i) {
                Some(&(e, inh)) if inh > e => -1.0,
                _ => 1.0,
            })
            .collect();
        let mut np_ids: HashMap<String, usize> = HashMap::new();
        let module: Vec<usize> = (0..n)
            .map(|i| match np_votes.get(&i) {
                Some(v) => {
                    let best = v.iter().max_by_key(|(k, c)| (**c, std::cmp::Reverse((*k).clone()))).map(|(k, _)| k.clone()).unwrap();
                    let next = np_ids.len();
                    *np_ids.entry(best).or_insert(next)
                }
                None => 0,
            })
            .collect();
        let n_modules = np_ids.len().max(1);
        Ok(Connectome { n, class, module, n_modules, sign, synapses })
    }

    // ------------------------------------------------------------------
    // Views
    // ------------------------------------------------------------------

    pub fn n_pairs(&self) -> usize {
        self.synapses.len()
    }

    pub fn total_synapses(&self) -> f64 {
        self.synapses.iter().map(|s| s.count as f64).sum()
    }

    pub fn indices_of(&self, c: CellClass) -> Vec<usize> {
        (0..self.n).filter(|&i| self.class[i] == c).collect()
    }

    /// Dense `[post, pre]` matrix of raw synapse counts (row-major `n*n`).
    pub fn count_matrix(&self) -> Vec<f32> {
        let mut m = vec![0.0f32; self.n * self.n];
        for s in &self.synapses {
            m[s.post * self.n + s.pre] = s.count;
        }
        m
    }

    /// Dense `[post, pre]` 0/1 wiring mask.
    pub fn mask_matrix(&self) -> Vec<f32> {
        let mut m = vec![0.0f32; self.n * self.n];
        for s in &self.synapses {
            m[s.post * self.n + s.pre] = 1.0;
        }
        m
    }

    /// Dense `[post, pre]` signed count matrix: `sign[pre] * count` (Dale's law).
    pub fn signed_count_matrix(&self) -> Vec<f32> {
        let mut m = self.count_matrix();
        for post in 0..self.n {
            for pre in 0..self.n {
                m[post * self.n + pre] *= self.sign[pre];
            }
        }
        m
    }

    /// Undirected simple graph (an edge wherever either direction exists):
    /// the object `nbsc`'s Hashimoto/Ihara machinery operates on.
    pub fn symmetrized_graph(&self) -> Graph {
        let mut g = Graph::new(self.n);
        for s in &self.synapses {
            g.add_edge(s.pre, s.post);
        }
        g
    }

    /// Largest weakly-connected component as a re-indexed `Graph`, plus the
    /// original index of each of its nodes.
    pub fn largest_component_graph(&self) -> (Graph, Vec<usize>) {
        let g = self.symmetrized_graph();
        let es: HashSet<(usize, usize)> = g.edges.iter().copied().collect();
        let comps = weak_components(self.n, &es);
        let mut giant = comps.into_iter().max_by_key(|c| c.len()).unwrap_or_default();
        giant.sort_unstable();
        let pos: HashMap<usize, usize> = giant.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        let mut sub = Graph::new(giant.len());
        for &(u, v) in &g.edges {
            if let (Some(&a), Some(&b)) = (pos.get(&u), pos.get(&v)) {
                sub.add_edge(a, b);
            }
        }
        (sub, giant)
    }

    /// **Axonal-arbor hypergraph**: one hyperedge per presynaptic neuron,
    /// `{neuron} ∪ {its postsynaptic targets}`. This is the natural
    /// higher-order object of a connectome (one neuron fans out to many
    /// targets at once), and it is what `shpinn`'s true hypergraph Laplacian
    /// is applied to — not a pairwise clique approximation.
    pub fn arbor_hypergraph(&self) -> Result<SpectralHypergraph> {
        let mut b = HypergraphBuilder::with_capacity(self.n, self.n, 16);
        let ids: Vec<_> = (0..self.n).map(|i| b.add_vertex(format!("n{i}"))).collect::<std::result::Result<_, _>>()?;
        let mut targets: Vec<Vec<usize>> = vec![Vec::new(); self.n];
        for s in &self.synapses {
            targets[s.pre].push(s.post);
        }
        for (i, t) in targets.iter().enumerate() {
            if t.is_empty() {
                continue;
            }
            let mut members = vec![ids[i]];
            members.extend(t.iter().map(|&j| ids[j]));
            b.add_hyperedge(&members, 1.0)?;
        }
        Ok(b.build()?)
    }

    // ------------------------------------------------------------------
    // Null models ("what does the real wiring buy you?")
    // ------------------------------------------------------------------

    /// Directed degree-preserving rewiring (Maslov–Sneppen edge swaps):
    /// keeps every neuron's in- and out-degree and the multiset of synapse
    /// counts, destroys everything else (modules, motifs, reciprocity).
    pub fn degree_preserving_null(&self, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut edges: Vec<Synapse> = self.synapses.clone();
        let mut set: HashSet<(usize, usize)> = edges.iter().map(|s| (s.pre, s.post)).collect();
        let m = edges.len();
        for _ in 0..(20 * m) {
            let i = rng.gen_range(0..m);
            let j = rng.gen_range(0..m);
            if i == j {
                continue;
            }
            let (a, b) = (edges[i].pre, edges[i].post);
            let (c, d) = (edges[j].pre, edges[j].post);
            // (a->b),(c->d)  =>  (a->d),(c->b)
            if a == d || c == b || set.contains(&(a, d)) || set.contains(&(c, b)) {
                continue;
            }
            set.remove(&(a, b));
            set.remove(&(c, d));
            set.insert((a, d));
            set.insert((c, b));
            edges[i].post = d;
            edges[j].post = b;
        }
        Connectome { synapses: edges, ..self.clone() }
    }

    /// Directed Erdős–Rényi wiring with the same number of connected pairs
    /// and the same multiset of counts, respecting the same cell-class
    /// constraints (no input to sensory, no output from output neurons).
    pub fn density_matched_random(&self, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut counts: Vec<f32> = self.synapses.iter().map(|s| s.count).collect();
        counts.shuffle(&mut rng);
        let mut set: HashSet<(usize, usize)> = HashSet::new();
        let mut synapses = Vec::with_capacity(counts.len());
        for c in counts {
            loop {
                let a = rng.gen_range(0..self.n);
                let b = rng.gen_range(0..self.n);
                if a == b
                    || self.class[b] == CellClass::Sensory
                    || self.class[a] == CellClass::Output
                    || !set.insert((a, b))
                {
                    continue;
                }
                synapses.push(Synapse { pre: a, post: b, count: c });
                break;
            }
        }
        Connectome { synapses, ..self.clone() }
    }

    // ------------------------------------------------------------------
    // Simple statistics
    // ------------------------------------------------------------------

    pub fn reciprocity(&self) -> f64 {
        let set: HashSet<(usize, usize)> = self.synapses.iter().map(|s| (s.pre, s.post)).collect();
        let recip = set.iter().filter(|&&(a, b)| set.contains(&(b, a))).count();
        recip as f64 / set.len().max(1) as f64
    }

    pub fn out_degrees(&self) -> Vec<usize> {
        let mut d = vec![0; self.n];
        for s in &self.synapses {
            d[s.pre] += 1;
        }
        d
    }

    pub fn in_degrees(&self) -> Vec<usize> {
        let mut d = vec![0; self.n];
        for s in &self.synapses {
            d[s.post] += 1;
        }
        d
    }

    /// Average local clustering coefficient of the symmetrized graph.
    pub fn clustering_coefficient(&self) -> f64 {
        let g = self.symmetrized_graph();
        let sets: Vec<HashSet<usize>> = g.neighbors.iter().map(|v| v.iter().copied().collect()).collect();
        let mut total = 0.0;
        let mut cnt = 0usize;
        for v in 0..g.n {
            let d = g.neighbors[v].len();
            if d < 2 {
                continue;
            }
            let mut tri = 0usize;
            for (i, &a) in g.neighbors[v].iter().enumerate() {
                for &b in &g.neighbors[v][i + 1..] {
                    if sets[a].contains(&b) {
                        tri += 1;
                    }
                }
            }
            total += 2.0 * tri as f64 / (d * (d - 1)) as f64;
            cnt += 1;
        }
        total / cnt.max(1) as f64
    }
}

fn weak_components(n: usize, edges: &HashSet<(usize, usize)>) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); n];
    for &(a, b) in edges {
        adj[a].push(b);
        adj[b].push(a);
    }
    let mut seen = vec![false; n];
    let mut comps = Vec::new();
    for s in 0..n {
        if seen[s] {
            continue;
        }
        let mut comp = Vec::new();
        let mut q = VecDeque::from([s]);
        seen[s] = true;
        while let Some(u) = q.pop_front() {
            comp.push(u);
            for &v in &adj[u] {
                if !seen[v] {
                    seen[v] = true;
                    q.push_back(v);
                }
            }
        }
        comps.push(comp);
    }
    comps
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small() -> Connectome {
        Connectome::synthetic(&SyntheticConfig { n: 160, n_modules: 8, seed: 3, ..Default::default() })
    }

    #[test]
    fn synthetic_is_connected_and_respects_class_rules() {
        let c = small();
        let (g, idx) = c.largest_component_graph();
        assert_eq!(idx.len(), c.n, "graph must be weakly connected");
        assert!(g.m() > c.n);
        for s in &c.synapses {
            assert_ne!(s.pre, s.post);
            assert_ne!(c.class[s.post], CellClass::Sensory);
            assert_ne!(c.class[s.pre], CellClass::Output);
            assert!(s.count >= 1.0);
        }
        assert!(!c.indices_of(CellClass::Sensory).is_empty());
        assert!(!c.indices_of(CellClass::Output).is_empty());
    }

    #[test]
    fn degree_preserving_null_keeps_degrees_but_not_edges() {
        let c = small();
        let z = c.degree_preserving_null(1);
        assert_eq!(c.out_degrees(), z.out_degrees());
        assert_eq!(c.in_degrees(), z.in_degrees());
        let a: HashSet<_> = c.synapses.iter().map(|s| (s.pre, s.post)).collect();
        let b: HashSet<_> = z.synapses.iter().map(|s| (s.pre, s.post)).collect();
        assert_eq!(a.len(), b.len());
        assert!(a.intersection(&b).count() < a.len() * 9 / 10, "rewiring should change most edges");
        assert!(z.reciprocity() < c.reciprocity());
    }

    #[test]
    fn density_matched_random_has_same_edge_count() {
        let c = small();
        let r = c.density_matched_random(2);
        assert_eq!(c.n_pairs(), r.n_pairs());
        let mut a: Vec<f32> = c.synapses.iter().map(|s| s.count).collect();
        let mut b: Vec<f32> = r.synapses.iter().map(|s| s.count).collect();
        a.sort_by(|x, y| x.partial_cmp(y).unwrap());
        b.sort_by(|x, y| x.partial_cmp(y).unwrap());
        assert_eq!(a, b);
    }

    #[test]
    fn csv_roundtrip_parses_signs_classes_and_pools_duplicates() {
        let csv = "pre_id,post_id,syn_count,nt_type,neuropil\n\
                   a,b,3,ACH,MB\n a,b,2,ACH,MB\n b,c,4,GABA,MB\n c,d,1,GLUT,LH\n b,d,5,GABA,MB\n";
        let c = Connectome::from_edge_csv(csv).unwrap();
        assert_eq!(c.n, 4);
        assert_eq!(c.n_pairs(), 4);
        assert_eq!(c.synapses.iter().find(|s| s.pre == 0 && s.post == 1).unwrap().count, 5.0);
        assert_eq!(c.class[0], CellClass::Sensory); // 'a' has no inputs
        assert_eq!(c.class[3], CellClass::Output); // 'd' has no outputs
        assert_eq!(c.sign[1], -1.0); // all of b's outgoing rows are GABA
        assert_eq!(c.sign[0], 1.0); // a is cholinergic
    }

    #[test]
    fn arbor_hypergraph_has_one_edge_per_projecting_neuron() {
        let c = small();
        let hg = c.arbor_hypergraph().unwrap();
        let projecting = c.out_degrees().iter().filter(|&&d| d > 0).count();
        assert_eq!(hg.num_hyperedges(), projecting);
        assert_eq!(hg.num_vertices(), c.n);
    }
}

#[cfg(test)]
mod determinism_tests {
    use super::*;

    #[test]
    fn synthetic_generation_is_reproducible_across_calls() {
        let cfg = SyntheticConfig { n: 128, n_modules: 8, seed: 5, ..Default::default() };
        let a = Connectome::synthetic(&cfg);
        for _ in 0..4 {
            let b = Connectome::synthetic(&cfg);
            assert_eq!(a.synapses.len(), b.synapses.len());
            assert!(a.synapses.iter().zip(&b.synapses).all(|(x, y)| x.pre == y.pre && x.post == y.post && x.count == y.count));
            assert_eq!(a.sign, b.sign);
        }
    }
}
