//! Structural + spectral report of a connectome, compared against two null
//! models. With no argument it uses the synthetic fly-like generator; pass a
//! FlyWire-style CSV (`pre_id,post_id,syn_count[,nt_type[,neuropil]]`) to
//! analyse real data:
//!
//! `cargo run --release -p flybrain --example connectome_report [-- edges.csv]`

use flybrain::analysis::{module_recovery_purity, report, ConnectomeReport};
use flybrain::connectome::{Connectome, SyntheticConfig};

fn row(name: &str, r: &ConnectomeReport) {
    let (w, s) = (&r.wiring, &r.spectral);
    println!(
        "{name:<22} pairs {:>6}  recip {:.3}  clust {:.3}  maxOut {:>3} maxIn {:>3} | rho(W) {:>6.3}  rho(|W|) {:>6.3}  rho_B {:>6.3}  ||A||/rho_B {:.3}",
        w.connected_pairs, w.reciprocity, w.clustering, w.max_out_degree, w.max_in_degree, s.rho_signed, s.rho_abs, s.rho_b, s.expansivity
    );
}

fn main() {
    let c = match std::env::args().nth(1) {
        Some(path) => {
            let text = std::fs::read_to_string(&path).expect("cannot read csv");
            Connectome::from_edge_csv(&text).expect("bad csv")
        }
        None => Connectome::synthetic(&SyntheticConfig::default()),
    };
    println!("neurons: {}   connected pairs: {}   synapses: {:.0}   modules: {}", c.n, c.n_pairs(), c.total_synapses(), c.n_modules);
    let real = report(&c, 0);
    let null_deg = c.degree_preserving_null(1);
    let null_er = c.density_matched_random(1);
    row("connectome", &real);
    row("degree-preserving null", &report(&null_deg, 0));
    row("density-matched random", &report(&null_er, 0));
    println!("inhibitory fraction among projecting neurons: {:.2}", real.wiring.frac_inhibitory);

    if c.n_modules > 1 {
        let k = c.n_modules;
        let p = |x: &Connectome| module_recovery_purity(x, k, 0).map(|v| format!("{v:.3}")).unwrap_or_else(|e| format!("err {e}"));
        println!("\nneuropil recovery from hypergraph-Laplacian eigenmap (k-means purity, chance ~ {:.2}):", 1.0 / k as f64);
        println!("  connectome {}   degree-preserving null {}   density-matched random {}", p(&c), p(&null_deg), p(&null_er));
    }
}
