//! End-to-end experiment: does fixing the network's wiring to the connectome
//! help, and do the NBSC energy diagnostics / SHPINN surrogate add anything?
//!
//! Experiment 1 (teacher-student, full recording)
//!   A hidden "fly" = the connectome's wiring and signs with log-normally
//!   perturbed synapse magnitudes. A student network is fitted to its
//!   activity under random sensory stimulation and scored on held-out
//!   stimuli:
//!     * connectome-constrained (true wiring)            [FlyBrainNet]
//!     * connectome-constrained + Dirichlet-energy loss  [nbsc energy regulariser]
//!     * density-matched random wiring (wrong edges)
//!     * fully dense, unconstrained
//! Experiment 2 (partial recording)
//!   Same, but the loss only sees 30% of neurons; error is reported on the
//!   neurons that were never seen.
//! Experiment 3 (SHPINN surrogate)
//!   Continuous-time surrogate imputing unrecorded neurons, with/without
//!   the connectome's rate equation as the physics residual.
//!
//! `cargo run --release -p flybrain --example fly_pipeline [-- --quick]`

use std::time::Instant;

use burn::tensor::backend::Backend;
use flybrain::analysis::wiring_stats;
use flybrain::connectome::{Connectome, SyntheticConfig};
use flybrain::dynamics::*;
use flybrain::observer::{ObserverConfig, ObserverKind};
use flybrain::surrogate::{PhysicsMode, SurrogateConfig, SurrogateProblem};

fn mean_sd(v: &[f64]) -> (f64, f64) {
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let var = v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / v.len().max(1) as f64;
    (m, var.sqrt())
}

fn fmt(v: &[f64]) -> String {
    let (m, s) = mean_sd(v);
    format!("{m:.4} ± {s:.4}")
}

#[derive(Default)]
struct Acc {
    train: Vec<f64>,
    test: Vec<f64>,
    unobs: Vec<f64>,
    corr: Vec<f64>,
    energy: Vec<Vec<f32>>,
    secs: Vec<f64>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let quick = args.iter().any(|a| a == "--quick");
    // `--only=1,3` runs just those experiments.
    let only: Option<Vec<u32>> = args.iter().find_map(|a| a.strip_prefix("--only=").map(|v| v.split(',').filter_map(|x| x.parse().ok()).collect()));
    let want = |k: u32| only.as_ref().map(|o| o.contains(&k)).unwrap_or(true);
    let seeds_override: Option<u64> = args.iter().find_map(|a| a.strip_prefix("--seeds=").and_then(|v| v.parse().ok()));
    let no_pinn = args.iter().any(|a| a == "--no-pinn");
    let (n, seeds, epochs, sur_epochs) = if quick { (192usize, 1u64, 80usize, 300usize) } else { (512, 3, 300, 1500) };
    let seeds = seeds_override.unwrap_or(seeds);
    let device = <Be as Backend>::Device::default();
    let net_cfg = NetConfig::default();

    println!("# flybrain pipeline  (n = {n}, seeds = {seeds}, epochs = {epochs})\n");
    let c = Connectome::synthetic(&SyntheticConfig { n, seed: 0, ..Default::default() });
    let w = wiring_stats(&c);
    println!(
        "connectome: {} neurons, {} connected pairs ({:.2}% density), {:.0} synapses, reciprocity {:.2}, {:.0}% inhibitory, max out/in degree {}/{}\n",
        w.n, w.connected_pairs, 100.0 * w.density, w.total_synapses, w.reciprocity, 100.0 * w.frac_inhibitory, w.max_out_degree, w.max_in_degree
    );
    let graph = c.symmetrized_graph();
    let mask = c.mask_matrix();
    let n_params_constrained = c.n_pairs();
    let n_params_dense = n * (n - 1);

    let lambdas = [0.0, 0.003];
    let names = [
        "connectome-constrained".to_string(),
        format!("connectome-constrained + energy reg (λ={})", lambdas[1]),
        "random wiring, same density".to_string(),
        "dense unconstrained".to_string(),
    ];
    let mut full: Vec<Acc> = (0..4).map(|_| Acc::default()).collect();
    let mut partial: Vec<Acc> = (0..3).map(|_| Acc::default()).collect();
    let mut teacher_energy: Vec<Vec<f32>> = Vec::new();

    for seed in 0..seeds {
        let teacher = make_teacher(&c, &net_cfg, 0.5, 1000 + seed);
        let train = make_dataset(&c, &teacher, &net_cfg, 16, 20, 1.0, 2000 + seed);
        let test = make_dataset(&c, &teacher, &net_cfg, 16, 20, 1.0, 3000 + seed);
        let random_c = c.density_matched_random(4000 + seed);
        let random_mask = random_c.mask_matrix();

        // ---- Experiment 1: full recording
        for cond in 0..4 {
            if !want(1) {
                break;
            }
            let t0 = Instant::now();
            let (net, fc, m, tt): (FlyBrainNet<Be>, FitConfig, &[f32], Option<&[f32]>) = match cond {
                0 | 1 => (
                    FlyBrainNet::from_connectome(&c, &net_cfg, &device),
                    FitConfig { epochs, lr: 0.03, energy_lambda: lambdas[cond], ..Default::default() },
                    &mask,
                    Some(&teacher.theta),
                ),
                2 => (FlyBrainNet::from_connectome(&random_c, &net_cfg, &device), FitConfig { epochs, lr: 0.03, ..Default::default() }, &random_mask, None),
                _ => (FlyBrainNet::dense_unconstrained(n, &net_cfg, 5000 + seed, &device), FitConfig { epochs, lr: 0.005, ..Default::default() }, &mask, None),
            };
            let (_net, r) = fit(net, &train, &test, &graph, m, tt, &fc);
            let a = &mut full[cond];
            a.train.push(r.train.rel_mse);
            a.test.push(r.test.rel_mse);
            if let Some(cr) = r.theta_corr {
                a.corr.push(cr);
            }
            a.energy.push(r.test.energy_profile.clone());
            a.secs.push(t0.elapsed().as_secs_f64());
            if cond == 0 {
                teacher_energy.push(r.teacher_energy_profile.clone());
            }
            println!("[seed {seed}] exp1 {:<48} train {:.4}  test {:.4}  ({:.0}s)", names[cond], r.train.rel_mse, r.test.rel_mse, t0.elapsed().as_secs_f64());
        }

        // ---- Experiment 2: 30% of neurons recorded
        let obs = random_obs_mask(n, 0.3, 6000 + seed);
        for cond in 0..3 {
            if !want(2) {
                break;
            }
            let t0 = Instant::now();
            let (net, lr, m, tt): (FlyBrainNet<Be>, f64, &[f32], Option<&[f32]>) = match cond {
                0 => (FlyBrainNet::from_connectome(&c, &net_cfg, &device), 0.03, &mask, Some(&teacher.theta)),
                1 => (FlyBrainNet::from_connectome(&random_c, &net_cfg, &device), 0.03, &random_mask, None),
                _ => (FlyBrainNet::dense_unconstrained(n, &net_cfg, 5000 + seed, &device), 0.005, &mask, None),
            };
            let fc = FitConfig { epochs, lr, obs_mask: Some(obs.clone()), ..Default::default() };
            let (_net, r) = fit(net, &train, &test, &graph, m, tt, &fc);
            let a = &mut partial[cond];
            a.test.push(r.test.rel_mse);
            a.unobs.push(r.test.rel_mse_unobserved);
            if let Some(cr) = r.theta_corr {
                a.corr.push(cr);
            }
            println!("[seed {seed}] exp2 {:<48} test(all) {:.4}  test(unrecorded) {:.4}  ({:.0}s)", ["connectome-constrained", "random wiring", "dense unconstrained"][cond], r.test.rel_mse, r.test.rel_mse_unobserved, t0.elapsed().as_secs_f64());
        }
    }

    if want(1) {
    println!("\n## Experiment 1 — teacher-student, all neurons recorded (relative MSE; lower is better; mean ± sd over {seeds} teachers)\n");
    println!("| student | learned params | train | held-out stimuli | θ recovery (r) |");
    println!("|---|---:|---:|---:|---:|");
    for (i, a) in full.iter().enumerate() {
        let p = if i == 3 { n_params_dense } else { n_params_constrained };
        let corr = if a.corr.is_empty() { "—".into() } else { fmt(&a.corr) };
        println!("| {} | {} | {} | {} | {} |", names[i], p, fmt(&a.train), fmt(&a.test), corr);
    }

    }
    if want(2) {
    println!("\n## Experiment 2 — 30% of neurons recorded; error on the 70% never seen\n");
    println!("| student | held-out, all neurons | held-out, unrecorded neurons | θ recovery (r) |");
    println!("|---|---:|---:|---:|");
    for (i, nm) in ["connectome-constrained", "random wiring, same density", "dense unconstrained"].iter().enumerate() {
        let a = &partial[i];
        let corr = if a.corr.is_empty() { "—".into() } else { fmt(&a.corr) };
        println!("| {} | {} | {} | {} |", nm, fmt(&a.test), fmt(&a.unobs), corr);
    }

    }

    if want(1) {
    println!("\n## Dirichlet energy per unrolled step (held-out stimuli, seed-mean; normalised as in nbsc::burn_layer::dirichlet_energy)\n");
    let steps = [1usize, 5, 10, 15, 20];
    print!("| source |");
    for s in steps {
        print!(" step {s} |");
    }
    println!("\n|---|---:|---:|---:|---:|---:|");
    let avg = |v: &Vec<Vec<f32>>, s: usize| v.iter().map(|e| e[s - 1] as f64).sum::<f64>() / v.len() as f64;
    print!("| teacher (target) |");
    for s in steps {
        print!(" {:.3} |", avg(&teacher_energy, s));
    }
    println!();
    for i in [0usize, 1, 2, 3] {
        print!("| {} |", names[i]);
        for s in steps {
            let e: Vec<Vec<f32>> = full[i].energy.clone();
            print!(" {:.3} |", avg(&e, s));
        }
        println!();
    }

    }
    if !want(3) && !want(4) {
        return;
    }
    // ---- Experiment 3: SHPINN surrogate (own, smaller connectome: the surrogate is the most expensive part)
    let n3 = if quick { 96 } else { 256 };
    let c3 = Connectome::synthetic(&SyntheticConfig { n: n3, seed: 7, ..Default::default() });
    let modes: Vec<PhysicsMode> = if no_pinn { vec![] } else { vec![PhysicsMode::DataOnly, PhysicsMode::NominalW, PhysicsMode::OracleW] };
    let obs_kinds = [ObserverKind::Nominal, ObserverKind::Learned, ObserverKind::Oracle];
    let ocfg = ObserverConfig::default();
    let regimes: [(&str, f64, usize); 2] = [
        ("3a: imputation — 30% of neurons recorded, a sample every 0.1", 0.3, 20),
        ("3b: temporal super-resolution — all neurons recorded, a sample every 0.5", 1.0, 100),
    ];
    for (title, obs_frac, stride) in regimes.into_iter().filter(|_| want(3)) {
        println!("\n## Experiment {title} (n = {n3}; relative L2 error at held-out times; mean ± sd over {seeds} problems)\n");
        let mut sur: Vec<(Vec<f64>, Vec<f64>)> = vec![(vec![], vec![]); modes.len()];
        let mut obsv: Vec<(Vec<f64>, Vec<f64>)> = vec![(vec![], vec![]); obs_kinds.len()];
        let (mut interp, mut cubic) = (Vec::new(), Vec::new());
        for seed in 0..seeds {
            let teacher = make_teacher(&c3, &net_cfg, 0.5, 1000 + seed);
            let scfg = SurrogateConfig {
                epochs: sur_epochs,
                k_embed: if quick { 12 } else { 24 },
                hidden: vec![64, 64, 64],
                data_times_per_epoch: 4,
                colloc_times_per_epoch: 4,
                obs_frac,
                data_stride: stride,
                seed,
                ..Default::default()
            };
            let problem = SurrogateProblem::new(&c3, &teacher, &net_cfg, scfg, 1.0).expect("surrogate setup");
            interp.push(problem.linear_interp_baseline_rel_err());
            cubic.push(problem.cubic_interp_baseline_rel_err());
            for (i, m) in modes.iter().enumerate() {
                let t0 = Instant::now();
                let r = problem.run(*m);
                println!("[seed {seed}] exp{} {:<28} unrecorded {:.4}  recorded {:.4}  loss {:.5} ({:.0}s)", &title[..2], m.name(), r.rel_err_unobserved, r.rel_err_observed, r.final_loss, t0.elapsed().as_secs_f64());
                sur[i].0.push(r.rel_err_unobserved);
                sur[i].1.push(r.rel_err_observed);
            }
            if obs_frac < 1.0 {
                for (i, k) in obs_kinds.iter().enumerate() {
                    let t0 = Instant::now();
                    let r = problem.run_observer(*k, &ocfg);
                    println!("[seed {seed}] exp{} {:<28} unrecorded {:.4}  recorded {:.4}  loss {:.5} ({:.0}s)", &title[..2], k.name(), r.rel_err_unobserved, r.rel_err_observed, r.final_loss, t0.elapsed().as_secs_f64());
                    obsv[i].0.push(r.rel_err_unobserved);
                    obsv[i].1.push(r.rel_err_observed);
                }
            }
        }
        println!("| variant | unrecorded neurons | recorded neurons |");
        println!("|---|---:|---:|");
        for (i, m) in modes.iter().enumerate() {
            let un = if sur[i].0.iter().any(|x| x.is_nan()) { "—".to_string() } else { fmt(&sur[i].0) };
            println!("| SHPINN, {} | {} | {} |", m.name(), un, fmt(&sur[i].1));
        }
        if obs_frac < 1.0 {
            for (i, k) in obs_kinds.iter().enumerate() {
                println!("| {} | {} | {} |", k.name(), fmt(&obsv[i].0), fmt(&obsv[i].1));
            }
        }
        println!("| linear interpolation (no network) | — | {} |", fmt(&interp));
        println!("| cubic interpolation (no network) | — | {} |", fmt(&cubic));
        println!("| predict zero | 1.0000 | 1.0000 |");
    }

    if !want(4) {
        return;
    }
    let ocfg = ObserverConfig { epochs: 100, ..Default::default() };
    // ---- Experiment 3c: when does the physics pay off? sweep noise x sampling interval (observer only; no SHPINN)
    println!("\n## Experiment 3c: observer vs interpolation as recordings get noisier / sparser (30% recorded, n = {n3}, mean over {seeds} problems; unrecorded-neuron error | recorded-neuron error)\n");
    println!("| sample every | noise sd | observer nominal | observer learned | observer oracle | cubic interp (recorded) | linear interp (recorded) |");
    println!("|---:|---:|---:|---:|---:|---:|---:|");
    for stride in [20usize, 100] {
        for noise in [0.0f64, 0.05, 0.15] {
            let mut cols: Vec<Vec<f64>> = vec![vec![]; 7];
            for seed in 0..seeds {
                let teacher = make_teacher(&c3, &net_cfg, 0.5, 1000 + seed);
                let scfg = SurrogateConfig { epochs: 1, k_embed: 12, hidden: vec![8], obs_frac: 0.3, data_stride: stride, obs_noise: noise, seed, ..Default::default() };
                let problem = SurrogateProblem::new(&c3, &teacher, &net_cfg, scfg, 1.0).expect("surrogate setup");
                for (i, k) in obs_kinds.iter().enumerate() {
                    let r = problem.run_observer(*k, &ocfg);
                    cols[i].push(r.rel_err_unobserved);
                    if i == 0 {
                        cols[3].push(r.rel_err_observed);
                    }
                }
                cols[4].push(problem.linear_interp_baseline_rel_err());
            }
            let m = |v: &Vec<f64>| mean_sd(v).0;
            println!("| {:.2} | {:.2} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |", stride as f64 * 0.005, noise, m(&cols[0]), m(&cols[1]), m(&cols[2]), m(&cols[3]), m(&cols[4]));
        }
    }
}
