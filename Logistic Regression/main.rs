use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use serde_json::json;

#[derive(Clone, Debug)]
struct Args {
    dataset: String,
    epochs: usize,
    lr: f64,

    // Split config
    test_ratio: f64,
    seed: u64,
    resplit_each_trial: bool,

    // Normalization
    normalize: bool,

    // Fixed workload: number of internal trials (K)
    trials: usize,
}

// Simple CLI parsing (no clap) to keep WASI-friendly
fn get_arg_value(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|x| x == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn get_flag_bool01(args: &[String], key: &str, default: bool) -> bool {
    match get_arg_value(args, key) {
        Some(v) => {
            let v = v.trim();
            v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes")
        }
        None => default,
    }
}

fn parse_args() -> Args {
    let args: Vec<String> = env::args().collect();

    // Dataset: support `--datasets <path>` (your runner uses this)
    // If multiple are provided in the future, keep first for now.
    let dataset = get_arg_value(&args, "--datasets")
        .unwrap_or_else(|| "data/breastCancer_100000.csv".to_string());

    let epochs = get_arg_value(&args, "--epochs")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(10);

    let lr = get_arg_value(&args, "--lr")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.001);

    let test_ratio = get_arg_value(&args, "--test-ratio")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.2);

    let seed = get_arg_value(&args, "--seed")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(42);

    let resplit_each_trial = get_flag_bool01(&args, "--resplit-each-trial", true);
    let normalize = get_flag_bool01(&args, "--normalize", true);

    // Supervisor-compliant: fixed K internal trials.
    // Use --min-runs as K to remain compatible with your runner.
    // If someone passes --trials, accept it too.
    let trials = get_arg_value(&args, "--trials")
        .and_then(|v| v.parse::<usize>().ok())
        .or_else(|| {
            get_arg_value(&args, "--min-runs")
                .and_then(|v| v.parse::<usize>().ok())
        })
        .unwrap_or(50);

    // NOTE: We intentionally ignore:
    // --max-runs, --rel-precision, --min-seconds (if present)
    // because CI-based stopping inside the model program is forbidden per supervisor.

    Args {
        dataset,
        epochs,
        lr,
        test_ratio,
        seed,
        resplit_each_trial,
        normalize,
        trials,
    }
}

#[derive(Clone)]
struct Dataset {
    x: Vec<Vec<f64>>,
    y: Vec<u8>,
}

fn load_dataset_csv(path: &str) -> Dataset {
    let file = File::open(path).unwrap_or_else(|e| panic!("Cannot open dataset {}: {}", path, e));
    let reader = BufReader::new(file);

    let mut x: Vec<Vec<f64>> = Vec::new();
    let mut y: Vec<u8> = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line.unwrap();
        if i == 0 {
            continue; // header
        }
        if line.trim().is_empty() {
            continue;
        }

        // Expected format (your files):
        // col0 = ID
        // col1..colN-2 = features
        // last col = Diagnosis (M/B)
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 3 {
            continue;
        }

        let label_str = parts[parts.len() - 1].trim();
        let label = if label_str == "M" { 1u8 } else { 0u8 };

        let mut feats: Vec<f64> = Vec::with_capacity(parts.len() - 2);
        for v in &parts[1..parts.len() - 1] {
            feats.push(v.trim().parse::<f64>().unwrap());
        }

        x.push(feats);
        y.push(label);
    }

    Dataset { x, y }
}

fn train_test_split_indices(n: usize, test_ratio: f64, rng: &mut StdRng) -> (Vec<usize>, Vec<usize>) {
    let mut idx: Vec<usize> = (0..n).collect();
    idx.shuffle(rng);

    let test_n = ((n as f64) * test_ratio).round() as usize;
    let test_n = test_n.clamp(1, n.saturating_sub(1)); // avoid empty train or test
    let test_idx = idx[..test_n].to_vec();
    let train_idx = idx[test_n..].to_vec();
    (train_idx, test_idx)
}

fn compute_norm_stats(x: &[Vec<f64>], train_idx: &[usize]) -> (Vec<f64>, Vec<f64>) {
    let d = x[0].len();
    let mut mean = vec![0.0; d];
    let mut var = vec![0.0; d];

    let n = train_idx.len() as f64;

    for &i in train_idx {
        for j in 0..d {
            mean[j] += x[i][j];
        }
    }
    for j in 0..d {
        mean[j] /= n;
    }

    for &i in train_idx {
        for j in 0..d {
            let diff = x[i][j] - mean[j];
            var[j] += diff * diff;
        }
    }
    for j in 0..d {
        var[j] /= n;
    }

    let mut std = vec![0.0; d];
    for j in 0..d {
        std[j] = var[j].sqrt();
        if std[j] == 0.0 {
            std[j] = 1.0; // avoid division by zero
        }
    }

    (mean, std)
}

fn apply_normalization_inplace(x: &mut [Vec<f64>], mean: &[f64], std: &[f64]) {
    for row in x.iter_mut() {
        for j in 0..row.len() {
            row[j] = (row[j] - mean[j]) / std[j];
        }
    }
}

fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, w)| x * w).sum()
}

fn train_logreg_sgd(x: &[Vec<f64>], y: &[u8], train_idx: &[usize], epochs: usize, lr: f64, rng: &mut StdRng) -> Vec<f64> {
    let d = x[0].len();
    let mut w = vec![0.0; d];

    let mut order: Vec<usize> = train_idx.to_vec();

    for _ in 0..epochs {
        order.shuffle(rng);
        for &i in &order {
            let z = dot(&x[i], &w);
            let p = sigmoid(z);
            let yi = y[i] as f64;
            let err = p - yi;

            // gradient step
            for j in 0..d {
                w[j] -= lr * err * x[i][j];
            }
        }
    }
    w
}

fn eval_metrics(x: &[Vec<f64>], y: &[u8], test_idx: &[usize], w: &[f64]) -> (f64, f64, f64, f64) {
    let mut tp = 0.0;
    let mut tn = 0.0;
    let mut fp = 0.0;
    let mut fn_ = 0.0;

    for &i in test_idx {
        let z = dot(&x[i], w);
        let p = sigmoid(z);
        let pred = if p >= 0.5 { 1u8 } else { 0u8 };
        let yi = y[i];

        match (pred, yi) {
            (1, 1) => tp += 1.0,
            (0, 0) => tn += 1.0,
            (1, 0) => fp += 1.0,
            (0, 1) => fn_ += 1.0,
            _ => {}
        }
    }

    let total = tp + tn + fp + fn_;
    let acc = if total > 0.0 { (tp + tn) / total } else { 0.0 };
    let prec = if (tp + fp) > 0.0 { tp / (tp + fp) } else { 0.0 };
    let rec = if (tp + fn_) > 0.0 { tp / (tp + fn_) } else { 0.0 };
    let f1 = if (prec + rec) > 0.0 { 2.0 * prec * rec / (prec + rec) } else { 0.0 };

    (acc, prec, rec, f1)
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { return 0.0; }
    v.iter().sum::<f64>() / (v.len() as f64)
}

fn main() {
    let args = parse_args();

    let mut data = load_dataset_csv(&args.dataset);
    let n = data.x.len();
    if n < 5 {
        panic!("Dataset too small or failed to parse: {}", args.dataset);
    }

    // We keep normalization consistent with the trial’s train split:
    // compute mean/std from TRAIN only, apply to all samples.
    let mut accs = Vec::with_capacity(args.trials);
    let mut precs = Vec::with_capacity(args.trials);
    let mut recs = Vec::with_capacity(args.trials);
    let mut f1s = Vec::with_capacity(args.trials);

    // If not resplitting each trial, generate one split once.
    let mut base_rng = StdRng::seed_from_u64(args.seed);
    let (fixed_train, fixed_test) = train_test_split_indices(n, args.test_ratio, &mut base_rng);

    for t in 0..args.trials {
        let mut rng = StdRng::seed_from_u64(args.seed.wrapping_add(t as u64));

        let (train_idx, test_idx) = if args.resplit_each_trial {
            train_test_split_indices(n, args.test_ratio, &mut rng)
        } else {
            (fixed_train.clone(), fixed_test.clone())
        };

        // work on a local copy of X if normalization is enabled
        let mut x_local = data.x.clone();
        if args.normalize {
            let (mu, sd) = compute_norm_stats(&x_local, &train_idx);
            apply_normalization_inplace(&mut x_local, &mu, &sd);
        }

        // Train fixed epochs over fixed train set
        let w = train_logreg_sgd(&x_local, &data.y, &train_idx, args.epochs, args.lr, &mut rng);

        // Evaluate on test set
        let (acc, prec, rec, f1) = eval_metrics(&x_local, &data.y, &test_idx, &w);

        accs.push(acc);
        precs.push(prec);
        recs.push(rec);
        f1s.push(f1);
    }

    // Output ONLY means (Rust does not do CI stopping)
    let out = json!({
        "accuracy_mean": mean(&accs),
        "precision_mean": mean(&precs),
        "recall_mean": mean(&recs),
        "f1_mean": mean(&f1s),
        "trials_k": args.trials
    });

    println!("{}", out);
}
