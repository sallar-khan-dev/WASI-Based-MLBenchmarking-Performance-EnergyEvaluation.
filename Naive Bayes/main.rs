use std::collections::HashMap;
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
    test_ratio: f64,
    seed: u64,
    resplit_each_trial: bool,
    normalize: bool,
    trials: usize,
    var_smoothing: f64,
}

#[derive(Clone)]
struct Dataset {
    x: Vec<Vec<f64>>,
    y: Vec<usize>,
}

#[derive(Clone)]
struct GaussianNB {
    class_labels: Vec<usize>,
    log_priors: Vec<f64>,
    means: Vec<Vec<f64>>,
    vars: Vec<Vec<f64>>,
}

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

    let dataset = get_arg_value(&args, "--datasets")
        .or_else(|| get_arg_value(&args, "--dataset"))
        .unwrap_or_else(|| "data/breastCancer_100000.csv".to_string());

    let test_ratio = get_arg_value(&args, "--test-ratio")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.2);

    let seed = get_arg_value(&args, "--seed")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(42);

    let resplit_each_trial = get_flag_bool01(&args, "--resplit-each-trial", true);
    let normalize = get_flag_bool01(&args, "--normalize", true);

    let trials = get_arg_value(&args, "--trials")
        .and_then(|v| v.parse::<usize>().ok())
        .or_else(|| get_arg_value(&args, "--min-runs").and_then(|v| v.parse::<usize>().ok()))
        .unwrap_or(50);

    let var_smoothing = get_arg_value(&args, "--var-smoothing")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(1e-9);

    Args {
        dataset,
        test_ratio,
        seed,
        resplit_each_trial,
        normalize,
        trials,
        var_smoothing,
    }
}

fn load_dataset_csv(path: &str) -> Dataset {
    let file = File::open(path).unwrap_or_else(|e| panic!("Cannot open dataset {}: {}", path, e));
    let reader = BufReader::new(file);

    let mut lines = reader.lines();
    let header = lines
        .next()
        .unwrap_or_else(|| panic!("Dataset {} is empty", path))
        .unwrap();
    let headers: Vec<String> = header.split(',').map(|s| s.trim().to_string()).collect();
    let skip_first = headers
        .get(0)
        .map(|h| {
            let h = h.to_ascii_lowercase();
            h == "id" || h.ends_with("_id") || h.contains("identifier")
        })
        .unwrap_or(false);

    let mut x: Vec<Vec<f64>> = Vec::new();
    let mut y: Vec<usize> = Vec::new();
    let mut label_map: HashMap<String, usize> = HashMap::new();

    for line in lines {
        let line = line.unwrap();
        if line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 2 {
            continue;
        }

        let label_raw = parts[parts.len() - 1].trim().to_string();
        let next_id = label_map.len();
        let label_id = *label_map.entry(label_raw).or_insert(next_id);

        let feat_start = if skip_first { 1 } else { 0 };
        let feat_end = parts.len() - 1;
        if feat_start >= feat_end {
            continue;
        }

        let mut row = Vec::with_capacity(feat_end - feat_start);
        for v in &parts[feat_start..feat_end] {
            let parsed = v.trim().parse::<f64>().unwrap_or_else(|_| {
                panic!("Non-numeric feature '{}' encountered in {}", v.trim(), path)
            });
            row.push(parsed);
        }

        x.push(row);
        y.push(label_id);
    }

    Dataset { x, y }
}

fn train_test_split_indices(n: usize, test_ratio: f64, rng: &mut StdRng) -> (Vec<usize>, Vec<usize>) {
    let mut idx: Vec<usize> = (0..n).collect();
    idx.shuffle(rng);

    let test_n = ((n as f64) * test_ratio).round() as usize;
    let test_n = test_n.clamp(1, n.saturating_sub(1));
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

    let mut std = vec![0.0; d];
    for j in 0..d {
        std[j] = (var[j] / n).sqrt();
        if std[j] == 0.0 {
            std[j] = 1.0;
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

fn fit_gaussian_nb(x: &[Vec<f64>], y: &[usize], train_idx: &[usize], var_smoothing: f64) -> GaussianNB {
    let d = x[0].len();
    let mut classes: Vec<usize> = train_idx.iter().map(|&i| y[i]).collect();
    classes.sort_unstable();
    classes.dedup();

    let mut global_max_var = 0.0f64;
    for j in 0..d {
        let mut mu = 0.0;
        for &i in train_idx {
            mu += x[i][j];
        }
        mu /= train_idx.len() as f64;

        let mut v = 0.0;
        for &i in train_idx {
            let diff = x[i][j] - mu;
            v += diff * diff;
        }
        v /= train_idx.len() as f64;
        if v > global_max_var {
            global_max_var = v;
        }
    }
    let eps = if global_max_var > 0.0 {
        var_smoothing * global_max_var
    } else {
        var_smoothing.max(1e-12)
    };

    let mut log_priors = Vec::with_capacity(classes.len());
    let mut means = Vec::with_capacity(classes.len());
    let mut vars = Vec::with_capacity(classes.len());

    for &class_id in &classes {
        let class_idx: Vec<usize> = train_idx.iter().copied().filter(|&i| y[i] == class_id).collect();
        let n_c = class_idx.len();
        let prior = (n_c as f64) / (train_idx.len() as f64);
        log_priors.push(prior.ln());

        let mut mu = vec![0.0; d];
        for &i in &class_idx {
            for j in 0..d {
                mu[j] += x[i][j];
            }
        }
        for j in 0..d {
            mu[j] /= n_c as f64;
        }

        let mut var = vec![0.0; d];
        for &i in &class_idx {
            for j in 0..d {
                let diff = x[i][j] - mu[j];
                var[j] += diff * diff;
            }
        }
        for j in 0..d {
            var[j] = (var[j] / n_c as f64) + eps;
            if var[j] <= 0.0 {
                var[j] = eps.max(1e-12);
            }
        }

        means.push(mu);
        vars.push(var);
    }

    GaussianNB {
        class_labels: classes,
        log_priors,
        means,
        vars,
    }
}

fn predict_one(model: &GaussianNB, row: &[f64]) -> usize {
    let ln_2pi = (2.0 * std::f64::consts::PI).ln();
    let mut best_class = model.class_labels[0];
    let mut best_score = f64::NEG_INFINITY;

    for c in 0..model.class_labels.len() {
        let mut score = model.log_priors[c];
        for j in 0..row.len() {
            let var = model.vars[c][j];
            let diff = row[j] - model.means[c][j];
            score += -0.5 * ((diff * diff) / var + var.ln() + ln_2pi);
        }
        if score > best_score {
            best_score = score;
            best_class = model.class_labels[c];
        }
    }
    best_class
}

fn predict(model: &GaussianNB, x: &[Vec<f64>], test_idx: &[usize]) -> Vec<usize> {
    test_idx.iter().map(|&i| predict_one(model, &x[i])).collect()
}

fn accuracy(y_true: &[usize], y_pred: &[usize]) -> f64 {
    let correct = y_true.iter().zip(y_pred.iter()).filter(|(a, b)| a == b).count();
    correct as f64 / y_true.len() as f64
}

fn precision_recall_f1_macro(y_true: &[usize], y_pred: &[usize]) -> (f64, f64, f64) {
    let max_label = *y_true.iter().chain(y_pred.iter()).max().unwrap_or(&0);
    let mut p_sum = 0.0;
    let mut r_sum = 0.0;
    let mut f1_sum = 0.0;
    let classes = max_label + 1;

    for c in 0..=max_label {
        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut fn_ = 0usize;

        for (&yt, &yp) in y_true.iter().zip(y_pred.iter()) {
            if yp == c && yt == c {
                tp += 1;
            } else if yp == c && yt != c {
                fp += 1;
            } else if yp != c && yt == c {
                fn_ += 1;
            }
        }

        let p = if tp + fp == 0 { 0.0 } else { tp as f64 / (tp + fp) as f64 };
        let r = if tp + fn_ == 0 { 0.0 } else { tp as f64 / (tp + fn_) as f64 };
        let f1 = if (p + r) == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) };

        p_sum += p;
        r_sum += r;
        f1_sum += f1;
    }

    (
        p_sum / classes as f64,
        r_sum / classes as f64,
        f1_sum / classes as f64,
    )
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / (v.len() as f64)
    }
}

fn main() {
    let args = parse_args();

    let data = load_dataset_csv(&args.dataset);
    let n = data.x.len();
    if n < 5 {
        panic!("Dataset too small or failed to parse: {}", args.dataset);
    }

    let mut accs = Vec::with_capacity(args.trials);
    let mut precs = Vec::with_capacity(args.trials);
    let mut recs = Vec::with_capacity(args.trials);
    let mut f1s = Vec::with_capacity(args.trials);

    let mut base_rng = StdRng::seed_from_u64(args.seed);
    let (fixed_train, fixed_test) = train_test_split_indices(n, args.test_ratio, &mut base_rng);

    for t in 0..args.trials {
        let mut rng = StdRng::seed_from_u64(args.seed.wrapping_add(t as u64));

        let (train_idx, test_idx) = if args.resplit_each_trial {
            train_test_split_indices(n, args.test_ratio, &mut rng)
        } else {
            (fixed_train.clone(), fixed_test.clone())
        };

        let mut x_local = data.x.clone();
        if args.normalize {
            let (mu, sd) = compute_norm_stats(&x_local, &train_idx);
            apply_normalization_inplace(&mut x_local, &mu, &sd);
        }

        let model = fit_gaussian_nb(&x_local, &data.y, &train_idx, args.var_smoothing);
        let y_true: Vec<usize> = test_idx.iter().map(|&i| data.y[i]).collect();
        let y_pred = predict(&model, &x_local, &test_idx);

        let acc = accuracy(&y_true, &y_pred);
        let (prec, rec, f1) = precision_recall_f1_macro(&y_true, &y_pred);

        accs.push(acc);
        precs.push(prec);
        recs.push(rec);
        f1s.push(f1);
    }

    let out = json!({
        "accuracy_mean": mean(&accs),
        "precision_mean": mean(&precs),
        "recall_mean": mean(&recs),
        "f1_mean": mean(&f1s),
        "trials_k": args.trials
    });

    println!("{}", out);
}
