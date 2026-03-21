use std::cmp::Ordering;
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
    max_depth: usize,
    min_samples_split: usize,
    min_samples_leaf: usize,
    threshold_bins: usize,
    train_cap: usize,
}

#[derive(Clone)]
struct Dataset {
    x: Vec<Vec<f64>>,
    y: Vec<usize>,
    n_classes: usize,
}

#[derive(Clone)]
enum Node {
    Leaf {
        class_id: usize,
    },
    Split {
        feature: usize,
        threshold: f64,
        left: Box<Node>,
        right: Box<Node>,
    },
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
        .unwrap_or_else(|| "data/decisionTree_100000.csv".to_string());

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

    let max_depth = get_arg_value(&args, "--max-depth")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8);

    let min_samples_split = get_arg_value(&args, "--min-samples-split")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(32);

    let min_samples_leaf = get_arg_value(&args, "--min-samples-leaf")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8);

    let threshold_bins = get_arg_value(&args, "--threshold-bins")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8);

    let train_cap = get_arg_value(&args, "--train-cap")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4096);

    Args {
        dataset,
        test_ratio,
        seed,
        resplit_each_trial,
        normalize,
        trials,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        threshold_bins,
        train_cap,
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
        if parts.len() < 3 {
            continue;
        }

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

        let label_raw = parts[parts.len() - 1].trim().to_string();
        let next_id = label_map.len();
        let label_id = *label_map.entry(label_raw).or_insert(next_id);

        x.push(row);
        y.push(label_id);
    }

    Dataset {
        x,
        y,
        n_classes: label_map.len(),
    }
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

fn majority_class(y: &[usize], idx: &[usize], n_classes: usize) -> usize {
    let mut counts = vec![0usize; n_classes];
    for &i in idx {
        counts[y[i]] += 1;
    }

    let mut best_class = 0usize;
    let mut best_count = 0usize;
    for c in 0..n_classes {
        if counts[c] > best_count {
            best_count = counts[c];
            best_class = c;
        }
    }
    best_class
}

fn is_pure(y: &[usize], idx: &[usize]) -> bool {
    if idx.is_empty() {
        return true;
    }
    let first = y[idx[0]];
    for &i in idx {
        if y[i] != first {
            return false;
        }
    }
    true
}

fn gini_impurity_from_counts(counts: &[usize], total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let t = total as f64;
    let mut sum_sq = 0.0;
    for &c in counts {
        let p = (c as f64) / t;
        sum_sq += p * p;
    }
    1.0 - sum_sq
}

fn quantile_thresholds(values: &[(f64, usize)], bins: usize) -> Vec<f64> {
    let n = values.len();
    if n < 2 {
        return Vec::new();
    }

    let mut thresholds = Vec::new();
    let b = bins.max(2);
    for k in 1..b {
        let pos = ((k * n) / b).min(n - 1);
        let left = values[pos - 1].0;
        let right = values[pos].0;
        if right > left {
            thresholds.push(0.5 * (left + right));
        }
    }

    thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    thresholds.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    thresholds
}

#[derive(Clone, Copy)]
struct SplitSpec {
    feature: usize,
    threshold: f64,
}

fn best_split(
    x: &[Vec<f64>],
    y: &[usize],
    idx: &[usize],
    n_classes: usize,
    min_samples_leaf: usize,
    threshold_bins: usize,
) -> Option<SplitSpec> {
    if idx.len() < 2 * min_samples_leaf {
        return None;
    }

    let d = x[0].len();
    let mut best_gini = f64::INFINITY;
    let mut best: Option<SplitSpec> = None;

    for feature in 0..d {
        let mut pairs: Vec<(f64, usize)> = idx.iter().map(|&i| (x[i][feature], y[i])).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        let thresholds = quantile_thresholds(&pairs, threshold_bins);
        if thresholds.is_empty() {
            continue;
        }

        for &threshold in &thresholds {
            let mut left_counts = vec![0usize; n_classes];
            let mut right_counts = vec![0usize; n_classes];
            let mut left_n = 0usize;
            let mut right_n = 0usize;

            for &(value, class_id) in &pairs {
                if value <= threshold {
                    left_counts[class_id] += 1;
                    left_n += 1;
                } else {
                    right_counts[class_id] += 1;
                    right_n += 1;
                }
            }

            if left_n < min_samples_leaf || right_n < min_samples_leaf {
                continue;
            }

            let g_left = gini_impurity_from_counts(&left_counts, left_n);
            let g_right = gini_impurity_from_counts(&right_counts, right_n);
            let weighted = ((left_n as f64) * g_left + (right_n as f64) * g_right) / (idx.len() as f64);

            if weighted < best_gini {
                best_gini = weighted;
                best = Some(SplitSpec { feature, threshold });
            }
        }
    }

    best
}

fn build_tree(
    x: &[Vec<f64>],
    y: &[usize],
    idx: &[usize],
    depth: usize,
    args: &Args,
    n_classes: usize,
) -> Node {
    let majority = majority_class(y, idx, n_classes);

    if idx.is_empty()
        || depth >= args.max_depth
        || idx.len() < args.min_samples_split
        || is_pure(y, idx)
    {
        return Node::Leaf { class_id: majority };
    }

    let split = match best_split(
        x,
        y,
        idx,
        n_classes,
        args.min_samples_leaf,
        args.threshold_bins,
    ) {
        Some(s) => s,
        None => return Node::Leaf { class_id: majority },
    };

    let mut left_idx = Vec::new();
    let mut right_idx = Vec::new();

    for &i in idx {
        if x[i][split.feature] <= split.threshold {
            left_idx.push(i);
        } else {
            right_idx.push(i);
        }
    }

    if left_idx.len() < args.min_samples_leaf || right_idx.len() < args.min_samples_leaf {
        return Node::Leaf { class_id: majority };
    }

    let left = build_tree(x, y, &left_idx, depth + 1, args, n_classes);
    let right = build_tree(x, y, &right_idx, depth + 1, args, n_classes);

    Node::Split {
        feature: split.feature,
        threshold: split.threshold,
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn predict_one_recursive(node: &Node, row: &[f64]) -> usize {
    match node {
        Node::Leaf { class_id } => *class_id,
        Node::Split {
            feature,
            threshold,
            left,
            right,
        } => {
            if row[*feature] <= *threshold {
                predict_one_recursive(left, row)
            } else {
                predict_one_recursive(right, row)
            }
        }
    }
}

fn predict_many(node: &Node, x: &[Vec<f64>], test_idx: &[usize]) -> Vec<usize> {
    let mut preds = Vec::with_capacity(test_idx.len());
    for &i in test_idx {
        preds.push(predict_one_recursive(node, &x[i]));
    }
    preds
}

fn tree_depth(node: &Node) -> usize {
    match node {
        Node::Leaf { .. } => 1,
        Node::Split { left, right, .. } => 1 + usize::max(tree_depth(left), tree_depth(right)),
    }
}

fn tree_nodes(node: &Node) -> usize {
    match node {
        Node::Leaf { .. } => 1,
        Node::Split { left, right, .. } => 1 + tree_nodes(left) + tree_nodes(right),
    }
}

fn eval_metrics(y_true: &[usize], y_pred: &[usize], n_classes: usize) -> (f64, f64, f64, f64) {
    let mut correct = 0usize;
    for (&yt, &yp) in y_true.iter().zip(y_pred.iter()) {
        if yt == yp {
            correct += 1;
        }
    }
    let acc = (correct as f64) / (y_true.len() as f64);

    let mut prec_sum = 0.0;
    let mut rec_sum = 0.0;
    let mut f1_sum = 0.0;

    for c in 0..n_classes {
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

        let p = if tp + fp == 0 {
            0.0
        } else {
            (tp as f64) / ((tp + fp) as f64)
        };

        let r = if tp + fn_ == 0 {
            0.0
        } else {
            (tp as f64) / ((tp + fn_) as f64)
        };

        let f1 = if (p + r) == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        };

        prec_sum += p;
        rec_sum += r;
        f1_sum += f1;
    }

    let denom = n_classes as f64;
    (acc, prec_sum / denom, rec_sum / denom, f1_sum / denom)
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / (v.len() as f64)
}

fn mean_usize(v: &[usize]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let total: usize = v.iter().sum();
    (total as f64) / (v.len() as f64)
}

fn capped_training_subset(train_idx: &[usize], cap: usize, rng: &mut StdRng) -> Vec<usize> {
    if train_idx.len() <= cap {
        return train_idx.to_vec();
    }
    let mut subset = train_idx.to_vec();
    subset.shuffle(rng);
    subset.truncate(cap);
    subset
}

fn main() {
    let args = parse_args();

    let data = load_dataset_csv(&args.dataset);
    let n = data.x.len();
    if n < 10 {
        panic!("Dataset too small or failed to parse: {}", args.dataset);
    }

    let mut accs = Vec::with_capacity(args.trials);
    let mut precs = Vec::with_capacity(args.trials);
    let mut recs = Vec::with_capacity(args.trials);
    let mut f1s = Vec::with_capacity(args.trials);
    let mut node_counts = Vec::with_capacity(args.trials);
    let mut depths = Vec::with_capacity(args.trials);

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

        let tree_train_idx = capped_training_subset(&train_idx, args.train_cap, &mut rng);
        let tree = build_tree(&x_local, &data.y, &tree_train_idx, 0, &args, data.n_classes);

        let preds = predict_many(&tree, &x_local, &test_idx);
        let y_true: Vec<usize> = test_idx.iter().map(|&i| data.y[i]).collect();

        let (acc, prec, rec, f1) = eval_metrics(&y_true, &preds, data.n_classes);

        accs.push(acc);
        precs.push(prec);
        recs.push(rec);
        f1s.push(f1);
        node_counts.push(tree_nodes(&tree));
        depths.push(tree_depth(&tree));
    }

    let out = json!({
        "accuracy_mean": mean(&accs),
        "precision_mean": mean(&precs),
        "recall_mean": mean(&recs),
        "f1_mean": mean(&f1s),
        "tree_nodes_mean": mean_usize(&node_counts),
        "tree_depth_mean": mean_usize(&depths),
        "train_cap": args.train_cap,
        "trials_k": args.trials
    });

    println!("{}", out);
}
