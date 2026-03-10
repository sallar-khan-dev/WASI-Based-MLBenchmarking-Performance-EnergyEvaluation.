use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn parse_args() -> (String, usize, usize, usize) {
    let args: Vec<String> = env::args().collect();
    let mut dataset = String::new();
    let mut k = 3usize;
    let mut max_iters = 20usize;
    let mut trials = 10usize;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--dataset" => {
                dataset = args[i + 1].clone();
                i += 2;
            }
            "--k" => {
                k = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--max-iters" => {
                max_iters = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--trials" => {
                trials = args[i + 1].parse().unwrap();
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }
    (dataset, k, max_iters, trials)
}

fn load_dataset(path: &str) -> (Vec<Vec<f64>>, Vec<usize>) {
    let file = File::open(path).expect("Cannot open dataset");
    let reader = BufReader::new(file);
    let mut data = Vec::new();
    let mut labels = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let l = line.unwrap();
        if i == 0 {
            continue;
        }
        let parts: Vec<&str> = l.split(',').collect();
        let mut row = Vec::new();
        for j in 0..parts.len() - 1 {
            row.push(parts[j].parse::<f64>().unwrap());
        }
        data.push(row);
        labels.push(parts.last().unwrap().parse::<usize>().unwrap());
    }
    (data, labels)
}

fn dist2(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn kmeans(data: &[Vec<f64>], k: usize, iters: usize) -> Vec<usize> {
    let dim = data[0].len();
    let mut centroids = data[0..k].to_vec();
    let mut assign = vec![0usize; data.len()];

    for _ in 0..iters {
        for (i, p) in data.iter().enumerate() {
            let mut best = 0usize;
            let mut best_d = f64::MAX;
            for (c, cent) in centroids.iter().enumerate() {
                let d = dist2(p, cent);
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            assign[i] = best;
        }

        let mut newc = vec![vec![0.0; dim]; k];
        let mut counts = vec![0usize; k];

        for (i, p) in data.iter().enumerate() {
            let c = assign[i];
            counts[c] += 1;
            for d in 0..dim {
                newc[c][d] += p[d];
            }
        }

        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..dim {
                    newc[c][d] /= counts[c] as f64;
                }
            }
        }
        centroids = newc;
    }

    assign
}

fn majority_map(pred_clusters: &[usize], labels: &[usize], k: usize) -> Vec<usize> {
    let max_label = *labels.iter().max().unwrap_or(&0);
    let mut table = vec![vec![0usize; max_label + 1]; k];

    for (&c, &y) in pred_clusters.iter().zip(labels.iter()) {
        table[c][y] += 1;
    }

    let mut mapping = vec![0usize; k];
    for c in 0..k {
        let mut best_label = 0usize;
        let mut best_count = 0usize;
        for y in 0..=max_label {
            if table[c][y] > best_count {
                best_count = table[c][y];
                best_label = y;
            }
        }
        mapping[c] = best_label;
    }
    mapping
}

fn relabel(pred_clusters: &[usize], mapping: &[usize]) -> Vec<usize> {
    pred_clusters.iter().map(|&c| mapping[c]).collect()
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

fn main() {
    let (dataset, k, max_iters, trials) = parse_args();
    let (data, labels) = load_dataset(&dataset);

    let mut acc_sum = 0.0;
    let mut prec_sum = 0.0;
    let mut rec_sum = 0.0;
    let mut f1_sum = 0.0;

    for _ in 0..trials {
        let pred_clusters = kmeans(&data, k, max_iters);
        let mapping = majority_map(&pred_clusters, &labels, k);
        let y_pred = relabel(&pred_clusters, &mapping);

        let acc = accuracy(&labels, &y_pred);
        let (prec, rec, f1) = precision_recall_f1_macro(&labels, &y_pred);

        acc_sum += acc;
        prec_sum += prec;
        rec_sum += rec;
        f1_sum += f1;
    }

    let t = trials as f64;
    println!(
        "{{\"accuracy_mean\":{},\"precision_mean\":{},\"recall_mean\":{},\"f1_mean\":{},\"trials_k\":{}}}",
        acc_sum / t,
        prec_sum / t,
        rec_sum / t,
        f1_sum / t,
        trials
    );
}
