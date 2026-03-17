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
    hidden: usize,
    epochs: usize,
    lr: f64,
    test_ratio: f64,
    seed: u64,
    resplit_each_trial: bool,
    normalize: bool,
    trials: usize,
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

    let hidden = get_arg_value(&args, "--hidden")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(32);

    let epochs = get_arg_value(&args, "--epochs")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(10);

    let lr = get_arg_value(&args, "--lr")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.01);

    let test_ratio = get_arg_value(&args, "--test-ratio")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.2);

    let seed = get_arg_value(&args, "--seed")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(42);

    // Default to a fixed split so repeated internal trials reflect model-training variation,
    // not repeated full-dataset re-splitting and re-normalization overhead.
    let resplit_each_trial = get_flag_bool01(&args, "--resplit-each-trial", false);
    let normalize = get_flag_bool01(&args, "--normalize", true);

    let trials = get_arg_value(&args, "--trials")
        .and_then(|v| v.parse::<usize>().ok())
        .or_else(|| get_arg_value(&args, "--min-runs").and_then(|v| v.parse::<usize>().ok()))
        .unwrap_or(50);

    Args {
        dataset,
        hidden,
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
    n: usize,
    d: usize,
    x: Vec<f64>,
    y: Vec<u8>,
}

impl Dataset {
    #[inline]
    fn row(&self, i: usize) -> &[f64] {
        let start = i * self.d;
        &self.x[start..start + self.d]
    }
}

fn load_dataset_csv(path: &str) -> Dataset {
    let file = File::open(path).unwrap_or_else(|e| panic!("Cannot open dataset {}: {}", path, e));
    let reader = BufReader::new(file);

    let mut x: Vec<f64> = Vec::new();
    let mut y: Vec<u8> = Vec::new();
    let mut d: Option<usize> = None;

    for (i, line) in reader.lines().enumerate() {
        let line = line.unwrap();
        if i == 0 || line.trim().is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 3 {
            continue;
        }

        let label_str = parts[parts.len() - 1].trim();
        let label = if label_str == "M" || label_str == "1" { 1u8 } else { 0u8 };

        let feat_count = parts.len() - 2;
        if d.is_none() {
            d = Some(feat_count);
        }
        let cur_d = d.unwrap();
        assert_eq!(feat_count, cur_d, "Inconsistent feature count in dataset");

        for v in &parts[1..parts.len() - 1] {
            x.push(v.trim().parse::<f64>().unwrap());
        }
        y.push(label);
    }

    let d = d.unwrap_or(0);
    let n = y.len();
    assert_eq!(x.len(), n * d, "Dataset shape mismatch");

    Dataset { n, d, x, y }
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

fn compute_norm_stats(ds: &Dataset, train_idx: &[usize]) -> (Vec<f64>, Vec<f64>) {
    let d = ds.d;
    let mut mean = vec![0.0; d];
    let mut var = vec![0.0; d];
    let n = train_idx.len() as f64;

    for &i in train_idx {
        let row = ds.row(i);
        for j in 0..d {
            mean[j] += row[j];
        }
    }
    for j in 0..d {
        mean[j] /= n;
    }
    for &i in train_idx {
        let row = ds.row(i);
        for j in 0..d {
            let diff = row[j] - mean[j];
            var[j] += diff * diff;
        }
    }
    for j in 0..d {
        let mut s = (var[j] / n).sqrt();
        if s == 0.0 {
            s = 1.0;
        }
        var[j] = s;
    }
    (mean, var)
}

fn normalized_copy(ds: &Dataset, mean: &[f64], std: &[f64]) -> Dataset {
    let mut x = ds.x.clone();
    for row in x.chunks_exact_mut(ds.d) {
        for j in 0..ds.d {
            row[j] = (row[j] - mean[j]) / std[j];
        }
    }
    Dataset {
        n: ds.n,
        d: ds.d,
        x,
        y: ds.y.clone(),
    }
}

#[inline]
fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        let ez = (-z).exp();
        1.0 / (1.0 + ez)
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)
    }
}

#[inline]
fn tanh_deriv_from_output(h: f64) -> f64 {
    1.0 - h * h
}

#[derive(Clone)]
struct Mlp {
    d: usize,
    hidden: usize,
    w1: Vec<f64>, // [hidden * d]
    b1: Vec<f64>,
    w2: Vec<f64>,
    b2: f64,
}

fn init_mlp(d: usize, hidden: usize, rng: &mut StdRng) -> Mlp {
    use rand::Rng;
    let scale1 = (2.0 / (d as f64 + hidden as f64)).sqrt();
    let scale2 = (2.0 / (hidden as f64 + 1.0)).sqrt();

    let mut w1 = vec![0.0; hidden * d];
    let b1 = vec![0.0; hidden];
    let mut w2 = vec![0.0; hidden];

    for h in 0..hidden {
        let row = &mut w1[h * d..(h + 1) * d];
        for w in row.iter_mut() {
            *w = rng.gen_range(-scale1..scale1);
        }
        w2[h] = rng.gen_range(-scale2..scale2);
    }

    Mlp { d, hidden, w1, b1, w2, b2: 0.0 }
}

#[inline]
fn forward(model: &Mlp, x: &[f64], hidden_buf: &mut [f64]) -> f64 {
    for h in 0..model.hidden {
        let row = &model.w1[h * model.d..(h + 1) * model.d];
        let mut z = model.b1[h];
        for j in 0..model.d {
            z += row[j] * x[j];
        }
        hidden_buf[h] = z.tanh();
    }
    let mut z2 = model.b2;
    for h in 0..model.hidden {
        z2 += model.w2[h] * hidden_buf[h];
    }
    sigmoid(z2)
}

fn train_mlp_sgd(ds: &Dataset, train_idx: &[usize], hidden: usize, epochs: usize, lr: f64, rng: &mut StdRng) -> Mlp {
    let mut model = init_mlp(ds.d, hidden, rng);
    let mut order: Vec<usize> = train_idx.to_vec();
    let mut hidden_buf = vec![0.0; hidden];
    let mut delta1 = vec![0.0; hidden];

    for _ in 0..epochs {
        order.shuffle(rng);
        for &i in &order {
            let x = ds.row(i);
            let p = forward(&model, x, &mut hidden_buf);
            let yi = ds.y[i] as f64;
            let delta2 = p - yi;

            for h in 0..hidden {
                delta1[h] = delta2 * model.w2[h] * tanh_deriv_from_output(hidden_buf[h]);
            }

            for h in 0..hidden {
                model.w2[h] -= lr * delta2 * hidden_buf[h];
            }
            model.b2 -= lr * delta2;

            for h in 0..hidden {
                let row = &mut model.w1[h * model.d..(h + 1) * model.d];
                let dh = delta1[h];
                for j in 0..model.d {
                    row[j] -= lr * dh * x[j];
                }
                model.b1[h] -= lr * dh;
            }
        }
    }
    model
}

fn eval_metrics(ds: &Dataset, test_idx: &[usize], model: &Mlp) -> (f64, f64, f64, f64) {
    let mut tp = 0.0;
    let mut tn = 0.0;
    let mut fp = 0.0;
    let mut fn_ = 0.0;
    let mut hidden_buf = vec![0.0; model.hidden];

    for &i in test_idx {
        let p = forward(model, ds.row(i), &mut hidden_buf);
        let pred = if p >= 0.5 { 1u8 } else { 0u8 };
        let yi = ds.y[i];
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
    v.iter().sum::<f64>() / v.len() as f64
}

fn main() {
    let args = parse_args();
    let ds_raw = load_dataset_csv(&args.dataset);
    assert!(ds_raw.n > 0, "Dataset is empty");
    assert!(args.hidden > 0, "hidden must be > 0");

    let mut trial_acc = Vec::with_capacity(args.trials);
    let mut trial_prec = Vec::with_capacity(args.trials);
    let mut trial_rec = Vec::with_capacity(args.trials);
    let mut trial_f1 = Vec::with_capacity(args.trials);

    let mut base_rng = StdRng::seed_from_u64(args.seed);
    let fixed_split = if args.resplit_each_trial {
        None
    } else {
        Some(train_test_split_indices(ds_raw.n, args.test_ratio, &mut base_rng))
    };

    // Fast path: fixed split and fixed normalization stats reused across all trials.
    let precomputed_ds = if let Some((ref train_idx, _)) = fixed_split {
        if args.normalize {
            let (mean, std) = compute_norm_stats(&ds_raw, train_idx);
            Some(normalized_copy(&ds_raw, &mean, &std))
        } else {
            Some(ds_raw.clone())
        }
    } else {
        None
    };

    for t in 0..args.trials {
        let mut rng = StdRng::seed_from_u64(args.seed + t as u64);
        let (train_idx, test_idx) = match &fixed_split {
            Some((tr, te)) => (tr.clone(), te.clone()),
            None => train_test_split_indices(ds_raw.n, args.test_ratio, &mut rng),
        };

        let ds_trial = match &precomputed_ds {
            Some(ds) => ds,
            None => {
                if args.normalize {
                    let (mean, std) = compute_norm_stats(&ds_raw, &train_idx);
                    let ds_norm = normalized_copy(&ds_raw, &mean, &std);
                    let model = train_mlp_sgd(&ds_norm, &train_idx, args.hidden, args.epochs, args.lr, &mut rng);
                    let (acc, prec, rec, f1) = eval_metrics(&ds_norm, &test_idx, &model);
                    trial_acc.push(acc);
                    trial_prec.push(prec);
                    trial_rec.push(rec);
                    trial_f1.push(f1);
                    continue;
                } else {
                    &ds_raw
                }
            }
        };

        let model = train_mlp_sgd(ds_trial, &train_idx, args.hidden, args.epochs, args.lr, &mut rng);
        let (acc, prec, rec, f1) = eval_metrics(ds_trial, &test_idx, &model);
        trial_acc.push(acc);
        trial_prec.push(prec);
        trial_rec.push(rec);
        trial_f1.push(f1);
    }

    let out = json!({
        "accuracy_mean": mean(&trial_acc),
        "precision_mean": mean(&trial_prec),
        "recall_mean": mean(&trial_rec),
        "f1_mean": mean(&trial_f1),
        "trials_k": args.trials,
        "hidden": args.hidden,
        "epochs": args.epochs,
        "lr": args.lr,
        "resplit_each_trial": args.resplit_each_trial,
        "normalize": args.normalize,
    });

    println!("{}", out.to_string());
}
