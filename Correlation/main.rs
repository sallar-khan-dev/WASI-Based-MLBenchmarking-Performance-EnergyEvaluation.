use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};

use serde_json::json;

#[derive(Clone, Debug)]
struct Args { dataset: String, trials: usize }

fn get_arg_value(args: &[String], key: &str) -> Option<String> {
    args.iter().position(|x| x == key).and_then(|i| args.get(i + 1)).cloned()
}

fn parse_args() -> Args {
    let args: Vec<String> = env::args().collect();
    let dataset = get_arg_value(&args, "--dataset")
        .or_else(|| get_arg_value(&args, "--datasets"))
        .unwrap_or_else(|| "data/correlation_100K.csv".to_string());
    let trials = get_arg_value(&args, "--trials").and_then(|v| v.parse::<usize>().ok()).unwrap_or(50);
    Args { dataset, trials }
}

#[derive(Clone)]
struct CorrData {
    m: usize,
    n: usize,
    x: Vec<f64>,
}

fn parse_f64_line(line: &str) -> Vec<f64> {
    line.split(',').filter_map(|x| x.trim().parse::<f64>().ok()).collect()
}

fn load_dataset_csv(path: &str) -> CorrData {
    let file = File::open(path).unwrap_or_else(|e| panic!("Cannot open dataset {}: {}", path, e));
    let mut reader = BufReader::new(file).lines();

    let hdr = reader.next().unwrap().unwrap();
    let p: Vec<&str> = hdr.split(',').collect();
    assert!(p.len() == 3 && p[0].trim() == "dims", "Expected dims,M,N");
    let m = p[1].trim().parse::<usize>().unwrap();
    let n = p[2].trim().parse::<usize>().unwrap();

    let tag = reader.next().unwrap().unwrap();
    assert!(tag.trim() == "X", "Expected X block");
    let mut x = Vec::with_capacity(m * n);
    for _ in 0..m {
        let line = reader.next().unwrap().unwrap();
        let row = parse_f64_line(&line);
        assert!(row.len() == n, "Row width mismatch");
        x.extend(row);
    }

    CorrData { m, n, x }
}

fn checksum(corr: &[f64]) -> f64 {
    let mut s = 0.0;
    for (i, v) in corr.iter().enumerate() {
        s += *v * (((i % 101) + 1) as f64);
    }
    s
}

fn run_kernel(data: &CorrData) -> f64 {
    let m = data.m;
    let n = data.n;
    let eps = 1.0e-8f64;

    let mut mean = vec![0.0f64; n];
    let mut stddev = vec![0.0f64; n];
    let mut norm = data.x.clone();

    for j in 0..n {
        for i in 0..m {
            mean[j] += norm[i * n + j];
        }
        mean[j] /= m as f64;
    }

    for j in 0..n {
        for i in 0..m {
            let d = norm[i * n + j] - mean[j];
            stddev[j] += d * d;
        }
        stddev[j] = (stddev[j] / (m as f64)).sqrt();
        if stddev[j] <= eps {
            stddev[j] = 1.0;
        }
    }

    let scale = (m as f64).sqrt();
    for i in 0..m {
        for j in 0..n {
            norm[i * n + j] = (norm[i * n + j] - mean[j]) / (scale * stddev[j]);
        }
    }

    let mut corr = vec![0.0f64; n * n];
    for i in 0..n {
        corr[i * n + i] = 1.0;
        for j in (i + 1)..n {
            let mut s = 0.0;
            for k in 0..m {
                s += norm[k * n + i] * norm[k * n + j];
            }
            corr[i * n + j] = s;
            corr[j * n + i] = s;
        }
    }

    checksum(&corr)
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / (v.len() as f64) }
}

fn stddev(v: &[f64]) -> f64 {
    if v.len() < 2 { return 0.0; }
    let m = mean(v);
    let var = v.iter().map(|x| { let d = *x - m; d * d }).sum::<f64>() / (v.len() as f64);
    var.sqrt()
}

fn main() {
    let args = parse_args();
    let data = load_dataset_csv(&args.dataset);
    let mut results = Vec::with_capacity(args.trials);
    for _ in 0..args.trials {
        results.push(run_kernel(&data));
    }
    println!("{}", json!({
        "result_mean": mean(&results),
        "result_std": stddev(&results),
        "trials_k": args.trials,
        "kernel": "correlation",
        "m": data.m,
        "n": data.n
    }));
}
