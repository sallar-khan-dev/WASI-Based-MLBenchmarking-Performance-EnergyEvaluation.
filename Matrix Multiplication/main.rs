use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};

use serde_json::json;

#[derive(Clone, Debug)]
struct Args {
    dataset: String,
    trials: usize,
}

fn get_arg_value(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|x| x == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn parse_args() -> Args {
    let args: Vec<String> = env::args().collect();
    let dataset = get_arg_value(&args, "--dataset")
        .or_else(|| get_arg_value(&args, "--datasets"))
        .unwrap_or_else(|| "data/matrix_mul_100K.csv".to_string());
    let trials = get_arg_value(&args, "--trials")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(50);
    Args { dataset, trials }
}

#[derive(Clone)]
struct MatrixMulData {
    n: usize,
    a: Vec<f64>,
    b: Vec<f64>,
}

fn parse_f64_line(line: &str) -> Vec<f64> {
    line.split(',').filter_map(|x| x.trim().parse::<f64>().ok()).collect()
}

fn load_dataset_csv(path: &str) -> MatrixMulData {
    let file = File::open(path).unwrap_or_else(|e| panic!("Cannot open dataset {}: {}", path, e));
    let mut reader = BufReader::new(file).lines();

    let header = reader.next().unwrap().unwrap();
    let parts: Vec<&str> = header.split(',').collect();
    assert!(parts.len() == 2 && parts[0].trim() == "n", "Expected header n,<value>");
    let n = parts[1].trim().parse::<usize>().unwrap();

    let tag_a = reader.next().unwrap().unwrap();
    assert!(tag_a.trim() == "A", "Expected A row");
    let mut a = Vec::with_capacity(n * n);
    for _ in 0..n {
        let line = reader.next().unwrap().unwrap();
        let row = parse_f64_line(&line);
        assert!(row.len() == n, "A row length mismatch");
        a.extend(row);
    }

    let tag_b = reader.next().unwrap().unwrap();
    assert!(tag_b.trim() == "B", "Expected B row");
    let mut b = Vec::with_capacity(n * n);
    for _ in 0..n {
        let line = reader.next().unwrap().unwrap();
        let row = parse_f64_line(&line);
        assert!(row.len() == n, "B row length mismatch");
        b.extend(row);
    }

    MatrixMulData { n, a, b }
}

fn checksum(c: &[f64]) -> f64 {
    let mut s = 0.0;
    for (i, v) in c.iter().enumerate() {
        s += *v * (((i % 97) + 1) as f64);
    }
    s
}

fn run_kernel(data: &MatrixMulData) -> f64 {
    let n = data.n;
    let mut c = vec![0.0f64; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = data.a[i * n + k];
            for j in 0..n {
                c[i * n + j] += aik * data.b[k * n + j];
            }
        }
    }
    checksum(&c)
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / (v.len() as f64) }
}

fn stddev(v: &[f64]) -> f64 {
    if v.len() < 2 { return 0.0; }
    let m = mean(v);
    let var = v.iter().map(|x| {
        let d = *x - m;
        d * d
    }).sum::<f64>() / (v.len() as f64);
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
        "kernel": "matrix_mul",
        "n": data.n
    }));
}
