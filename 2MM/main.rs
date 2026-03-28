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
        .unwrap_or_else(|| "data/2mm_100K.csv".to_string());
    let trials = get_arg_value(&args, "--trials").and_then(|v| v.parse::<usize>().ok()).unwrap_or(50);
    Args { dataset, trials }
}

#[derive(Clone)]
struct TwoMMData {
    ni: usize,
    nj: usize,
    nk: usize,
    nl: usize,
    alpha: f64,
    beta: f64,
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    d: Vec<f64>,
}

fn parse_f64_line(line: &str) -> Vec<f64> {
    line.split(',').filter_map(|x| x.trim().parse::<f64>().ok()).collect()
}

fn load_block(reader: &mut impl Iterator<Item = std::io::Result<String>>, tag: &str, rows: usize, cols: usize) -> Vec<f64> {
    let hdr = reader.next().unwrap().unwrap();
    assert!(hdr.trim() == tag, "Expected block {}", tag);
    let mut out = Vec::with_capacity(rows * cols);
    for _ in 0..rows {
        let line = reader.next().unwrap().unwrap();
        let row = parse_f64_line(&line);
        assert!(row.len() == cols, "Block {} width mismatch", tag);
        out.extend(row);
    }
    out
}

fn load_dataset_csv(path: &str) -> TwoMMData {
    let file = File::open(path).unwrap_or_else(|e| panic!("Cannot open dataset {}: {}", path, e));
    let mut reader = BufReader::new(file).lines();

    let dims = reader.next().unwrap().unwrap();
    let p: Vec<&str> = dims.split(',').collect();
    assert!(p.len() == 5 && p[0].trim() == "dims", "Expected dims,NI,NJ,NK,NL");
    let ni = p[1].trim().parse::<usize>().unwrap();
    let nj = p[2].trim().parse::<usize>().unwrap();
    let nk = p[3].trim().parse::<usize>().unwrap();
    let nl = p[4].trim().parse::<usize>().unwrap();

    let coeffs = reader.next().unwrap().unwrap();
    let c: Vec<&str> = coeffs.split(',').collect();
    assert!(c.len() == 3 && c[0].trim() == "coeff", "Expected coeff,alpha,beta");
    let alpha = c[1].trim().parse::<f64>().unwrap();
    let beta = c[2].trim().parse::<f64>().unwrap();

    let a = load_block(&mut reader, "A", ni, nk);
    let b = load_block(&mut reader, "B", nk, nj);
    let cmat = load_block(&mut reader, "C", nj, nl);
    let d = load_block(&mut reader, "D", ni, nl);

    TwoMMData { ni, nj, nk, nl, alpha, beta, a, b, c: cmat, d }
}

fn checksum(x: &[f64]) -> f64 {
    let mut s = 0.0;
    for (i, v) in x.iter().enumerate() {
        s += *v * (((i % 89) + 1) as f64);
    }
    s
}

fn run_kernel(data: &TwoMMData) -> f64 {
    let mut tmp = vec![0.0f64; data.ni * data.nj];
    let mut out = data.d.clone();

    for i in 0..data.ni {
        for k in 0..data.nk {
            let a_ik = data.a[i * data.nk + k];
            for j in 0..data.nj {
                tmp[i * data.nj + j] += data.alpha * a_ik * data.b[k * data.nj + j];
            }
        }
    }

    for i in 0..data.ni {
        for j in 0..data.nl {
            out[i * data.nl + j] *= data.beta;
        }
        for k in 0..data.nj {
            let tmp_ik = tmp[i * data.nj + k];
            for j in 0..data.nl {
                out[i * data.nl + j] += tmp_ik * data.c[k * data.nl + j];
            }
        }
    }

    checksum(&out)
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
        "kernel": "2mm",
        "ni": data.ni,
        "nj": data.nj,
        "nk": data.nk,
        "nl": data.nl
    }));
}
