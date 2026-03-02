#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statistics as stats
import math
from statistics import mean, stdev


# ---- Wattmeter settings ----
WATTMETER_BIN = "/home/skhan/energyWork/wattmetre-read/v3/wattmetre-readnew"
WATTMETER_TTY = "/dev/ttyUSB0"
WATTMETER_NB  = "6"


# ---------------- Wattmeter helpers ----------------
def start_meter(csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(csv_path, "w", buffering=1)
    proc = subprocess.Popen(
        [WATTMETER_BIN, f"--tty={WATTMETER_TTY}", f"--nb={WATTMETER_NB}"],
        stdout=fh,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return proc, fh


def stop_meter(proc, fh):
    try:
        proc.terminate()
        proc.wait(timeout=3)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        fh.flush()
        try:
            os.fsync(fh.fileno())
        except Exception:
            pass
    except Exception:
        pass
    try:
        fh.close()
    except Exception:
        pass


def _read_series(csv_file: Path) -> pd.DataFrame:
    if not csv_file.exists() or csv_file.stat().st_size == 0:
        return pd.DataFrame(columns=["ts", "p"])

    try:
        df = pd.read_csv(csv_file, engine="python", on_bad_lines="skip")
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["ts", "p"])

    if df.empty or "#timestamp" not in df.columns:
        return pd.DataFrame(columns=["ts", "p"])

    ts = pd.to_numeric(df["#timestamp"], errors="coerce")
    df = df[ts.notna()].copy()
    df["ts"] = pd.to_numeric(df["#timestamp"], errors="coerce").astype(float)

    p1 = pd.to_numeric(df.get("#activepow1"), errors="coerce")
    p5 = pd.to_numeric(df.get("#activepow5"), errors="coerce")
    df["p"] = (p1.fillna(0.0) + p5.fillna(0.0)).astype(float)

    return df[["ts", "p"]].dropna().sort_values("ts").reset_index(drop=True)


def integrate_energy(df: pd.DataFrame, t_start: float, t_end: float):
    if t_end <= t_start:
        return 0.0, 0.0, 0

    w = df[(df["ts"] >= t_start) & (df["ts"] <= t_end)].copy()
    if w.empty:
        return 0.0, float(t_end - t_start), 0

    ts = w["ts"].to_numpy(dtype=float)
    p = w["p"].to_numpy(dtype=float)

    dt = np.empty_like(ts)
    if len(ts) == 1:
        dt[0] = max(0.0, float(t_end - ts[0]))
    else:
        dt[:-1] = np.clip(np.diff(ts), 0, 10)
        dt[-1] = max(0.0, float(t_end - ts[-1]))  # hold until t_end

    energy = float(np.sum(p * dt))
    duration = float(t_end - t_start)
    return energy, duration, int(len(ts))


def mean_power_from_idle(csv_path: Path, seconds: float) -> float:
    proc, fh = start_meter(csv_path)
    time.sleep(seconds)
    stop_meter(proc, fh)
    df = _read_series(csv_path)
    if df.empty:
        return 0.0
    return float(df["p"].mean())


# ---------------- Runtime helpers ----------------
def run_cmd(cmd, cwd=None):
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, (out or "") + "\n" + (err or "")


def extract_first_json_anywhere(text: str):
    # quick per-line
    for line in text.splitlines():
        if "{" in line and "}" in line:
            start = line.find("{")
            end = line.rfind("}")
            cand = line[start:end + 1]
            try:
                obj = json.loads(cand)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

    # multi-line scan (bounded)
    starts = [i for i, ch in enumerate(text) if ch == "{"][:60]
    for s in starts:
        for e in range(len(text) - 1, s, -1):
            if text[e] != "}":
                continue
            cand = text[s:e + 1]
            try:
                obj = json.loads(cand)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None


def build_runtime_cmd(runtime: str, root: Path, wasm: Path, cwasm: Path, wasm_args: list[str]):
    home = Path.home()

    if runtime == "wasmtime_jit":
        return [str(home/".wasmtime/bin/wasmtime"), "run",
                "--dir", f"{root}::/work",
                str(wasm), "--"] + wasm_args

    if runtime == "wasmtime_aot":
        return [str(home/".wasmtime/bin/wasmtime"), "run",
                "--allow-precompiled",
                "--dir", f"{root}::/work",
                str(cwasm), "--"] + wasm_args

    if runtime == "wasmer_cranelift":
        return [str(home/".wasmer/bin/wasmer"), "run",
                "--cranelift", str(wasm),
                "--mapdir", f"/work:{root}",
                "--"] + wasm_args

    if runtime == "wasmer_llvm":
        return [str(home/".wasmer/bin/wasmer"), "run",
                "--llvm", str(wasm),
                "--mapdir", f"/work:{root}",
                "--"] + wasm_args

    if runtime == "wasmer_v8":
        return [str(home/".wasmer/bin/wasmer"), "run",
                "--v8", str(wasm),
                "--mapdir", f"/work:{root}",
                "--"] + wasm_args

    if runtime == "wasmedge":
        return [str(home/".wasmedge/bin/wasmedge"),
                "--dir", f"/work:{root}",
                str(wasm), "--"] + wasm_args

    if runtime == "wazero":
        wazero = root / "wazero/bin/wazero"
        return [str(wazero), "run",
                "--mount", f"{root}:/work",
                str(wasm),
                "--"] + wasm_args

    if runtime == "wamr_iwasm":
        iwasm = root / "WAMR/wasm-micro-runtime/product-mini/platforms/linux/build/iwasm"
        return [str(iwasm),
                f"--map-dir=/data::{str(root/'data')}",
                str(wasm)] + wasm_args

    raise SystemExit(f"Unknown runtime: {runtime}")


def safe_mean(x, default=0.0):
    return float(stats.mean(x)) if x else float(default)


# ---------------- Outer precision control (95% CI) ----------------
def t_critical_95(n: int) -> float:
    # 95% two-sided t critical values, conservative for small n
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
        18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042,
        40: 2.021, 60: 2.000, 120: 1.980
    }
    df = max(1, n - 1)
    keys = sorted(table.keys())
    for k in keys:
        if df <= k:
            return table[k]
    return 1.960


def rel_ci_halfwidth_95(samples: list[float]) -> float:
    """
    Relative CI half-width = half_width / |mean|.
    Returns inf if n<2 or mean ~0.
    """
    n = len(samples)
    if n < 2:
        return float("inf")
    m = mean(samples)
    if abs(m) < 1e-12:
        return float("inf")
    s = stdev(samples)
    t = t_critical_95(n)
    half = t * (s / math.sqrt(n))
    return abs(half / m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", required=True,
                    choices=["wasmtime_jit","wasmtime_aot","wasmer_cranelift","wasmer_llvm","wasmer_v8",
                             "wasmedge","wazero","wamr_iwasm"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.001)

    # Fixed internal workload K
    ap.add_argument("--trials", type=int, default=50)

    # Outer loop controls (kept names compatible with your command)
    ap.add_argument("--repeat-min", type=int, default=20)   # minimum outer runs
    ap.add_argument("--repeat-max", type=int, default=60)   # maximum outer runs
    ap.add_argument("--rel-precision", type=float, default=0.025)  # target relative CI half-width (95%)

    ap.add_argument("--idle-window-s", type=float, default=10.0)
    ap.add_argument("--cooldown-s", type=float, default=10.0)
    ap.add_argument("--warmup", type=int, default=1)

    ap.add_argument("--outdir", type=str, default="energy_metrics_min")

    args = ap.parse_args()

    root = Path.cwd()
    wasm  = root / "mlRust-wasi/target/wasm32-wasip1/release/ml_wasi_lr.wasm"
    cwasm = root / "ml_wasi_lr.cwasm"

    if not wasm.exists():
        raise SystemExit(f"WASM not found: {wasm}")

    # dataset mount
    host_dataset = Path(args.dataset)
    if not host_dataset.is_absolute():
        host_dataset = (root / host_dataset).resolve()
    rel_from_root = host_dataset.relative_to(root)

    guest_work = f"/work/{rel_from_root.as_posix()}"
    guest_data = f"/data/{host_dataset.name}"
    dataset_guest = guest_data if args.runtime == "wamr_iwasm" else guest_work

    K = int(args.trials)

    # NOTE: Pass fixed K through min/max-runs for backward compatibility with Rust parser.
    wasm_args = [
        "--datasets", dataset_guest,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--min-runs", str(K),
        "--max-runs", str(K),
        "--rel-precision", "0.0",  # ignored by Rust (kept for compatibility)
    ]

    cmd = build_runtime_cmd(args.runtime, root, wasm, cwasm, wasm_args)

    outdir = (root / args.outdir / args.runtime / host_dataset.stem)
    logs = outdir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    # warmup
    for i in range(args.warmup):
        rc, out = run_cmd(cmd, cwd=str(root))
        if rc != 0:
            (outdir / "warmup_failed.txt").write_text(out)
            raise SystemExit(f"Warmup failed for {args.runtime}. See {outdir/'warmup_failed.txt'}")
        print(f"[{args.runtime}] warmup {i+1}/{args.warmup}", flush=True)
        time.sleep(args.cooldown_s)

    dyn_list, dyn_pre_list, dyn_post_list = [], [], []
    idle_pre_w_list, idle_post_w_list, idle_avg_w_list = [], [], []
    time_list = []
    acc_list, prec_list, rec_list, f1_list = [], [], [], []

    good_runs = 0
    attempt = 0

    target = float(args.rel_precision)
    min_runs = int(args.repeat_min)
    max_runs = int(args.repeat_max)

    # Keep trying until we either:
    # - achieve precision after >= min_runs, OR
    # - hit max_runs successful runs.
    while good_runs < max_runs and attempt < max_runs * 5:
        attempt += 1
        run_id = good_runs + 1

        idle_pre_csv  = logs / f"idle_pre_run{run_id}.csv"
        idle_post_csv = logs / f"idle_post_run{run_id}.csv"
        run_csv       = logs / f"run{run_id}.csv"

        # idle power window before run (>= 10s recommended)
        idle_pre_power = mean_power_from_idle(idle_pre_csv, args.idle_window_s)

        # run energy capture
        meter_p, meter_f = start_meter(run_csv)
        time.sleep(0.2)  # guard before start

        t_start = time.time()
        rc, out = run_cmd(cmd, cwd=str(root))
        t_end = time.time()

        time.sleep(0.5)  # guard after end
        stop_meter(meter_p, meter_f)

        if rc != 0:
            (logs / f"run{run_id}_failed_out.txt").write_text(out)
            print(f"[{args.runtime}] run {run_id:03d} FAILED (rc!=0). Retrying...", flush=True)
            time.sleep(args.cooldown_s)
            continue

        df_run = _read_series(run_csv)
        if df_run.empty:
            print(f"[{args.runtime}] run {run_id:03d} empty wattmeter log. Retrying...", flush=True)
            time.sleep(args.cooldown_s)
            continue

        ts_min = float(df_run["ts"].min())
        ts_max = float(df_run["ts"].max())

        # window coverage tolerance (0.20s)
        if ts_min > (t_start + 0.20) or ts_max < (t_end - 0.20):
            print(f"[{args.runtime}] WARNING: meter did not cover full window for run {run_id} "
                  f"(ts_min={ts_min:.3f}, ts_max={ts_max:.3f}, t_start={t_start:.3f}, t_end={t_end:.3f}). Retrying.",
                  flush=True)
            time.sleep(args.cooldown_s)
            continue

        total_e, duration_s, samples = integrate_energy(df_run, t_start, t_end)
        # idle power window after run
        idle_post_power = mean_power_from_idle(idle_post_csv, args.idle_window_s)
        idle_avg_power = 0.5 * (idle_pre_power + idle_post_power)

        # dynamic energy computed using mean idle power (equivalently mean of pre/post-corrected energies)
        base_e = idle_avg_power * duration_s
        dyn_pre = total_e - (idle_pre_power * duration_s)
        dyn_post = total_e - (idle_post_power * duration_s)
        dyn_e = 0.5 * (dyn_pre + dyn_post)

        # parse metrics JSON (must exist)
        ml = extract_first_json_anywhere(out)
        if ml is None:
            (logs / f"run{run_id}_nojson_out.txt").write_text(out)
            print(f"[{args.runtime}] run {run_id:03d} missing JSON metrics. Retrying...", flush=True)
            time.sleep(args.cooldown_s)
            continue

        # store
        dyn_list.append(dyn_e)
        dyn_pre_list.append(dyn_pre)
        dyn_post_list.append(dyn_post)
        idle_pre_w_list.append(idle_pre_power)
        idle_post_w_list.append(idle_post_power)
        idle_avg_w_list.append(idle_avg_power)
        time_list.append(duration_s)

        acc_list.append(float(ml.get("accuracy_mean", ml.get("acc_mean", 0.0))))
        prec_list.append(float(ml.get("precision_mean", 0.0)))
        rec_list.append(float(ml.get("recall_mean", 0.0)))
        f1_list.append(float(ml.get("f1_mean", 0.0)))

        good_runs += 1
        print(f"[{args.runtime}] run {run_id:03d} | wall_s={duration_s:.3f} dynJ={dyn_e:.3f} (pre={dyn_pre:.3f}, post={dyn_post:.3f}) idleW_avg={idle_avg_power:.2f} samples={samples}", flush=True)

        # precision check (outer dynamic energy samples)
        if good_runs >= min_runs:
            rel_hw = rel_ci_halfwidth_95(dyn_list)
            print(f"[{args.runtime}] precision check: n={good_runs} relCI_halfwidth={rel_hw:.4f} target={target:.4f}", flush=True)
            if rel_hw <= target:
                print(f"[{args.runtime}] stopping early: precision achieved at n={good_runs}", flush=True)
                break

        time.sleep(args.cooldown_s)

    # final output
    final = {
        "execution_time_s": safe_mean(time_list),
        "dynamic_energy_j": safe_mean(dyn_list),  # mean of per-run (pre/post averaged) dynamic energy
        "dynamic_energy_pre_j": safe_mean(dyn_pre_list),
        "dynamic_energy_post_j": safe_mean(dyn_post_list),
        "idle_pre_w_mean": safe_mean(idle_pre_w_list),
        "idle_post_w_mean": safe_mean(idle_post_w_list),
        "idle_avg_w_mean": safe_mean(idle_avg_w_list),
        "accuracy": safe_mean(acc_list, default=0.0),
        "precision": safe_mean(prec_list, default=0.0),
        "recall": safe_mean(rec_list, default=0.0),
        "f1_score": safe_mean(f1_list, default=0.0),
        "outer_runs": int(good_runs),
        "rel_ci_halfwidth_95_dyn_energy": float(rel_ci_halfwidth_95(dyn_list)) if len(dyn_list) >= 2 else None,
        "rel_precision_target": float(target),
        "trials_k": int(K),
        "idle_window_s": float(args.idle_window_s),
        "cooldown_s": float(args.cooldown_s),
    }

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "final.json").write_text(json.dumps(final, indent=2) + "\n")
    pd.DataFrame([final]).to_csv(outdir / "final.csv", index=False)

    print("\n=== FINAL OUTPUT ===")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
