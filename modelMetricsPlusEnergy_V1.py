#!/usr/bin/env python3import argparse
//enabling new runtimes. 

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import math
from statistics import mean, stdev

WATTMETER_BIN = "/home/skhan/energyWork/wattmetre-read/v3/wattmetre-readnew"
WATTMETER_TTY = "/dev/ttyUSB0"
WATTMETER_NB  = "6"

METER_GUARD_S = 1.0


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
        os.fsync(fh.fileno())
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
        dt[-1] = max(0.0, float(t_end - ts[-1]))

    energy = float(np.sum(p * dt))
    duration = float(t_end - t_start)
    return energy, duration, int(len(ts))


def t_critical_95(n: int) -> float:
    table = {1:12.706,2:4.303,3:3.182,4:2.776,5:2.571,6:2.447,
             7:2.365,8:2.306,9:2.262,10:2.228,11:2.201,12:2.179,
             13:2.160,14:2.145,15:2.131,16:2.120,17:2.110,
             18:2.101,19:2.093,20:2.086,25:2.060,30:2.042,
             40:2.021,60:2.000,120:1.980}
    df = max(1, n - 1)
    for k in sorted(table.keys()):
        if df <= k:
            return table[k]
    return 1.960


def rel_ci_halfwidth_95(samples):
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
    ap.add_argument("--runtime", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--repeat-min", type=int, default=20)
    ap.add_argument("--repeat-max", type=int, default=60)
    ap.add_argument("--rel-precision", type=float, default=0.025)
    ap.add_argument("--idle-window-s", type=float, default=10.0)
    ap.add_argument("--cooldown-s", type=float, default=10.0)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--outdir", type=str, default="energy_metrics_min")

    args = ap.parse_args()

    root = Path.cwd()
    wasm = root / "mlRust-wasi/target/wasm32-wasip1/release/ml_wasi_lr.wasm"

    dataset_guest = f"/work/{args.dataset}"

    cmd = ["wasmtime", "run", "--dir", f"{root}::/work",
           str(wasm), "--",
           "--datasets", dataset_guest,
           "--min-runs", str(args.trials),
           "--max-runs", str(args.trials)]

    outdir = root / args.outdir / args.runtime
    logs = outdir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    dyn_list = []
    time_list = []

    good_runs = 0
    min_runs = args.repeat_min
    max_runs = args.repeat_max
    target = args.rel_precision

    while good_runs < max_runs:
        run_id = good_runs + 1
        run_csv = logs / f"run{run_id}.csv"

        meter_p, meter_f = start_meter(run_csv)

        time.sleep(args.idle_window_s)

        t_start = time.time()
        rc = subprocess.call(cmd)
        t_end = time.time()

        time.sleep(args.idle_window_s)
        stop_meter(meter_p, meter_f)

        if rc != 0:
            continue

        df = _read_series(run_csv)
        if df.empty:
            continue

        ts_min = float(df["ts"].min())
        ts_max = float(df["ts"].max())

        t_pre0 = t_start - args.idle_window_s
        t_post1 = t_end + args.idle_window_s

        if ts_min > (t_pre0 + METER_GUARD_S) or ts_max < (t_post1 - METER_GUARD_S):
            print(f"[{args.runtime}] WARNING: meter jitter too large. Retrying.")
            time.sleep(args.cooldown_s)
            continue

        pre_seg = df[(df["ts"] >= t_pre0) & (df["ts"] <= t_start)]
        post_seg = df[(df["ts"] >= t_end) & (df["ts"] <= t_post1)]

        if pre_seg.empty or post_seg.empty:
            continue

        idle_pre = float(pre_seg["p"].mean())
        idle_post = float(post_seg["p"].mean())

        total_e, duration_s, _ = integrate_energy(df, t_start, t_end)

        dyn_pre = total_e - idle_pre * duration_s
        dyn_post = total_e - idle_post * duration_s
        dyn_e = 0.5 * (dyn_pre + dyn_post)

        dyn_list.append(dyn_e)
        time_list.append(duration_s)

        good_runs += 1
        print(f"[{args.runtime}] run {run_id:03d} | dynJ={dyn_e:.3f} wall_s={duration_s:.3f}")

        if good_runs >= min_runs:
            rel_hw = rel_ci_halfwidth_95(dyn_list)
            print(f"[{args.runtime}] precision check: n={good_runs} relCI_halfwidth={rel_hw:.4f}")
            if rel_hw <= target:
                print(f"[{args.runtime}] stopping early (precision met).")
                break

        time.sleep(args.cooldown_s)

    final = {
        "execution_time_s": float(np.mean(time_list)),
        "dynamic_energy_j": float(np.mean(dyn_list)),
        "outer_runs": good_runs,
        "rel_ci_halfwidth_95_dyn_energy": float(rel_ci_halfwidth_95(dyn_list)) if len(dyn_list) >= 2 else None,
    }

    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "final.json", "w") as f:
        json.dump(final, f, indent=2)

    print("\n=== FINAL OUTPUT ===")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
