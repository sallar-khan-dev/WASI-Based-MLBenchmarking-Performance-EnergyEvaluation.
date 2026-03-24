#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics as stats
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd

# ---- Wattmeter settings ----
WATTMETER_BIN = "/home/skhan/energyWork/wattmetre-read/v3/wattmetre-readnew"
WATTMETER_TTY = "/dev/ttyUSB0"
WATTMETER_NB = "6"

# ---- Meter timing pads ----
METER_START_PAD_S = 1.0
METER_END_PAD_S = 1.0
WAIT_FIRST_SAMPLE_TIMEOUT_S = 4.0
WAIT_FIRST_SAMPLE_POLL_S = 0.10
METER_COVERAGE_TOLERANCE_S = 0.50

RUNTIME_CHOICES = [
    "wasmtime_jit",
    "wasmtime_aot",
    "wasmtime_winch",
    "wasmer_cranelift",
    "wasmer_llvm",
    "wasmer_v8",
    "wasmedge",
    "wazero",
    "wamr_interp",
    "wamr_fast_interp",
    "wamr_iwasm",
    "wamr_aot",
    "wasm3",
    "wasmi",
    "native",
]

BENCH_CHOICES = ["matrix_mul", "2mm", "correlation", "covariance"]


def which_or_none(path_or_name: str) -> str | None:
    p = Path(path_or_name)
    if p.is_absolute() or "/" in path_or_name:
        return str(p) if p.exists() else None
    return shutil.which(path_or_name)


def first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def resolve_runtime_binary(candidates: list[str | Path], label: str) -> str:
    for c in candidates:
        s = str(c)
        found = which_or_none(s)
        if found:
            return found
    raise SystemExit(
        f"Could not find binary for {label}. Checked: "
        + ", ".join(str(c) for c in candidates)
    )


def json_dump(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def start_meter(csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(csv_path, "w", buffering=1)
    proc = subprocess.Popen(
        [
            "stdbuf",
            "-oL",
            "-eL",
            WATTMETER_BIN,
            f"--tty={WATTMETER_TTY}",
            f"--nb={WATTMETER_NB}",
        ],
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
    p1 = pd.to_numeric(df.get("#activepow1"), errors="coerce")
    p5 = pd.to_numeric(df.get("#activepow5"), errors="coerce")
    p = p1.fillna(0.0) + p5.fillna(0.0)

    mask = ts.notna() & p.notna()
    out = pd.DataFrame({"ts": ts[mask].astype(float), "p": p[mask].astype(float)})
    if out.empty:
        return pd.DataFrame(columns=["ts", "p"])
    return out.sort_values("ts").reset_index(drop=True)


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


def mean_power_from_idle(csv_path: Path, seconds: float) -> float:
    proc, fh = start_meter(csv_path)
    time.sleep(seconds)
    stop_meter(proc, fh)
    df = _read_series(csv_path)
    if df.empty:
        return 0.0
    return float(df["p"].mean())


def wait_for_first_valid_sample(csv_path: Path, timeout_s: float = WAIT_FIRST_SAMPLE_TIMEOUT_S) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        df = _read_series(csv_path)
        if not df.empty:
            return True
        time.sleep(WAIT_FIRST_SAMPLE_POLL_S)
    return False


def run_cmd(cmd: list[str], cwd: str | None = None):
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, (out or "") + "\n" + (err or "")


def extract_first_json_anywhere(text: str):
    for line in text.splitlines():
        if "{" in line and "}" in line:
            start = line.find("{")
            end = line.rfind("}")
            cand = line[start : end + 1]
            try:
                obj = json.loads(cand)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

    starts = [i for i, ch in enumerate(text) if ch == "{"][:60]
    for s in starts:
        for e in range(len(text) - 1, s, -1):
            if text[e] != "}":
                continue
            cand = text[s : e + 1]
            try:
                obj = json.loads(cand)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return None


def runtime_artifact_names(bench: str) -> tuple[str, str, str]:
    wasm_name = f"poly_{bench}.wasm"
    cwasm_name = f"poly_{bench}.cwasm"
    native_name = f"poly_{bench}"
    return wasm_name, cwasm_name, native_name


def build_runtime_cmd(runtime: str, root: Path, wasm: Path, cwasm: Path | None, wasm_args: list[str], native_name: str) -> list[str]:
    home = Path.home()

    if runtime == "wasmtime_jit":
        wasmtime = resolve_runtime_binary([home / ".wasmtime/bin/wasmtime", "wasmtime"], "wasmtime")
        return [wasmtime, "run", "--dir", f"{root}::/work", str(wasm), "--"] + wasm_args

    if runtime == "wasmtime_aot":
        if cwasm is None or not cwasm.exists():
            raise SystemExit("wasmtime_aot selected but precompiled cwasm file was not found.")
        wasmtime = resolve_runtime_binary([home / ".wasmtime/bin/wasmtime", "wasmtime"], "wasmtime")
        return [wasmtime, "run", "--allow-precompiled", "--dir", f"{root}::/work", str(cwasm), "--"] + wasm_args

    if runtime == "wasmtime_winch":
        runner = home / "wasmWork/wasiWork/wasi/fourthPaper_MLRust/wasmtime_winch_runner/target/release/wasmtime_winch_runner"
        if not runner.exists():
            raise SystemExit(f"Wasmtime Winch runner not found: {runner}")
        return [str(runner), str(wasm)] + wasm_args

    if runtime == "wasmer_cranelift":
        wasmer = resolve_runtime_binary([home / ".wasmer/bin/wasmer", "wasmer"], "wasmer")
        return [wasmer, "run", "--cranelift", str(wasm), "--mapdir", f"/work:{root}", "--"] + wasm_args

    if runtime == "wasmer_llvm":
        wasmer = resolve_runtime_binary([home / ".wasmer/bin/wasmer", "wasmer"], "wasmer")
        return [wasmer, "run", "--llvm", str(wasm), "--mapdir", f"/work:{root}", "--"] + wasm_args

    if runtime == "wasmer_v8":
        wasmer = resolve_runtime_binary([home / ".wasmer/bin/wasmer", "wasmer"], "wasmer")
        return [wasmer, "run", "--v8", str(wasm), "--mapdir", f"/work:{root}", "--"] + wasm_args

    if runtime == "wasmedge":
        wasmedge = resolve_runtime_binary([home / ".wasmedge/bin/wasmedge", "wasmedge"], "wasmedge")
        return [wasmedge, "--dir", f"/work:{root}", str(wasm), "--"] + wasm_args

    if runtime == "wazero":
        wazero = resolve_runtime_binary([root / "wazero/bin/wazero", "wazero"], "wazero")
        return [wazero, "run", "--mount", f"{root}:/work", str(wasm), "--"] + wasm_args

    if runtime == "wasmi":
        wasmi = resolve_runtime_binary(
            [home / ".cargo/bin/wasmi_cli", home / ".cargo/bin/wasmi", "wasmi_cli", "wasmi"],
            "wasmi",
        )
        return [wasmi, "--dir", ".", str(wasm), "--"] + wasm_args

    if runtime in ("wamr_interp", "wamr_fast_interp", "wamr_iwasm", "wamr_aot"):
        iwasm = resolve_runtime_binary(
            [
                root / "WAMR/wasm-micro-runtime/product-mini/platforms/linux/build_fastjit/iwasm",
                root / "WAMR/wasm-micro-runtime/product-mini/platforms/linux/build/iwasm",
                root / "WAMR/wasm-micro-runtime/product-mini/platforms/linux/build/bin/iwasm",
                "iwasm",
            ],
            "iwasm",
        )

        if runtime == "wamr_aot":
            aot_candidates = [
                root / f"{Path(native_name).stem}.aot",
                root / f"{Path(wasm).stem}.aot",
                root / f"{Path(wasm).stem}_wamr.aot",
                root / "target/wasm32-wasip1/release" / f"{Path(wasm).stem}.aot",
            ]
            aot = first_existing(aot_candidates)
            if aot is None:
                raise SystemExit("WAMR AOT file not found. Build it first with wamrc and rerun wamr_aot.")
            return [iwasm, f"--map-dir=/data::{str((root / 'data').resolve())}", str(aot)] + wasm_args

        if runtime == "wamr_fast_interp":
            return [iwasm, "--fast-jit", "--map-dir=/data::" + str((root / "data").resolve()), str(wasm)] + wasm_args

        return [iwasm, f"--map-dir=/data::{str((root / 'data').resolve())}", str(wasm)] + wasm_args

    if runtime == "wasm3":
        wasm3 = resolve_runtime_binary(["wasm3", "m3"], "wasm3")
        # wasm3 should receive a host-relative dataset path like data/foo.csv,
        # not a guest path like /work/data/foo.csv.
        ds = wasm_args[1]
        if isinstance(ds, str) and ds.startswith("/work/"):
            ds = ds[len("/work/"):]
        elif isinstance(ds, str) and ds.startswith("/data/"):
            ds = ds[len("/"):]
        elif isinstance(ds, str) and Path(ds).is_absolute():
            try:
                ds = str(Path(ds).resolve().relative_to(root.resolve()))
            except Exception:
                ds = Path(ds).name
        wasm3_args = ["--dataset", str(ds), "--trials", str(wasm_args[3])]
        return [wasm3, str(wasm)] + wasm3_args

    if runtime == "native":
        native_candidates = [
            root / native_name,
            root / "target/release" / native_name,
            root / "native/target/release" / native_name,
            root / "target/native-release" / native_name,
        ]
        nb = first_existing(native_candidates)
        if nb is None:
            raise SystemExit(f"Native binary not found: {native_name}")
        return [str(nb)] + wasm_args

    raise SystemExit(f"Unknown runtime: {runtime}")


def safe_mean(x, default=0.0):
    return float(stats.mean(x)) if x else float(default)


def t_critical_95(n: int) -> float:
    table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
        13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
        19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980,
    }
    df = max(1, n - 1)
    for k in sorted(table):
        if df <= k:
            return table[k]
    return 1.960


def rel_ci_halfwidth_95(samples: list[float]) -> float:
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
    ap.add_argument("--runtime", required=True, choices=RUNTIME_CHOICES)
    ap.add_argument("--benchmark", required=True, choices=BENCH_CHOICES)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--repeat-min", type=int, default=20)
    ap.add_argument("--repeat-max", type=int, default=60)
    ap.add_argument("--rel-precision", type=float, default=0.025)
    ap.add_argument("--idle-window-s", type=float, default=10.0)
    ap.add_argument("--cooldown-s", type=float, default=10.0)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--outdir", type=str, default="energy_metrics_arith")
    ap.add_argument("--wasm", type=str, default=None)
    ap.add_argument("--cwasm", type=str, default=None)
    args = ap.parse_args()

    root = Path.cwd().resolve()
    default_wasm_name, default_cwasm_name, default_native_name = runtime_artifact_names(args.benchmark)

    wasm_arg = args.wasm or f"target/wasm32-wasip1/release/{default_wasm_name}"
    cwasm_arg = args.cwasm or default_cwasm_name

    wasm = (root / wasm_arg).resolve() if not Path(wasm_arg).is_absolute() else Path(wasm_arg).resolve()
    cwasm = (root / cwasm_arg).resolve() if not Path(cwasm_arg).is_absolute() else Path(cwasm_arg).resolve()

    if args.runtime != "native" and not wasm.exists():
        raise SystemExit(f"WASM not found: {wasm}")

    host_dataset = Path(args.dataset)
    if not host_dataset.is_absolute():
        host_dataset = (root / host_dataset).resolve()
    if not host_dataset.exists():
        raise SystemExit(f"Dataset not found: {host_dataset}")

    try:
        rel_from_root = host_dataset.relative_to(root)
    except ValueError:
        raise SystemExit(
            f"Dataset must be inside the project root so a stable guest path can be constructed.\n"
            f"root={root}\n"
            f"dataset={host_dataset}"
        )

    guest_work = f"/work/{rel_from_root.as_posix()}"
    guest_data = f"/data/{host_dataset.name}"

    if args.runtime == "native":
        dataset_guest = str(host_dataset)
    elif args.runtime in ("wamr_interp", "wamr_fast_interp", "wamr_iwasm", "wamr_aot"):
        dataset_guest = guest_data
    elif args.runtime == "wasmi":
        dataset_guest = rel_from_root.as_posix()
    else:
        dataset_guest = guest_work

    k_trials = int(args.trials)
    common_args = [
        "--dataset", dataset_guest,
        "--trials", str(k_trials),
    ]

    cmd = build_runtime_cmd(args.runtime, root, wasm, cwasm, common_args, default_native_name)

    outdir = root / args.outdir / args.benchmark / args.runtime / host_dataset.stem
    logs = outdir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    metadata = {
        "runtime": args.runtime,
        "benchmark": args.benchmark,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": str(root),
        "wasm_path": str(wasm),
        "cwasm_path": str(cwasm),
        "dataset_host_path": str(host_dataset),
        "dataset_guest_path": dataset_guest,
        "trials_k": int(k_trials),
        "repeat_min": int(args.repeat_min),
        "repeat_max": int(args.repeat_max),
        "rel_precision_target": float(args.rel_precision),
        "idle_window_s": float(args.idle_window_s),
        "cooldown_s": float(args.cooldown_s),
        "warmup": int(args.warmup),
        "meter_start_pad_s": float(METER_START_PAD_S),
        "meter_end_pad_s": float(METER_END_PAD_S),
        "meter_coverage_tolerance_s": float(METER_COVERAGE_TOLERANCE_S),
        "command": cmd,
    }
    json_dump(outdir / "metadata.json", metadata)

    for i in range(args.warmup):
        rc, out = run_cmd(cmd, cwd=str(root))
        if rc != 0:
            (outdir / "warmup_failed.txt").write_text(out)
            raise SystemExit(f"Warmup failed for {args.runtime}. See {outdir/'warmup_failed.txt'}")
        print(f"[{args.runtime}] warmup {i + 1}/{args.warmup}", flush=True)
        time.sleep(args.cooldown_s)

    dyn_list, dyn_pre_list, dyn_post_list = [], [], []
    idle_pre_w_list, idle_post_w_list, idle_avg_w_list = [], [], []
    time_list, result_means, result_stds = [], [], []

    good_runs = 0
    attempt = 0
    target = float(args.rel_precision)
    min_runs = int(args.repeat_min)
    max_runs = int(args.repeat_max)

    while good_runs < max_runs and attempt < max_runs * 5:
        attempt += 1
        run_id = good_runs + 1

        idle_pre_csv = logs / f"idle_pre_run{run_id}.csv"
        idle_post_csv = logs / f"idle_post_run{run_id}.csv"
        run_csv = logs / f"run{run_id}.csv"

        idle_pre_power = mean_power_from_idle(idle_pre_csv, args.idle_window_s)
        meter_p, meter_f = start_meter(run_csv)

        if not wait_for_first_valid_sample(run_csv, timeout_s=WAIT_FIRST_SAMPLE_TIMEOUT_S):
            stop_meter(meter_p, meter_f)
            print(f"[{args.runtime}] run {run_id:03d} no valid wattmeter samples yet. Retrying...", flush=True)
            time.sleep(args.cooldown_s)
            continue

        time.sleep(METER_START_PAD_S)
        t_start = time.time()
        rc, out = run_cmd(cmd, cwd=str(root))
        t_end = time.time()
        time.sleep(METER_END_PAD_S)
        stop_meter(meter_p, meter_f)

        if rc != 0:
            (logs / f"run{run_id}_failed_out.txt").write_text(out)
            print(f"[{args.runtime}] run {run_id:03d} failed (rc={rc}). Retrying...", flush=True)
            time.sleep(args.cooldown_s)
            continue

        df_run = _read_series(run_csv)
        if df_run.empty:
            print(f"[{args.runtime}] run {run_id:03d} empty wattmeter log. Retrying...", flush=True)
            time.sleep(args.cooldown_s)
            continue

        ts_min = float(df_run["ts"].min())
        ts_max = float(df_run["ts"].max())
        if ts_min > (t_start + METER_COVERAGE_TOLERANCE_S) or ts_max < (t_end - METER_COVERAGE_TOLERANCE_S):
            print(
                f"[{args.runtime}] WARNING: meter did not cover full window for run {run_id} "
                f"(ts_min={ts_min:.3f}, ts_max={ts_max:.3f}, t_start={t_start:.3f}, t_end={t_end:.3f}). Retrying.",
                flush=True,
            )
            time.sleep(args.cooldown_s)
            continue

        total_e, duration_s, samples = integrate_energy(df_run, t_start, t_end)
        idle_post_power = mean_power_from_idle(idle_post_csv, args.idle_window_s)
        idle_avg_power = 0.5 * (idle_pre_power + idle_post_power)
        dyn_pre = total_e - (idle_pre_power * duration_s)
        dyn_post = total_e - (idle_post_power * duration_s)
        dyn_e = 0.5 * (dyn_pre + dyn_post)

        bench = extract_first_json_anywhere(out)
        if bench is None:
            (logs / f"run{run_id}_nojson_out.txt").write_text(out)
            print(f"[{args.runtime}] run {run_id:03d} missing JSON metrics. Retrying...", flush=True)
            time.sleep(args.cooldown_s)
            continue

        dyn_list.append(float(dyn_e))
        dyn_pre_list.append(float(dyn_pre))
        dyn_post_list.append(float(dyn_post))
        idle_pre_w_list.append(float(idle_pre_power))
        idle_post_w_list.append(float(idle_post_power))
        idle_avg_w_list.append(float(idle_avg_power))
        time_list.append(float(duration_s))
        result_means.append(float(bench.get("result_mean", bench.get("checksum_mean", 0.0))))
        result_stds.append(float(bench.get("result_std", 0.0)))

        good_runs += 1
        print(
            f"[{args.runtime}] run {run_id:03d} | wall_s={duration_s:.3f} dynJ={dyn_e:.3f} "
            f"(pre={dyn_pre:.3f}, post={dyn_post:.3f}) idleW_avg={idle_avg_power:.2f} samples={samples}",
            flush=True,
        )

        if good_runs >= min_runs:
            rel_hw = rel_ci_halfwidth_95(dyn_list)
            print(
                f"[{args.runtime}] precision check: n={good_runs} relCI_halfwidth={rel_hw:.4f} target={target:.4f}",
                flush=True,
            )
            if rel_hw <= target:
                print(f"[{args.runtime}] stopping early: precision achieved at n={good_runs}", flush=True)
                break

        time.sleep(args.cooldown_s)

    final = {
        "execution_time_s": safe_mean(time_list),
        "dynamic_energy_j": safe_mean(dyn_list),
        "dynamic_energy_pre_j": safe_mean(dyn_pre_list),
        "dynamic_energy_post_j": safe_mean(dyn_post_list),
        "idle_pre_w_mean": safe_mean(idle_pre_w_list),
        "idle_post_w_mean": safe_mean(idle_post_w_list),
        "idle_avg_w_mean": safe_mean(idle_avg_w_list),
        "result_mean": safe_mean(result_means),
        "result_std": safe_mean(result_stds),
        "outer_runs": int(good_runs),
        "rel_ci_halfwidth_95_dyn_energy": float(rel_ci_halfwidth_95(dyn_list)) if len(dyn_list) >= 2 else None,
        "rel_precision_target": float(target),
        "trials_k": int(k_trials),
        "idle_window_s": float(args.idle_window_s),
        "cooldown_s": float(args.cooldown_s),
    }

    json_dump(outdir / "final.json", final)
    pd.DataFrame([final]).to_csv(outdir / "final.csv", index=False)

    print("\n=== FINAL OUTPUT ===")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
