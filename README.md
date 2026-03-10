**WASM Runtime Benchmark: End-to-End Time + Energy + Model Metrics**

**OVERVIEW**

This project benchmarks a WASI-compatible Rust ML multiple workloads across multiple WebAssembly runtimes and measures:

-   End-to-end execution time (wall-clock)
-   Energy (Total, Baseline, Dynamic = Total − Baseline)
-   Model metrics (Accuracy, Precision, Recall, F1)

**Methodology**:

1.  Fixed internal workload: Each program execution runs exactly K
    trials (e.g., K=50) inside Rust. No CI or early stopping exists
    inside the Rust program.

2.  Statistical precision control: Confidence interval logic is applied
    only at the outer energy level. The runner collects multiple energy
    samples and stops when the relative 95% CI half-width target (e.g.,
    ±2.5%) is achieved.

**PREREQUISITES**

-   Linux system
-   Python 3.9+
-   Rust toolchain
-   wasm32-wasip1 target
-   Installed runtimes (wasmtime, wasmer, wasmedge, wazero, wamr)
-   External wattmeter configured

**Install Python dependencies:** pip3 install numpy pandas

**BUILD RUST WASI MODULE**

cd mlRust-wasi rustup target add wasm32-wasip1 cargo build –release
–target wasm32-wasip1

**Output**: mlRust-wasi/target/wasm32-wasip1/release/ml_wasi_lr.wasm

(**Optional**) Build Wasmtime AOT: ~/.wasmtime/bin/wasmtime compile
mlRust-wasi/target/wasm32-wasip1/release/ml_wasi_lr.wasm -o
ml_wasi_lr.cwasm

**DATASET FORMAT**

CSV format: - First column: ID - Middle columns: numeric features - Last
column: label (e.g., M/B) - First row: header

**RUN BENCHMARK**

Example command:

python3 modelMetricsPlusEnergy.py –runtime wasmtime_aot –dataset
data/breastCancer_200000.csv –trials 50 –repeat-min 20 –repeat-max 60
–rel-precision 0.025 –idle-window-s 10 –cooldown-s 10 –warmup 1

Parameter Meaning: - –trials 50 -> Fixed internal workload K=50 -
–repeat-min 20 -> Minimum outer energy runs - –repeat-max 60 -> Maximum
outer energy runs - –rel-precision 0.025 -> ±2.5% relative 95% CI
target - –idle-window-s 10 -> Baseline idle window - –cooldown-s 10 ->
Cooling period between runs

**OUTPUT
**
Results are stored in:

energy_metrics_min///

Files include: - final.json - final.csv - logs/ (raw wattmeter logs)

Each final output includes: - execution_time_s - dynamic_energy_j -
accuracy - precision - recall - f1_score - outer_runs -
rel_ci_halfwidth_95_dyn_energy - trials_k

**METHODOLOGY SUMMARY**

Each outer run: 1. Measure baseline power (>=10s) 2. Start wattmeter 3.
Execute WASM program (fixed K trials) 4. Stop wattmeter 5. Compute total
energy 6. Compute dynamic energy = total − baseline 7. Cooldown >=10s

Repeat until statistical precision is achieved or maximum outer runs is
reached.

**LICENSE**

@HPC-NEXUS LAB 2026
