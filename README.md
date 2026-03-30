# WASM Runtime Benchmark Suite

### End-to-End Time, Energy, and Workload Metrics for ML & HPC Kernels.

---

## Overview

This repository presents a **comprehensive benchmarking framework** for evaluating **WebAssembly (WASM) runtimes** across both:

*  **Machine Learning workloads (4 models)**
* **Arithmetic/HPC kernels (4 benchmarks)**

The framework measures:

*  **End-to-end execution time (wall-clock)**
* **Energy consumption**

  * Total energy
  * Baseline (idle) energy
  * Dynamic energy = Total − Baseline
* **Workload-specific metrics**

  * ML: Accuracy, Precision, Recall, F1-score
  * Arithmetic: Result consistency (mean/std)

---

## Benchmarks Included

### Machine Learning (ML)

* Logistic Regression (LR)
* K-Means Clustering
* Naïve Bayes (NB)
* Shallow Neural Network (SNN)

### Arithmetic / HPC Kernels

* Matrix Multiplication
* 2MM (Two Matrix Multiplications)
* Correlation
* Covariance

---

## Runtimes Evaluated

A total of **14 execution backends**:

* **Wasmtime**: JIT, AOT, Winch
* **Wasmer**: Cranelift, LLVM, V8
* **WasmEdge**: Interpreter, AOT
* **WAMR**: Interpreter, Fast-JIT, AOT
* **Others**: Wazero, Wasm3, Wasmi
* **Baseline**: Native execution

---

## Methodology

### Two-Level Execution Design

#### 1. Inner Loop (Workload Level)

* Each program execution runs a **fixed number of trials (K)** inside Rust
* Example: `K = 50`
* Ensures:

  * Stable ML metrics
  * Deterministic workload size
* ❗ No early stopping or CI inside model code

#### 2. Outer Loop (Energy Measurement Level)

* Repeated executions to achieve **statistical confidence**
* Stops when:

  * **95% confidence interval (CI)** meets target precision (e.g., ±2.5%)

---

### Energy Measurement Pipeline

Each outer run performs:

1. Measure **idle baseline power** (≥ 10s)
2. Start wattmeter logging
3. Execute workload (WASM/native)
4. Stop wattmeter
5. Compute:

   * Total Energy (integration over time)
   * Baseline Energy (idle power × runtime)
   * **Dynamic Energy = Total − Baseline**
6. Apply cooldown (≥ 10s)

✔ Uses **pre + post idle averaging** for accurate baseline estimation
✔ Ensures **full wattmeter coverage** of execution window

---

## Prerequisites

* Linux system (tested on Fedora HPC servers)
* Python ≥ 3.9
* Rust toolchain
* WASI target:

  ```bash
  rustup target add wasm32-wasip1
  ```
* Installed runtimes:

  * Wasmtime, Wasmer, WasmEdge, WAMR, Wazero, Wasm3, Wasmi
* External wattmeter (serial interface)

### Python Dependencies

```bash
pip3 install numpy pandas
```

---

##  Build Instructions

### Compile WASM Modules

```bash
cargo build --release --target wasm32-wasip1
```

Example output:

```
target/wasm32-wasip1/release/<binary>.wasm
```

### Optional: AOT Compilation

#### Wasmtime

```bash
wasmtime compile input.wasm -o output.cwasm
```

#### WasmEdge

```bash
wasmedge compile input.wasm output.so
```

#### WAMR

```bash
wamrc -o output.aot input.wasm
```

---

##  Dataset Format

CSV structure:

* First column → ID
* Middle columns → Features
* Last column → Label (ML only)
* Header row required

---

## Running Benchmarks

### Example (ML)

```bash
python3 modelMetricsPlusEnergy.py \
  --runtime wasmtime_aot \
  --dataset data/breastCancer_200000.csv \
  --trials 50 \
  --repeat-min 20 \
  --repeat-max 60 \
  --rel-precision 0.025 \
  --idle-window-s 10 \
  --cooldown-s 10 \
  --warmup 1
```

### Example (Arithmetic)

```bash
python3 modelMetricsPlusEnergy_arith.py \
  --runtime wasmtime_jit \
  --dataset data/matrix_mul_200K.csv \
  --trials 50 \
  --repeat-min 20 \
  --repeat-max 60 \
  --rel-precision 0.025 \
  --idle-window-s 10 \
  --cooldown-s 10
```

---

## Key Parameters# WASM Runtime Benchmark Suite

### End-to-End Time, Energy, and Workload Metrics for ML & HPC Kernels.

---

## Overview

This repository presents a **comprehensive benchmarking framework** for evaluating **WebAssembly (WASM) runtimes** across both:

*  **Machine Learning workloads (4 models)**
* **Arithmetic/HPC kernels (4 benchmarks)**

The framework measures:

*  **End-to-end execution time (wall-clock)**
* **Energy consumption**

  * Total energy
  * Baseline (idle) energy
  * Dynamic energy = Total − Baseline
* **Workload-specific metrics**

  * ML: Accuracy, Precision, Recall, F1-score
  * Arithmetic: Result consistency (mean/std)

---

## Benchmarks Included

### Machine Learning (ML)

* Logistic Regression (LR)
* K-Means Clustering
* Naïve Bayes (NB)
* Shallow Neural Network (SNN)

### Arithmetic / HPC Kernels

* Matrix Multiplication
* 2MM (Two Matrix Multiplications)
* Correlation
* Covariance

---

## Runtimes Evaluated

A total of **14 execution backends**:

* **Wasmtime**: JIT, AOT, Winch
* **Wasmer**: Cranelift, LLVM, V8
* **WasmEdge**: Interpreter, AOT
* **WAMR**: Interpreter, Fast-JIT, AOT
* **Others**: Wazero, Wasm3, Wasmi
* **Baseline**: Native execution

---

## Methodology

### Two-Level Execution Design

#### 1. Inner Loop (Workload Level)

* Each program execution runs a **fixed number of trials (K)** inside Rust
* Example: `K = 50`
* Ensures:

  * Stable ML metrics
  * Deterministic workload size
* ❗ No early stopping or CI inside model code

#### 2. Outer Loop (Energy Measurement Level)

* Repeated executions to achieve **statistical confidence**
* Stops when:

  * **95% confidence interval (CI)** meets target precision (e.g., ±2.5%)

---

### ⚡ Energy Measurement Pipeline

Each outer run performs:

1. Measure **idle baseline power** (≥ 10s)
2. Start wattmeter logging
3. Execute workload (WASM/native)
4. Stop wattmeter
5. Compute:

   * Total Energy (integration over time)
   * Baseline Energy (idle power × runtime)
   * **Dynamic Energy = Total − Baseline**
6. Apply cooldown (≥ 10s)

✔ Uses **pre + post idle averaging** for accurate baseline estimation
✔ Ensures **full wattmeter coverage** of execution window

---

## Prerequisites

* Linux system (tested on Fedora HPC servers)
* Python ≥ 3.9
* Rust toolchain
* WASI target:

  ```bash
  rustup target add wasm32-wasip1
  ```
* Installed runtimes:

  * Wasmtime, Wasmer, WasmEdge, WAMR, Wazero, Wasm3, Wasmi
* External wattmeter (serial interface)

### Python Dependencies

```bash
pip3 install numpy pandas
```

---

##  Build Instructions

### Compile WASM Modules

```bash
cargo build --release --target wasm32-wasip1
```

Example output:

```
target/wasm32-wasip1/release/<binary>.wasm
```

### Optional: AOT Compilation

#### Wasmtime

```bash
wasmtime compile input.wasm -o output.cwasm
```

#### WasmEdge

```bash
wasmedge compile input.wasm output.so
```

#### WAMR

```bash
wamrc -o output.aot input.wasm
```

---

##  Dataset Format

CSV structure:

* First column → ID
* Middle columns → Features
* Last column → Label (ML only)
* Header row required

---

## Running Benchmarks

### Example (ML)

```bash
python3 modelMetricsPlusEnergy.py \
  --runtime wasmtime_aot \
  --dataset data/breastCancer_200000.csv \
  --trials 50 \
  --repeat-min 20 \
  --repeat-max 60 \
  --rel-precision 0.025 \
  --idle-window-s 10 \
  --cooldown-s 10 \
  --warmup 1
```

### Example (Arithmetic)

```bash
python3 modelMetricsPlusEnergy_arith.py \
  --runtime wasmtime_jit \
  --dataset data/matrix_mul_200K.csv \
  --trials 50 \
  --repeat-min 20 \
  --repeat-max 60 \
  --rel-precision 0.025 \
  --idle-window-s 10 \
  --cooldown-s 10
```

---

## Key Parameters

| Parameter         | Description                               |
| ----------------- | ----------------------------------------- |
| `--trials`        | Fixed internal workload (K trials)        |
| `--repeat-min`    | Minimum outer runs                        |
| `--repeat-max`    | Maximum outer runs                        |
| `--rel-precision` | Target CI precision (e.g., 0.025 = ±2.5%) |
| `--idle-window-s` | Idle baseline duration                    |
| `--cooldown-s`    | Cooling time between runs                 |

---

## Output Structure

Results stored in:

```
energy_metrics_unified/
```

Each run contains:

* `final.json`
* `final.csv`
* `logs/` (raw wattmeter traces)

### Example Output Fields

#### Common:

* `execution_time_s`
* `dynamic_energy_j`
* `idle_pre_w_mean`
* `idle_post_w_mean`
* `outer_runs`
* `rel_ci_halfwidth_95_dyn_energy`
* `trials_k`

#### ML-specific:

* `accuracy`
* `precision`
* `recall`
* `f1_score`

#### Arithmetic-specific:

* `result_mean`
* `result_std`

---

## Key Contributions

* ✅ Unified benchmarking of **ML + HPC workloads**
* ✅ **Energy-aware evaluation** with statistical guarantees
* ✅ Comparison across **14 WASM runtimes + native**
* ✅ Strict **end-to-end measurement methodology**
* ✅ Portable across **cloud and edge environments**

---

## Research Relevance

This framework supports investigation of:

* WASM vs Native performance trade-offs
* Runtime-level energy efficiency
* Suitability of WASM for **ML inference and HPC workloads**
* Foundations for **multi-tenant WASM-based AI systems**

---

## License

© 2026 HPC-NEXUS Lab

| Parameter         | Description                               |
| ----------------- | ----------------------------------------- |
| `--trials`        | Fixed internal workload (K trials)        |
| `--repeat-min`    | Minimum outer runs                        |
| `--repeat-max`    | Maximum outer runs                        |
| `--rel-precision` | Target CI precision (e.g., 0.025 = ±2.5%) |
| `--idle-window-s` | Idle baseline duration                    |
| `--cooldown-s`    | Cooling time between runs                 |

---

## Output Structure

Results stored in:

```
energy_metrics_unified/
```

Each run contains:

* `final.json`
* `final.csv`
* `logs/` (raw wattmeter traces)

### Example Output Fields

#### Common:

* `execution_time_s`
* `dynamic_energy_j`
* `idle_pre_w_mean`
* `idle_post_w_mean`
* `outer_runs`
* `rel_ci_halfwidth_95_dyn_energy`
* `trials_k`

#### ML-specific:

* `accuracy`
* `precision`
* `recall`
* `f1_score`

#### Arithmetic-specific:

* `result_mean`
* `result_std`

---

## Key Contributions

* ✅ Unified benchmarking of **ML + HPC workloads**
* ✅ **Energy-aware evaluation** with statistical guarantees
* ✅ Comparison across **14 WASM runtimes + native**
* ✅ Strict **end-to-end measurement methodology**
* ✅ Portable across **cloud and edge environments**

---

## Research Relevance

This framework supports investigation of:

* WASM vs Native performance trade-offs
* Runtime-level energy efficiency
* Suitability of WASM for **ML inference and HPC workloads**
* Foundations for **multi-tenant WASM-based AI systems**

---

## License

© 2026 HPC-NEXUS Lab
