# WASIMLarth Benchmarking Suite.   
### End-to-End Time, Energy, and Workload Metrics for ML & HPC Kernels

---

## Overview

A unified benchmarking framework to evaluate **WebAssembly (WASM) runtimes** across:

-  Machine Learning workloads (4 models)  
-  Arithmetic / HPC kernels (4 benchmarks)  

### Metrics Captured
-  Execution Time (wall-clock)
-  Energy Consumption
  - Total
  - Baseline (idle)
  - Dynamic (Total − Baseline)
-  Workload Metrics
  - ML: Accuracy, Precision, Recall, F1-score  
  - Arithmetic: Result consistency (mean, std)

---
##  Benchmarks

### Machine Learning
- Logistic Regression (LR)  
- K-Means Clustering  
- Naïve Bayes (NB)  
- Decision Tree (DT)

### Arithmetic / HPC
- Matrix Multiplication  
- 2MM (Two Matrix Multiplications)  
- Correlation  
- Covariance  

---

##  Runtimes Evaluated

**14 execution backends:**

- Wasmtime (JIT, AOT, Winch)  
- Wasmer (Cranelift, LLVM, V8)  
- WasmEdge (Interpreter, AOT)  
- WAMR (Interpreter, Fast-JIT, AOT)  
- Wazero, Wasm3, Wasmi  
- Native (baseline)

---

## Methodology

### Two-Level Execution Model

**Inner Loop (Workload Level)**
- Fixed trials per execution (e.g., K = 50)
- Ensures deterministic workload and stable metrics

**Outer Loop (Measurement Level)**
- Repeated runs until statistical convergence
- Stops when 95% CI meets target precision (±2.5%)

---

### Energy Measurement Pipeline

1. Idle baseline measurement (≥10s)
2. Start wattmeter logging
3. Execute workload
4. Stop logging
5. Compute:
   - Total energy
   - Baseline energy
   - Dynamic energy = Total − Baseline
6. Cooldown (≥10s)

✔ Pre + post idle averaging  
✔ Full measurement coverage ensured  

---

## Prerequisites

- Linux system  
- Python ≥ 3.9  
- Rust toolchain  
- WASI target:
  ```bash
  rustup target add wasm32-wasip1
  ```
- Installed runtimes (Wasmtime, Wasmer, WasmEdge, WAMR, etc.)  
- External wattmeter  

### Python Dependencies
```bash
pip3 install numpy pandas
```

---

## 🔧 Build

```bash
cargo build --release --target wasm32-wasip1
```

### Optional AOT

```bash
wasmtime compile input.wasm -o output.cwasm
wasmedge compile input.wasm output.so
wamrc -o output.aot input.wasm
```

---

## Dataset Format

- First column: ID  
- Middle: Features  
- Last: Label (ML only)  
- Header required  

---

## Run Benchmarks

### ML
```bash
python3 modelMetricsPlusEnergy.py   --runtime wasmtime_aot   --dataset data/breastCancer_200000.csv   --trials 50   --repeat-min 20   --repeat-max 60   --rel-precision 0.025   --idle-window-s 10   --cooldown-s 10   --warmup 1
```

### Arithmetic
```bash
python3 modelMetricsPlusEnergy_arith.py   --runtime wasmtime_jit   --dataset data/matrix_mul_200K.csv   --trials 50   --repeat-min 20   --repeat-max 60   --rel-precision 0.025   --idle-window-s 10   --cooldown-s 10
```

---

## 📊 Key Parameters

| Parameter         | Description |
|------------------|------------|
| `--trials`        | Workload size |
| `--repeat-min`    | Minimum runs |
| `--repeat-max`    | Maximum runs |
| `--rel-precision` | CI precision |
| `--idle-window-s` | Baseline duration |
| `--cooldown-s`    | Cooldown time |

---

##  Output

```
energy_metrics_unified/
```

Includes:
- final.json  
- final.csv  
- logs/

---
## Contributions

- Unified ML + HPC benchmarking  
- Energy-aware evaluation with statistical guarantees  
- 14 runtimes + native comparison  
- Reproducible methodology  
- Cloud + edge portability  

---

##  Research Scope

- WASM vs Native performance  
- Energy efficiency across runtimes  
- ML/HPC suitability in WASM  
- Multi-tenant WASM AI systems  

---

##  License

© 2026 HPC-NEXUS Lab
