# Intel AMX in Practice: Why Embedding Generation Can Become ~2× Faster

> **Example hardware:** AWS EC2 `c7i.2xlarge` (4th Gen Intel Xeon Sapphire Rapids)

---

# A Common Misconception

Intel AMX **does not make the entire CPU 2× faster.**

It accelerates workloads dominated by **large matrix multiplications (GEMM)**, including:

- Transformer embeddings
- LLM inference
- Computer Vision models
- Recommendation models

Everything else in your application still executes on the CPU as usual:

- Tokenization
- Python execution
- Memory access
- Networking
- Data loading
- Scheduling

Therefore, the overall speedup depends on **how much of your workload is spent performing matrix multiplication.**

---

# Example Workload

Suppose you're generating embeddings for **500 customer reviews**.

## Input

| Parameter | Value |
|-----------|------:|
| Model | `BAAI/bge-large-en-v1.5` |
| Reviews | 500 |
| Average review length | ~250 words |
| Tokens per review | ~320 |
| Embedding dimension | 1024 |
| Batch size | 32 |
| Framework | PyTorch + oneDNN |
| CPU | AWS `c7i.2xlarge` |
| Processor | 4th Gen Intel Xeon (Sapphire Rapids) |
| Intel AMX | ✅ Supported |

---

# Case 1 — Intel AMX Disabled

Without AMX, transformer matrix multiplications execute using **AVX-512 vector instructions**.

Approximate throughput

```text
40–60 reviews/sec
```

Assume

```text
50 reviews/sec
```

Processing time

```text
500 reviews
──────────────
50 reviews/sec

≈ 10 seconds
```

---

# Case 2 — Intel AMX Enabled

With AMX enabled, the transformer's large **GEMM (General Matrix Multiplication)** operations execute using dedicated **AMX BF16 tile instructions**.

Typical improvement

```text
1.5× – 2.5×
```

depending on:

- Batch size
- Model architecture
- Precision (BF16 vs FP32)
- Software backend

Assume the workload achieves **2× higher throughput**.

```text
100 reviews/sec
```

Processing time

```text
500 reviews
──────────────
100 reviews/sec

≈ 5 seconds
```

---

# Where Did the 5 Seconds Go?

The entire workload wasn't accelerated.

Only the expensive matrix multiplications became faster.

## Without AMX

| Stage | Time |
|------|------:|
| Tokenization | 0.8 s |
| Embedding lookup | 0.2 s |
| Matrix multiplications | **7.5 s** |
| LayerNorm + GELU + Softmax | 0.8 s |
| Memory overhead | 0.7 s |
| **Total** | **10.0 s** |

Notice that nearly

```text
75%
```

of the total execution time is spent performing matrix multiplication.

---

## With AMX

Only this stage changes.

```text
Matrix Multiplication

7.5 seconds
      │
      ▼
3.0 seconds
```

Everything else remains approximately the same.

| Stage | Time |
|------|------:|
| Tokenization | 0.8 s |
| Embedding lookup | 0.2 s |
| Matrix multiplications | **3.0 s** |
| LayerNorm + GELU + Softmax | 0.8 s |
| Memory overhead | 0.7 s |
| **Total** | **5.5 s** |

---

This demonstrates an important performance principle:

> Even if AMX makes matrix multiplication more than **2× faster**, the **end-to-end application** usually speeds up by **less than 2×**, because tokenization, memory access, and other stages are unaffected.

---

# Why Batch Size Matters

Intel AMX is designed for **large matrix operations**.

Larger batches create larger GEMM operations, allowing the hardware to remain fully utilized.

| Batch Size | Without AMX | With AMX |
|-----------:|------------:|---------:|
| 1 | 20 ms/review | 16 ms/review |
| 8 | 10 ms/review | 6 ms/review |
| 32 | 8 ms/review | 4 ms/review |
| 64 | 7 ms/review | 3 ms/review |

## Key Observation

### Small batches

- Lower AMX utilization
- Smaller GEMM operations
- More overhead relative to compute
- Smaller performance gains

### Large batches

- Larger matrix multiplications
- Better tile utilization
- Higher throughput
- Better overall efficiency

---

# Why This Matters for RAG Pipelines

Embedding generation occurs:

- During document ingestion
- During indexing
- Sometimes during query processing

Large-scale ingestion benefits significantly from higher embedding throughput.

Example:

```text
1,000,000 reviews
```

## Without AMX

```text
50 reviews/sec

↓

20,000 seconds

↓

≈ 5.5 hours
```

## With AMX

```text
100 reviews/sec

↓

10,000 seconds

↓

≈ 2.8 hours
```

Result

```text
Nearly 3 hours saved
```

for a single embedding job.

At production scale, this directly reduces:

- Infrastructure cost
- Indexing time
- Data refresh latency

---

# The Technical Nuances Most Benchmarks Skip

## 1. AMX Requires BF16 or INT8

One of the most common misconceptions is that **AMX accelerates every floating-point workload.**

It doesn't.

AMX is designed primarily for:

- **BF16 (Bfloat16)**
- **INT8**

If your model runs in standard **FP32**, the workload typically falls back to **AVX-512**, meaning you won't realize most of AMX's performance benefits.

### Why?

Transformer inference spends most of its time in GEMM operations.

AMX provides dedicated tile instructions specifically optimized for BF16 and INT8 matrix multiplication.

FP32 matrix multiplication still executes using traditional vector instructions.

### Practical implication

```text
FP32 model
      ↓
AVX-512
      ↓
Little or no AMX benefit
```

```text
BF16 model
      ↓
AMX Tile Instructions
      ↓
1.5×–2.5× higher throughput (typical)
```

For this reason, production inference on Sapphire Rapids is commonly deployed using **BF16**, which offers nearly the numerical stability of FP32 while significantly improving throughput.

---

## 2. AMX Is Not Completely Plug-and-Play

Owning an AMX-capable CPU does **not** automatically guarantee AMX acceleration.

Your software stack must dispatch the transformer's matrix multiplications to optimized AMX kernels.

Common optimization stacks include:

- Intel Extension for PyTorch (IPEX)
- oneDNN
- OpenVINO
- ONNX Runtime
- `torch.compile()` (when supported)

Without these optimized backends, standard PyTorch may execute GEMM using AVX-512 instead of AMX, leaving much of the available performance untapped.

Think of it as three layers:

```text
Hardware
    │
    ▼
Intel AMX
    │
    ▼
Optimized Runtime
    │
    ▼
Your Embedding Model
```

Missing the middle layer often means **the hardware cannot be fully utilized**.

---

## 3. CPU vs GPU Depends on the Workload

The previous example compares:

```text
CPU without AMX
        vs
CPU with AMX
```

It is **not** a comparison between CPUs and GPUs.

That's an important distinction.

### For Real-Time RAG

Typical request sizes are small:

- Batch size = 1
- Batch size = 4
- Batch size = 8
- Batch size = 16

In this range, CPUs with AMX can provide:

- Low latency
- Simpler deployment
- Lower infrastructure cost
- No PCIe transfer overhead
- Good enough throughput for online inference

This makes them an attractive option for many production RAG systems.

---

### For Offline Indexing

Now consider processing:

```text
500

5,000

500,000

5,000,000
```

documents.

These workloads are dominated by throughput rather than latency.

Modern GPUs contain thousands of parallel compute cores and much higher memory bandwidth, allowing them to process very large batches far more efficiently than CPUs.

For large offline embedding jobs, GPUs usually deliver the highest raw throughput.

A useful rule of thumb is:

| Workload | Best Fit |
|-----------|----------|
| Real-time user queries | CPU + AMX |
| Moderate online serving | CPU + AMX |
| Massive offline indexing | GPU |
| High-throughput embedding farms | GPU |

The right choice ultimately depends on your latency target, throughput requirements, infrastructure cost, and operational complexity.

---

# Are These Numbers Realistic?

Yes.

Benchmarks from Intel and independent evaluations on **Sapphire Rapids** processors consistently report approximately:

```text
1.5× – 2.5×
```

higher throughput for BF16 transformer inference when using optimized software stacks.

Actual improvements depend on several factors.

| Factor | Impact |
|---------|--------|
| Precision | BF16 benefits the most. FP32 gains are much smaller. |
| Model size | Larger transformer models spend more time in GEMM and benefit more. |
| Batch size | Larger batches improve AMX utilization. |
| Software backend | IPEX, oneDNN, OpenVINO, and ONNX Runtime expose optimized AMX kernels. |
| Memory bandwidth | Can become a bottleneck for some models. |

---

# Practical Takeaway

For an optimized BF16 embedding workload running on a `c7i.2xlarge`, reducing embedding latency from roughly

```text
10 seconds
```

to approximately

```text
5–6 seconds
```

for **500 reviews** is a realistic expectation.

However, the exact improvement depends on:

- Model architecture
- Precision
- Batch size
- Software stack
- Memory behavior

A safer production expectation is:

```text
30–60%
```

lower end-to-end latency rather than assuming a guaranteed **2× application speedup**.

---

# Key Takeaways

- Intel AMX accelerates **matrix multiplication**, not the entire application.
- AMX delivers its biggest gains with **BF16** and **INT8** workloads.
- Running a model in FP32 typically falls back to AVX-512 and limits AMX acceleration.
- Optimized runtimes such as **IPEX**, **oneDNN**, **OpenVINO**, or **ONNX Runtime** are usually required to fully utilize AMX.
- Transformer inference is dominated by GEMM operations, making it an ideal workload for AMX.
- Larger batches create larger matrix multiplications and improve AMX utilization.
- CPUs with AMX excel for low-latency, real-time RAG inference.
- GPUs generally remain the best choice for very large offline embedding jobs where maximum throughput is the priority.
- For large embedding pipelines, AMX can significantly reduce indexing time and infrastructure costs without changing the model itself.
