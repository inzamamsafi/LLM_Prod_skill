# Intel AMX in Practice: Why Embedding Generation Can Become ~2× Faster

> **Example hardware:** AWS EC2 `c7i.2xlarge` (4th Gen Intel Xeon Sapphire Rapids)

---

## A Common Misconception

Intel AMX **does not make the entire CPU 2× faster.**

It accelerates only workloads dominated by **large matrix multiplications**, such as:

- Transformer embeddings
- LLM inference
- Computer Vision models
- Recommendation models

Everything else in your pipeline still executes on the CPU as usual:

- Tokenization
- Python execution
- Memory access
- Networking
- Data loading
- Scheduling

Therefore, the overall speedup depends on **how much of your workload is actually spent performing matrix multiplication.**

---

# Example Workload

Suppose you're generating embeddings for **500 customer reviews**.

### Input

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

depending on

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

---

## Without AMX

| Stage | Time |
|------|------:|
| Tokenization | 0.8 s |
| Embedding lookup | 0.2 s |
| Matrix multiplications | **7.5 s** |
| LayerNorm + GELU + Softmax | 0.8 s |
| Memory overhead | 0.7 s |
| **Total** | **10.0 s** |

---

Notice that

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

This explains an important performance principle:

> Even if AMX makes matrix multiplication more than **2× faster**, the **end-to-end application** usually speeds up by **less than 2×**, because the remaining stages are unaffected.

---

# Why Batch Size Matters

Intel AMX is designed for **large matrix operations**.

Larger batches create larger GEMM operations, allowing the hardware to stay fully utilized.

| Batch Size | Without AMX | With AMX |
|-----------:|------------:|---------:|
| 1 | 20 ms/review | 16 ms/review |
| 8 | 10 ms/review | 6 ms/review |
| 32 | 8 ms/review | 4 ms/review |
| 64 | 7 ms/review | 3 ms/review |

---

## Key Observation

Small batches

- Less hardware utilization
- Smaller performance gains

Large batches

- Larger matrix multiplications
- Better AMX utilization
- Higher throughput

---

# Why This Matters for RAG Pipelines

Embedding generation happens:

- During document ingestion
- During indexing
- Sometimes during query processing

Large-scale ingestion benefits significantly from higher embedding throughput.

Example:

```text
1,000,000 reviews
```

---

## Without AMX

```text
50 reviews/sec

↓

20,000 seconds

↓

≈ 5.5 hours
```

---

## With AMX

```text
100 reviews/sec

↓

10,000 seconds

↓

≈ 2.8 hours
```

---

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

# Are These Numbers Realistic?

Yes.

Benchmarks from Intel and independent evaluations on **Sapphire Rapids** processors consistently report approximately:

```text
1.5× – 2.5×
```

higher throughput for BF16 transformer inference when using optimized software stacks.

The actual improvement depends on several factors.

| Factor | Impact |
|---------|--------|
| Precision | BF16 benefits the most. FP32 gains are smaller. |
| Model size | Larger transformer models spend more time in GEMM and benefit more. |
| Batch size | Larger batches improve AMX utilization. |
| Software backend | oneDNN, Intel Extension for PyTorch, and optimized ONNX Runtime expose the fastest AMX kernels. Standard PyTorch may not. |

---

# Practical Takeaway

For an optimized workload running on a `c7i.2xlarge`, reducing embedding latency from roughly

```text
10 seconds
```

to approximately

```text
5–6 seconds
```

for **500 reviews** is a reasonable expectation.

However, the exact improvement depends on your model, precision, batch size, and software stack.

A more conservative expectation for production systems is:

```text
30–60%
```

lower end-to-end latency rather than assuming a guaranteed **2× speedup**.

---

# Key Takeaways

- Intel AMX accelerates **matrix multiplication**, not the entire application.
- Transformer inference is dominated by GEMM operations, making it an ideal workload for AMX.
- Larger batches produce larger matrices and improve AMX utilization.
- BF16 workloads typically benefit the most.
- End-to-end speedup is limited by non-accelerated stages such as tokenization and memory access.
- For large embedding jobs, AMX can significantly reduce indexing time and infrastructure costs without changing the model itself.
