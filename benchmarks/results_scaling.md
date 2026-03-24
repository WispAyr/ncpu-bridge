# nCPU Scaling Benchmark Results

## Width Scaling (8 / 16 / 32-bit via set_width)

| Operation | Width | Mean (ms) | Ops/sec |
|-----------|-------|-----------|---------|
| ADD | 8-bit | 0.342 | 2,925 |
| ADD | 16-bit | 0.338 | 2,955 |
| ADD | 32-bit | 0.337 | 2,966 |
| MUL | 8-bit | 0.017 | 57,874 |
| MUL | 16-bit | 0.021 | 47,346 |
| MUL | 32-bit | 0.021 | 47,812 |
| XOR | 8-bit | 0.032 | 31,590 |
| XOR | 16-bit | 0.032 | 31,600 |
| XOR | 32-bit | 0.032 | 31,626 |

## ONNX Batch Scaling (arithmetic model)

| Batch Size | Latency (ms) | Effective Ops/sec | Speedup vs batch=1 |
|------------|-------------|-------------------|---------------------|
| 1 | 0.004 | 236,783 | 1.0x |
| 10 | 0.006 | 1,602,842 | 6.8x |
| 50 | 0.014 | 3,515,061 | 14.8x |
| 100 | 0.020 | 4,948,157 | 20.9x |
| 500 | 0.076 | 6,562,007 | 27.7x |
| 1000 | 0.092 | 10,848,492 | 45.8x |

## Sort Scaling (Bubble Sort, neural CMP)

| Input Size | Time (ms) | Neural CMPs (~n²) | Sorts/sec |
|------------|-----------|-------------------|-----------|
| 3 | 1.1 | ~9 | 872.7 |
| 5 | 3.5 | ~25 | 285.6 |
| 8 | 9.7 | ~64 | 103.1 |
| 10 | 15.6 | ~100 | 64.2 |
| 15 | 36.3 | ~225 | 27.6 |
| 20 | 65.8 | ~400 | 15.2 |