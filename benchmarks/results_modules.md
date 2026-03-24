# nCPU Module Benchmark Results


## Neural Hash (CRC32)

| Operation | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Ops/sec |
|-----------|-----------|-------------|----------|----------|---------|
| hash_5byte | 2.5 | 2.5 | 2.5 | 2.5 | 399.4 |
| hash_16byte | 7.9 | 7.9 | 7.9 | 8.0 | 125.9 |
| table_build | 1017.5 | 1028.5 | 939.4 | 1084.7 | 1.0 |

## Neural Sort

| Operation | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Ops/sec |
|-----------|-----------|-------------|----------|----------|---------|
| bubble_sort_5 | 3.4 | 3.4 | 3.3 | 3.4 | 298.1 |
| selection_sort_5 | 3.3 | 3.3 | 3.3 | 3.3 | 300.0 |
| merge_sort_5 | 2.0 | 2.0 | 2.0 | 2.0 | 499.0 |
| bubble_sort_10 | 14.9 | 14.9 | 14.9 | 15.0 | 66.9 |
| selection_sort_10 | 15.0 | 14.9 | 14.9 | 15.3 | 66.7 |
| merge_sort_10 | 8.0 | 8.0 | 8.0 | 8.0 | 125.6 |
| bubble_sort_20 | 63.0 | 63.0 | 63.0 | 63.1 | 15.9 |
| selection_sort_20 | 63.0 | 63.0 | 62.9 | 63.0 | 15.9 |
| merge_sort_20 | 21.2 | 21.2 | 21.2 | 21.3 | 47.1 |

## Neural DB (SQL)

| Operation | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Ops/sec |
|-----------|-----------|-------------|----------|----------|---------|
| insert | 42.0 | 45.8 | 23.3 | 57.6 | 23.8 |
| select_all | 0.0 | 0.0 | 0.0 | 0.0 | 390915.1 |
| select_where | 7.3 | 7.3 | 7.3 | 7.4 | 136.6 |

## C Compiler (compile+execute)

| Operation | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Ops/sec |
|-----------|-----------|-------------|----------|----------|---------|
| compile_add | 1.0 | 1.0 | 1.0 | 1.0 | 1001.6 |
| compile_complex | 1.0 | 1.0 | 0.9 | 1.0 | 1020.3 |
| compile_loop | 0.2 | 0.2 | 0.2 | 0.2 | 5965.7 |

## Neural Crypto

| Operation | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Ops/sec |
|-----------|-----------|-------------|----------|----------|---------|
| kdf_8byte | 6.6 | 6.6 | 6.6 | 6.7 | 151.2 |
| encrypt_5byte | 4.3 | 4.3 | 4.3 | 4.4 | 230.7 |
| decrypt_5byte | 4.3 | 4.3 | 4.3 | 4.3 | 233.7 |