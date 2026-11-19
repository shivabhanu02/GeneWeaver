# GeneWeaver — Day 11 GPU Benchmark

**Hardware:** NVIDIA Tesla T4  
**Numba:** 0.64.0  
**CUDA:** Available

## Objective
Evaluate CPU and CUDA Smith-Waterman performance across different candidate batch sizes and DNA sequence lengths.

## Benchmark configuration
- Candidate batch sizes: 32, 128, 256
- Sequence lengths: 128 bp, 256 bp
- Match score: +2
- Mismatch penalty: -1
- Gap penalty: -2
- CPU repetitions: 3
- GPU repetitions: 5
- Initial CUDA JIT compilation excluded from GPU timing.

GPU timing includes device allocation, data transfers, kernel execution and result transfer.

## Results

| Length (bp) | Batch | DP cells | CPU average (ms) | GPU average (ms) | Speedup | Correctness |
|---:|---:|---:|---:|---:|---:|---|
| 128 | 32 | 524,288 | 197.851 | 7.021 | 28.18x | PASS |
| 128 | 128 | 2,097,152 | 532.048 | 6.936 | 76.70x | PASS |
| 128 | 256 | 4,194,304 | 1048.935 | 7.825 | 134.05x | PASS |
| 256 | 32 | 2,097,152 | 1042.473 | 31.946 | 32.63x | PASS |
| 256 | 128 | 8,388,608 | 2041.783 | 31.042 | 65.77x | PASS |
| 256 | 256 | 16,777,216 | 4566.959 | 32.274 | 141.50x | PASS |

## Observations

- All six configurations completed successfully.
- CPU and GPU scores matched exactly in every configuration.
- The highest measured speedup was 141.50x at 256 candidates and 256 bp.
- Larger batches showed substantial GPU performance benefits in this workload.
- CUDA grid-size warnings indicate possible low occupancy for smaller batches.

## Conclusion

The CUDA implementation successfully processed all tested configurations with exact score agreement against the CPU baseline. Performance increased substantially for larger workloads on the NVIDIA Tesla T4.

These measurements are specific to the tested implementation, hardware and workload.
