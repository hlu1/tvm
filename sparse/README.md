# Benchmark deep&wide

Run benchmark on skylake(AVX2):
```
numactl --physcpubind=<NODE> --localalloc -- python sparse/deep_wide_relay.py --device skl
```

Use command `lscpu` to find the number of numa nodes and the corresponding online cpu list
