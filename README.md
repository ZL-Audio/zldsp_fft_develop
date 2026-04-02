# zlfft

zlfft aims at FFT implementation and analysis.

## Usage

Please make sure `Clang` (`AppleClang 16+` or `LLVM/Clang 17+`), `cmake` (minimum 3.20) are installed and configured on your OS.

You may need to edit the building commands in `benchmark/build_config.py`.

### Accuracy Benchmark

```console
python3 benchmark/accuracy.py <n0> <n1> <algorithm> <--avx2> <--double>
```

Run the `algorithm` forward method and compares the result with a naive Stockham implementation, from order `n0` to `n1`.

Example:

```console
python3 benchmark/accuracy.py 5 20 naive_cooley_radix2
...
Running accuracy benchmark for naive_cooley_radix2 from order 16 to 20...
Order      MSE               
----------------------------
16         1.03460500e-09    
17         2.25209500e-09    
18         4.90519100e-09    
19         1.06781200e-08    
20         2.29328400e-08 
```

### Throughput Benchmark

```console
python3 benchmark/throughput.py <n0> <n1> <algorithm> <--avx2> <--double>
```

Run the `algorithm` forward method and calculates the throughput, from order `n0` to `n1`.

Example:

```console
python3 benchmark/throughput.py 16 20 naive_cooley_radix2
...
Running throughput benchmark for naive_cooley_radix2 from order 16 to 20...
Order      Time (us)       Throughput (MFLOPS) 
---------------------------------------------
16         1550.5708       3381.2580           
17         3222.9900       3456.7653           
18         7010.0217       3365.6044           
19         13948.0256      3570.9255           
20         25840.5833      4057.8651    
```

### Algorithms

Naive algorithms, which rely on compiler optimization:

- `naive_stockham_radix2`
- `naive_cooley_radix2`
- `naive_stockham_radix4`
- `stockham_radix2_kernel24`
- `stockham_radix2_kernel248`

SIMD-accelerated algorithms, which utilize Google Highway:

- `simd_stockham_radix2`
- `simd_stockham_radix2_kernel1` 
- `simd_stockham_radix2_kernel2`
- `simd_stockham_radix2_kernel24`
- `simd_stockham_radix4`
- see more in `benchmark/build_config.py`

Optimized algorithm:

- `zldsp`

External libraries:

- `fftw3`
- `fftw3_estimate`
- `kfr`
- `pffft`
- `ipp`
- `vdsp`

## License

zlfft is licensed under Apache-2.0 license, as found in the [LICENSE.md](LICENSE.md) file.

All external libraries (submodules) are not covered by this license. All trademarks, product names, and company names are the property of their respective owners and are used for identification purposes only. Please refer to the individual licenses within each submodule:

- FFTW3: `fftw3_impl/fftw3` (GPL-2.0 license)
- KFR: `kfr_impl/kfr` (GPL-2.0 license)
- pffft: `pffft_impl/pffft` (BSD-like license)
- Google benchmark: `google/benchmark` (Apache-2.0 license)
- Google highway: `google/highway` (Apache-2.0 license)