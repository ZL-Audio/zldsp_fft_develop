# zldsp_fft

zldsp_fft aims at FFT implementation and analysis.

## Usage

Please make sure `Clang` (`AppleClang 16+` or `LLVM/Clang 17+`), `cmake` (minimum 3.20) are installed and configured on your OS.

You may need to edit the building commands in `benchmark/build_config.py`.

### Accuracy Benchmark

#### CFFT Accuracy Benchmark

```console
python3 benchmark/accuracy_mid.py <n0> <n1> <--avx2> <--double>
```

Run the `fftw3`, `kfr`, `zldsp`, `pffft` from order `n0` to `n1`.

Example

```console
python3 benchmark/accuracy_mid.py 5 10

Running accuracy_mid benchmark from order 5 to 10 (10 reps each)...
Order      kfr                fftw3              zldsp              pffft             
--------------------------------------------------------------------------------------
5          2.39176700e-14     1.17213200e-13     3.95274100e-14     1.63440100e-13     
6          1.64665400e-13     2.15260600e-13     1.08373000e-13     3.01948200e-13     
7          5.78072000e-13     5.63297200e-13     4.81077900e-13     8.61865300e-13     
8          1.05322900e-12     1.27926900e-12     8.51770200e-13     2.17439600e-12     
9          3.05149800e-12     2.91082200e-12     2.65517300e-12     5.28153300e-12     
10         6.62576900e-12     6.73661400e-12     6.31371100e-12     1.16500700e-11 
```

#### CFFT Accuracy Benchmark

```console
python3 benchmark/accuracy_cfft.py <n0> <n1> zldsp <--avx2> <--double>
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

- `zldsp`


External libraries:

- `fftw3`
- `fftw3_estimate`
- `kfr`
- `pffft`
- `ipp`
- `vdsp`

## License

zldsp_fft is licensed under Apache-2.0 license, as found in the [LICENSE.md](LICENSE.md) file.

All external libraries (submodules) are not covered by this license. All trademarks, product names, and company names are the property of their respective owners and are used for identification purposes only. Please refer to the individual licenses within each submodule:

- FFTW3: `fftw3_impl/fftw3` (GPL-2.0 license)
- KFR: `kfr_impl/kfr` (GPL-2.0 license)
- pffft: `pffft_impl/pffft` (BSD-like license)
- Google benchmark: `google/benchmark` (Apache-2.0 license)
- Google highway: `google/highway` (Apache-2.0 license)