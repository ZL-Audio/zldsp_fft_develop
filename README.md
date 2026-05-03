# zldsp_fft

zldsp_fft aims at FFT implementation and analysis.

## Usage

Please make sure `Clang` (`AppleClang 16+` or `LLVM/Clang 17+`), `cmake` (minimum 3.20) are installed and configured on your OS.

```console
git clone https://github.com/ZL-Audio/zldsp_fft_develop
cd zldsp_fft_develop
git submodule update --init
```

You may need to edit the building commands in `benchmark/build_config.py`.

### Accuracy Benchmark

#### CFFT Accuracy Benchmark

Run `fftw3`, `kfr`, `zldsp`, `pffft` CFFT from order `n0` to `n1`.

```console
python3 benchmark/accuracy_mid_cfft.py <n0> <n1> <--avx2> <--double>
```

Example
```console
python3 benchmark/accuracy_mid_cfft.py 5 10
Running accuracy CFFT benchmark from order 5 to 10 (10 reps each)...
Order      kfr                fftw3              zldsp              pffft             
--------------------------------------------------------------------------------------
5          2.39176700e-14     1.17213200e-13     3.95274100e-14     1.63440100e-13     
6          1.64665400e-13     2.15260600e-13     1.08373000e-13     3.01948200e-13     
7          5.78072000e-13     5.63297200e-13     4.81077900e-13     8.61865300e-13     
8          1.05322900e-12     1.27926900e-12     8.51770200e-13     2.17439600e-12     
9          3.05149800e-12     2.91082200e-12     2.65517300e-12     5.28153300e-12     
10         6.62576900e-12     6.73661400e-12     6.31371100e-12     1.16500700e-11
```

#### CFFT Consistent Benchmark

Run `zldsp` different CFFT forward/backward APIs from order `n0` to `n1`.

```console
python3 benchmark/accuracy_cfft.py <n0> <n1> <--avx2> <--double>
```

Example
```console
python3 benchmark/accuracy_cfft.py 5 10
Running accuracy CFFT benchmark for zldsp from order 5 to 10...
Order      Fwd Max MSE        Bwd Max MSE        Id Max MSE        
------------------------------------------------------------------
5          0.00000000e+00     0.00000000e+00     4.70630500e-15    
6          0.00000000e+00     0.00000000e+00     7.68742700e-15    
7          0.00000000e+00     0.00000000e+00     1.11504800e-14    
8          0.00000000e+00     0.00000000e+00     1.19794100e-14    
9          0.00000000e+00     0.00000000e+00     1.32861400e-14    
10         0.00000000e+00     0.00000000e+00     1.51891300e-14
```

#### RFFT Accuracy Benchmark

Run `fftw3`, `kfr`, `zldsp`, `pffft` RFFT from order `n0` to `n1`.

```console
python3 benchmark/accuracy_mid_rfft.py <n0> <n1> <--avx2> <--double>
```

Example

```console
python3 benchmark/accuracy_mid_rfft.py 5 10
Running accuracy RFFT benchmark from order 5 to 10 (10 reps each)...
Order      kfr                fftw3              zldsp              pffft             
--------------------------------------------------------------------------------------
5          7.26960800e-14     3.77433200e-14     3.38191300e-14     3.27714500e-14     
6          1.63731400e-13     1.01401400e-13     9.75312900e-14     1.05184300e-13     
7          3.56730700e-13     2.27370500e-13     2.42351800e-13     2.67729800e-13     
8          1.07237800e-12     5.58322300e-13     8.09634900e-13     5.90561800e-13     
9          1.91429100e-12     1.27614100e-12     1.43604000e-12     1.51917200e-12     
10         5.22122600e-12     2.91316900e-12     3.52626300e-12     3.38997800e-12
```

#### RFFT Consistent Benchmark

Run `zldsp` different RFFT forward/backward APIs from order `n0` to `n1`.

```console
python3 benchmark/accuracy_rfft.py <n0> <n1> <--avx2> <--double>
```

Example
```console
python3 benchmark/accuracy_rfft.py 5 10
Running accuracy RFFT benchmark for zldsp from order 5 to 10...
Order      Fwd Max MSE        Bwd Max MSE        Id Max MSE        
------------------------------------------------------------------
5          0.00000000e+00     0.00000000e+00     5.30825400e-15    
6          0.00000000e+00     0.00000000e+00     7.23466400e-15    
7          0.00000000e+00     0.00000000e+00     5.84959600e-15    
8          0.00000000e+00     0.00000000e+00     8.47672600e-15    
9          0.00000000e+00     0.00000000e+00     8.37233100e-15    
10         0.00000000e+00     0.00000000e+00     9.01347400e-15
```

### Throughput Benchmark

#### CFFT Throughput Benchmark

Run `<algorithm>` CFFT throughput benchmark from order `n0` to `n1`.

```console
python3 benchmark/throughput_cfft.py <n0> <n1> <algorithm> <--avx2> <--double>
```

Example
```console
python3 benchmark/throughput_cfft.py 5 10 zldsp
Running CFFT throughput benchmark for zldsp from order 5 to 10...
Order      Time (us)       Throughput (MFLOPS) 
---------------------------------------------
5          0.0189          42283.9468          
6          0.0281          68408.6276          
7          0.0635          70521.8659          
8          0.1303          78574.4614          
9          0.2998          76850.4027          
10         0.6359          80514.4208
```

#### RFFT Throughput Benchmark

Run `<algorithm>` RFFT throughput benchmark from order `n0` to `n1`.

```console
python3 benchmark/throughput_rfft.py <n0> <n1> <algorithm> <--avx2> <--double>
```

Example
```console
python3 benchmark/throughput_rfft.py 5 10 zldsp
Running RFFT throughput benchmark for zldsp from order 5 to 10...
Order      Time (us)       Throughput (MFLOPS) 
---------------------------------------------
5          0.0111          36143.1067          
6          0.0221          43344.2067          
7          0.0406          55150.3424          
8          0.0840          60944.2159          
9          0.1705          67558.5149          
10         0.3770          67910.7675
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