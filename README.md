# zldsp::fft (develop)

Development, benchmarking, and analysis repository for `zldsp::fft`. To use the library in a project, see [zldsp_fft](https://github.com/ZL-Audio/zldsp_fft).

## Usage

Requirements: `Clang` (`AppleClang 16+` or `LLVM/Clang 17+`), `CMake` (`3.25+`), and `Ninja`. On Windows, run from a Visual Studio developer shell.

```console
git clone https://github.com/ZL-Audio/zldsp_fft_develop
cd zldsp_fft_develop
git submodule update --init
```

Run all benchmarks via a single CLI (`1 <= n0 <= n1 <= 30`, FFT size `N = 2^order`):

```console
python3 benchmark/run.py <accuracy|throughput|compare> <cfft|rfft> <n0> <n1> [options]
```

### Common Options

- `--double`: Use `double` precision (default: `float`).
- `--avx2` / `--sse2`: Select x86 SIMD target (default: SSE4.2 on x86; ARM uses NEON).
- `--apis`: Test all `zldsp::fft` layouts and APIs (AoS/SoA, `forward_sqr_mag`, RFFT `convolve`).
- `--repeats <k>`: Repetitions for random accuracy signals (`3`), throughput timings (`3`), or paired comparisons (`6`).
- `--artifacts-dir <dir>`: Build/temp directory (default: `<repo>/build_fft`, or system temp directory for `compare`).
- `--json <file>`: Write structured results to JSON.

### Accuracy

```console
python3 benchmark/run.py accuracy cfft 2 20
python3 benchmark/run.py accuracy rfft 2 20 --apis --double
```

- **Library agreement (default)**: Compares `fftw3_estimate`, `kfr`, and `zldsp` against their component-wise median and checks normalized round trips (`1/N` for CFFT; `2/N` for RFFT since backward RFFT scales by `N/2`).
- **API consistency (`--apis`)**: Validates all `zldsp` AoS/SoA layouts, `forward_sqr_mag`, and `convolve` overloads against the AoS reference path and direct DFT.
- **Metrics**: Reports worst relative RMSE and max absolute error across deterministic and random signals; fails above `--threshold` (`2e-5` for `float`, `2e-12` for `double`).

### Throughput

```console
python3 benchmark/run.py throughput cfft 5 20
python3 benchmark/run.py throughput rfft 5 20 --libraries zldsp kfr pffft vdsp
python3 benchmark/run.py throughput rfft 5 20 --standard-libraries
python3 benchmark/run.py throughput cfft 5 20 --all-libraries --double
python3 benchmark/run.py throughput rfft 5 20 --apis
```

- **Targets** (default: `zldsp`): `--libraries <lib ...>`, `--standard-libraries` (`zldsp`, `kfr`, `pffft`, plus installed `vdsp`, `ipp`, `armpl`, `armpl_estimate`), `--all-libraries` (adds `fftw3`, `fftw3_estimate`), or `--apis` (all `zldsp` layouts, `forward_sqr_mag`, and fused vs. separate `convolve`).
- **Measurement**: Reuses plans and buffers to time only transforms (using native split/unordered formats for vDSP and PFFFT), runs all selected targets at each order before advancing, and frees buffers between orders.
- **Timing & builds**: Per-repetition duration scales exponentially from `--min-time0` (`0.5`s at `n0`) to `--min-time1` (`5`s at `n1`), or fixed via `--min-time <s>`. Use `--build-only` or `--skip-build` to decouple building from timing.
- **Metrics**: Reports nominal **MFLOPS** from median single-call times across `--repeats` runs (`5*N*log2(N)` for CFFT, `2.5*N*log2(N)` for RFFT).

### Comparing Git Revisions

```console
python3 benchmark/run.py compare cfft 5 20 --repeats 6
python3 benchmark/run.py compare rfft 5 20 --apis --current HEAD --previous HEAD^
```

Alternates `--current` (default `HEAD`) and `--previous` (default `HEAD^`) across `--repeats` paired runs, reporting median times, speedup ratios (`> 1` is faster), percentage changes, and MAD.

### Supported Backends

`zldsp`, `fftw3`, `fftw3_estimate`, `kfr`, `pffft`, `vdsp`, `ipp`, `armpl`, `armpl_estimate` (`fftw3` and `armpl` use `FFTW_MEASURE`; `*_estimate` use `FFTW_ESTIMATE`). Set `ARMPL_DIR` or `IPP_ROOT` for non-default installations.

## Use `zldsp::fft` in another project

`zldsp::fft` is a C++20 header-only library built on Google Highway (`hwy`), supporting fixed-width 128-bit and 256-bit SIMD targets:

- **Static Dispatch**: Compile for a single fixed SIMD target (`zldsp::fft::CFFT<F>`, `zldsp::fft::RFFT<F>`) with direct calls.
- **Caller-owned Dynamic Dispatch**: Compile multiple SIMD targets in a single translation unit via `<hwy/foreach_target.h>` and dispatch at plan creation. Recommended for **GCC and Clang** (`clang++` / `clang-cl`).
- **Separately Compiled Dispatch**: Compile each x86-64 SIMD backend (`SSE2`, `SSE4`, `AVX2`) as a separate object library with dedicated ISA flags and dispatch at plan creation. Recommended for **MSVC** (and supported on GCC/Clang).

### Static Dispatch

Include [`zldsp_fft_cfft.hpp`](zldsp_fft/zldsp_fft_cfft.hpp) / [`zldsp_fft_rfft.hpp`](zldsp_fft/zldsp_fft_rfft.hpp), define `HWY_COMPILE_ONLY_STATIC`, and select the target ISA with compiler flags:

```cmake
target_compile_definitions(my_static_target PRIVATE HWY_COMPILE_ONLY_STATIC)
```

| SIMD Target | GCC/Clang                         | MSVC             |
| ----------- | --------------------------------- | ---------------- |
| SSE2        | `-march=x86-64`                   | no flag required |
| SSE4        | `-march=x86-64-v2 -maes -mpclmul` | not supported    |
| AVX2        | `-march=x86-64-v3 -maes -mpclmul` | `/arch:AVX2`     |
| NEON        | `-march=armv8-a+simd`             | `/arch:armv8.0`  |

See [`static_dispatch_caller.cpp`](examples/static_dispatch_caller.cpp) for complete examples.

### Caller-owned Dynamic Dispatch

Re-include [`zldsp_fft_cfft-inl.hpp`](zldsp_fft/zldsp_fft_cfft-inl.hpp) and [`zldsp_fft_rfft-inl.hpp`](zldsp_fft/zldsp_fft_rfft-inl.hpp) with `<hwy/foreach_target.h>` in a wrapper translation unit to instantiate `zldsp::fft::HWY_NAMESPACE::CFFT<F>` / `RFFT<F>` per target and dispatch plan creation with `HWY_DYNAMIC_DISPATCH`.

Because `zldsp::fft` only supports 128-bit and 256-bit fixed-width SIMD, you must define `HWY_DISABLED_TARGETS` to disable unsupported targets (such as AVX-512 or SVE) and compile for your oldest supported baseline (e.g., `-march=x86-64` for SSE2):

```cmake
target_compile_definitions(my_dynamic_target PRIVATE "HWY_DISABLED_TARGETS=~(HWY_SSE2|HWY_AVX2)")
```

See [`dynamic_dispatch_wrapper.hpp`](examples/dynamic_dispatch_wrapper.hpp), [`dynamic_dispatch_wrapper.cpp`](examples/dynamic_dispatch_wrapper.cpp), and [`dynamic_dispatch_caller.cpp`](examples/dynamic_dispatch_caller.cpp).

> [!WARNING]
> MSVC may cause **severe performance degradation** when the dispatch target is AVX2 and the baseline is SSE2. Use **Separately Compiled Dispatch** for runtime dispatch with MSVC.

### Separately Compiled Dispatch

With Highway available as the `hwy` target, [`separate_dispatch.cmake`](examples/separate_dispatch.cmake) compiles [`dynamic_dispatch_wrapper.cpp`](examples/dynamic_dispatch_wrapper.cpp) once per x86-64 backend with `HWY_COMPILE_ONLY_STATIC` and matching ISA flags, then links them into `zldsp_fft_separate_dispatch`:

```cmake
set(ZLDSP_FFT_SEPARATE_TARGETS SSE2 SSE4 AVX2)
include(path/to/zldsp_fft_develop/examples/separate_dispatch.cmake)
target_link_libraries(my_separate_target PRIVATE zldsp_fft_separate_dispatch)
```

- **Backends**: Choose from `SSE2`, `SSE4`, and `AVX2` (`SSE4` is not supported by MSVC). Include `SSE2` for a baseline fallback; plan creation throws `std::runtime_error` if no compiled backend supports the CPU.
- **Build settings**: The helper sets ISA flags only on the per-backend object libraries; your application, `hwy`, and the dispatcher inherit your own baseline, optimization, PIC, and IPO settings.
- **Integration**: Link `zldsp_fft_separate_dispatch` instead of compiling [`dynamic_dispatch_wrapper.cpp`](examples/dynamic_dispatch_wrapper.cpp) directly. See [`separate_dispatch_wrapper.cpp`](examples/separate_dispatch_wrapper.cpp), [`dynamic_dispatch_wrapper.hpp`](examples/dynamic_dispatch_wrapper.hpp), and [`dynamic_dispatch_caller.cpp`](examples/dynamic_dispatch_caller.cpp).

## License

`zldsp_fft` is licensed under the Apache-2.0 license ([LICENSE.md](LICENSE.md)). External submodules retain their respective licenses:

- FFTW3: `fftw3_impl/fftw3` (GPL-2.0)
- KFR: `kfr_impl/kfr` (GPL-2.0)
- PFFFT: `pffft_impl/pffft` (BSD-like)
- Google Benchmark: `google/benchmark` (Apache-2.0)
- Google Highway: `google/highway` (Apache-2.0)
