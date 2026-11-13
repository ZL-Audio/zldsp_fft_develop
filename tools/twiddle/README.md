# Split Twiddle Generation

`zldsp::fft` generates CFFT and RFFT twiddle tables at plan initialization from a small static table of `DoubleDouble` (`{hi, lo}`) constants, with no runtime dependency on MPFR. Both `float` and `double` share compensated anchor construction; `double` retains compensated arithmetic through the final rotation and RFFT weights, while `float` combines anchor high words in `double` before narrowing.

## Generated Constants

[`generate.cpp`](generate.cpp) emits [`zldsp_fft_common_twiddle_constants.hpp`](../../zldsp_fft/common/zldsp_fft_common_twiddle_constants.hpp) in `zldsp::fft::detail::twiddle` (outside Highway's per-target namespaces so all precisions and SIMD targets share one copy per linked image):

- **`kAnchors[129]` (4,128 bytes)**: First-octant anchors $(\cos(2\pi j/1024),\; \sin(2\pi j/1024))$ for $0 \le j \le 128$, stored as `{hi, lo}` binary64 pairs.
- **Auxiliary constants (184 bytes)**:
  - `kTau`: Two-word $2\pi$.
  - `kInvSqrt2`: Directly rounded $\cos(\pi/4) = 1/\sqrt{2}$ (`1.0 / round(sqrt(2))` differs by 1 ULP in binary64).
  - `kSinResidual[5]` and `kCosResidual[5]`: Two-word Taylor coefficients for $\sin(t) - t$ (through degree 11) and $\cos(t) - 1$ (through degree 10) on $|t| < 2\pi/1024$.

Each constant is certified with outward-rounded MPFR bounds (`MPFR_RNDD` / `MPFR_RNDU`), doubling precision from 192 to 3,072 bits until both bounds round (`MPFR_RNDN`) to the same `hi` word and, after subtracting `hi`, the same residual `lo` word. Output uses hexadecimal float literals (`%a`) for host-independent reproduction.

## Algorithm and Rounding

Implemented in [`zldsp_fft_common_twiddle.hpp`](../../zldsp_fft/common/zldsp_fft_common_twiddle.hpp) and [`zldsp_fft_common_init.hpp`](../../zldsp_fft/common/zldsp_fft_common_init.hpp), assuming IEEE 754 binary32/binary64 round-to-nearest:

- **Two-table split (`SplitTable`)**: For a power-of-two size $N = 2^{\text{order}} \ge 4$ and first-octant bound $\text{last} \le N/8$, each index $k$ is split as $k = qB + r$ with $B = 2^{\lfloor(\text{order}-3)/2\rfloor} \approx \sqrt{N/8}$ ($B = 1$ for $\text{order} \le 3$). Temporary coarse ($qB$) and fine ($r$) anchors are built in a single `std::vector<Anchor>`.
- **Anchor construction (`SplitTable::anchor`)**: Maps an index to the nearest lower point in `kAnchors`. If the residual angle $|t| < 2\pi/1024$ is nonzero, evaluates the degree-11/10 Taylor series in `DoubleDouble` (`two_sum`, FastTwoSum `normalize`, and TwoProduct `product`) and rotates the grid anchor. `product` uses a scalar Highway native FMA (`HWY_NATIVE_FMA`) when available, or Dekker splitting ($2^{27} + 1$) on SSE2/SSE4 to avoid software `fma` calls.
- **Octant lookup (`SplitTable::octant<F>`)**:
  - `double`: Combines coarse and fine anchors with `DoubleDouble` rotation (skipping identity rotations when $q = 0$ or $r = 0$) and rounds via `rounded<double>` (`a.hi`).
  - `float`: For $N \le 1024$ ($\text{order} \le 10$), reads `kAnchors` high words directly and skips coarse/fine allocation. For $N > 1024$, rotates coarse and fine high words in ordinary `double` arithmetic. (`rounded<float>` also provides residual-based binary32 midpoint tie-breaking.)
- **CFFT stages (`Stage<F>`)**: Precomputes first-octant roots up to $\min(N/8, \text{max\_index})$ and folds arbitrary indices across octants via swaps and sign flips (`Stage::forward`), sharing roots across all three radix-4 harmonics ($m = 1, 2, 3$). Compact macro stages (`tile_width != 0`) generate only the fine prefix $3(\text{tile\_width} - 1)$ on grid $N$ and coarse factors on $N / \text{tile\_width}$ (runtime tile reconstruction during FFT execution introduces its own rounding).
- **RFFT weights (`generate_rfft_pre_post_w`, `rfft_weights`)**: Emits $k$ and mirrored $N/4 - k$ weights directly into the SIMD-blocked layout (padded with `{0, 0.5}`).
  - `double`: Retains `a.s.lo` through $1 + \sin\theta$ before exact scaling by `0.5`.
  - `float`: Computes `0.5 * cos(theta)` and `0.5 * (1.0 + sin(theta))` in `double` before narrowing to `float` (accepting rare 1-ULP differences without midpoint checks).

## Compiler Settings

Compensated arithmetic (`two_sum`, `normalize`, and Dekker `product`) requires uncontracted intermediate rounding while preserving explicit Highway intrinsic FMAs. [`zldsp_fft_common_twiddle.hpp`](../../zldsp_fft/common/zldsp_fft_common_twiddle.hpp) applies scoped pragmas and restores the surrounding settings afterward:

- **Clang / AppleClang**: Uses `float_control(precise, on, push)` and `clang fp contract(off)`. Supports default contraction, `-ffp-contract=off`, or `-ffp-contract=fast-honor-pragmas` (used by this repository's Release builds). **Do not use plain `-ffp-contract=fast`**, which can fuse operations in the backend despite IR pragmas.
- **GCC**: Uses scoped `GCC optimize("no-fast-math", "fp-contract=off")`.
- **MSVC** (Visual Studio 2022+): Uses scoped `float_control(precise, on, push)`, which disables implicit contraction while leaving intrinsic FMAs intact.

Unrestricted `-ffast-math` is outside the numerical contract.

## Regeneration and Verification

Offline tools require **MPFR 4.2+**, **GMP**, and **`pkg-config`**. From the repository root:

```sh
cmake -S . -B build_fft/twiddle-tools -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DTWIDDLE_ACCURACY_TEST=ON \
  -DBENCHMARK_SELF_TESTS=ON
cmake --build build_fft/twiddle-tools --parallel

# Verify the committed header matches generate.cpp.
build_fft/twiddle-tools/generate_twiddle_constants --check \
  zldsp_fft/common/zldsp_fft_common_twiddle_constants.hpp

# Regenerate after changing formulas or generator logic.
build_fft/twiddle-tools/generate_twiddle_constants \
  > zldsp_fft/common/zldsp_fft_common_twiddle_constants.hpp

# Run CTest (--quick audit + header check), then the full audit sweep.
ctest --test-dir build_fft/twiddle-tools --output-on-failure
build_fft/twiddle-tools/zldsp_fft_twiddle_accuracy
```

[`benchmark/twiddle_accuracy.cpp`](../../benchmark/twiddle_accuracy.cpp) certifies reference values with outward-rounded MPFR intervals (doubling precision on ties) and checks:

- **Primitives**: `kInvSqrt2`, `product` high/residual words, and `rounded<float>` midpoint tie-breaking.
- **First-octant grids & RFFT weights**: Exhaustive through order 16 and at orders 20 and 22 (`2..12` with `--quick`), and sampled through order 30 (2,048 random indices plus neighbors of seed-anchor boundaries, coarse/fine split boundaries, axes, diagonals, the order-28 regression index $N/2 - 124{,}806{,}999$, and the order-30 float RFFT midpoint index $93{,}157{,}219$).
- **Folding & stored tables**: Full-circle and partial-octant `Stage<F>::forward` folding (orders `2..12`), plus emitted micro/macro CFFT and capped-lane RFFT tables and padding (orders `1..16` and `20`; `1..12` and `16` with `--quick`).
- **Pass criteria**: `double` coefficients, embedded grids ($N \le 1024$), axes, and diagonals must match MPFR rounding (`0` ULPs; $\pm 0$ treated as equal). Other `float` coefficients ($N > 1024$) may differ by at most 1 ULP, reported as `float_rounding_differences` rather than `failures`.

Pass `-DUSE_AVX2=ON` or `-DUSE_SSE2=ON` on x86 to exercise those native paths. End-to-end transform and API accuracy remain covered by `python3 benchmark/run.py accuracy`.

## References

1. Martin Reinecke, **PocketFFT**, `sincos_2pibyn` in [`pocketfft_hdronly.h` (`4d865cba`)](https://github.com/mreineck/pocketfft/blob/4d865cba24359b05b37da4e75aee24498c3b78e7/pocketfft_hdronly.h#L345) — two-table angle-addition strategy.
2. Takeshi Ogita, Siegfried M. Rump, and Shin'ichi Oishi, **Accurate Sum and Dot Product**, *SIAM J. Sci. Comput.* 26(6), 1955–1988 (2005) — `TwoSum`, `FastTwoSum`, and `TwoProduct` (including FMA).
3. Laurent Fousse et al., **MPFR: A Multiple-Precision Binary Floating-Point Library with Correct Rounding**, *ACM Trans. Math. Softw.* 33(2), Article 13 (2007).

MPFR and GMP are offline tool/test dependencies and retain their own licenses; no third-party trigonometric code is included in the FFT headers.
