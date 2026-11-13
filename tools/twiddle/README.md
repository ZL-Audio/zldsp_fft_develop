# Split twiddle generation

The FFT library shares high/low anchor generation between `float` and `double`.
Double storage uses compensated rotation and RFFT weights. Float storage uses
ordinary binary64 rotation and forms the final RFFT weights in double before
narrowing. There is no additional lookup table or runtime dependency on MPFR.

## Generated data

`generate.cpp` produces
[`zldsp_fft_common_twiddle_constants.hpp`](../../zldsp_fft/common/zldsp_fft_common_twiddle_constants.hpp).
Its 129 first-octant anchors are

\[
(\cos(2\pi j/1024),\;\sin(2\pi j/1024)),\qquad 0\le j\le128.
\]

Each component is stored as a binary64 high word and the binary64 rounding of its
residual. The anchor array is exactly **4,128 bytes**. The auxiliary constants are
the two words of `2*pi`, ten two-word Taylor coefficients, and the correctly
rounded `1/sqrt(2)` value: another 184 bytes before compiler pooling/alignment.
The constants live outside Highway's target-specific namespaces so float, double,
and different SIMD targets can share one copy in a linked image.

The generator evaluates lower and upper MPFR bounds, rounds each to binary64,
and requires agreement. It repeats this check after subtracting the high word to
certify the residual word too. Precision increases when necessary. Hexadecimal
C++ literals preserve those bits; regenerating does not depend on the host's
`sin`, `cos`, `long double`, or decimal formatting precision.

## Algorithm and rounding

Integer phase indices are folded into the first octant. For a stage of length
`N`, the generator splits `k = q*B + r`, with a power-of-two `B` near the square
root of the octant length. It constructs coarse anchors for `q*B` and fine
anchors for `r`. The coarse/fine arrays are temporary runtime data. Compact macro
fine tables request only a prefix; their coarse factors use the smaller
`N/tile_width` grid.

To construct an anchor between embedded grid points, the remaining angle is
less than `2*pi/1024`. Degree-11 sine and degree-10 cosine Taylor expansions, with
two-word coefficients, evaluate this residual. TwoSum and TwoProduct retain
rounding residuals during anchor construction for both storage precisions.
The FMA version uses Highway's native fused operation; SSE2/SSE4 use Dekker
splitting instead of a software `fma` call.

For double storage, the final rotation and RFFT affine addition retain both
words. For float storage, the final rotation combines only the anchor high words
using ordinary binary64 arithmetic. Grids through 1,024 read the existing
embedded anchor high words directly and skip coarse/fine allocation. Neither
precision adds a new static table.

Each dense CFFT stage shares its first-octant results across the three radix-4
harmonics using swaps and signs. RFFT emits mirrored positions directly into the
existing blocked layout. Its float path computes `cos(theta)/2` and
`(1+sin(theta))/2` in double, then converts each final weight to float. There is
no midpoint check or compensated fallback on this path; rare adjacent-float
differences from correct rounding are accepted. The double path retains the
sine residual through the addition. The diagonal constant is generated directly;
taking the reciprocal of rounded `sqrt(2)` gives a different binary64 value.

The arithmetic assumes IEEE binary32/binary64 and round-to-nearest. Correct
rounding is not guaranteed for every coefficient. The audit below checks actual
stored coefficients independently.
Compact macro reconstruction during FFT execution still introduces its own
rounding after the factors have been stored.

## Compiler settings

Compensation requires the indicated intermediate operations to round separately.
The helper header uses local precise floating-point settings and disables
implicit contraction; explicit native FMAs remain enabled. The surrounding FFT
kernels retain their own floating-point settings.

- Clang/AppleClang: default contraction, `-ffp-contract=off`, or
  `-ffp-contract=fast-honor-pragmas` are supported. **Do not use plain
  `-ffp-contract=fast`**: it can fuse operations in the machine-code backend even
  when the local pragma has removed contraction flags from the intermediate IR.
  This repository uses `fast-honor-pragmas` for optimized builds.
- GCC: the helper uses a scoped `no-fast-math,fp-contract=off` optimization pragma.
- MSVC: use Visual Studio 2022 or newer; scoped `float_control(precise,on)` also
  disables implicit contraction there. Explicit intrinsic FMAs are unaffected.

Unrestricted fast-math is outside the numerical contract. The original settings
are restored after the helper definitions.

## Regeneration and verification

Install MPFR **4.2 or newer**, GMP development files, and `pkg-config`. They are
needed only for these tools. From the repository root:

```sh
cmake -S . -B build_fft/twiddle-tools -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DTWIDDLE_ACCURACY_TEST=ON \
  -DBENCHMARK_SELF_TESTS=ON
cmake --build build_fft/twiddle-tools --parallel

# Verify the committed file without changing it.
build_fft/twiddle-tools/generate_twiddle_constants --check \
  zldsp_fft/common/zldsp_fft_common_twiddle_constants.hpp

# Regenerate deliberately after changing the formulas/generator.
build_fft/twiddle-tools/generate_twiddle_constants \
  > zldsp_fft/common/zldsp_fft_common_twiddle_constants.hpp

# Short coefficient/layout suite, followed by the extended numerical sweep.
ctest --test-dir build_fft/twiddle-tools --output-on-failure
build_fft/twiddle-tools/zldsp_fft_twiddle_accuracy
```

The default audit exhausts first-octant grids through order 16 and at orders 20
and 22, checks both mirrored RFFT weights, and samples other orders through 30.
Samples include seed/split boundaries, axes, diagonals, and the recorded order-28
counterexample, plus the float RFFT midpoint at order 30, index 93,157,219.
It also checks both storage precisions, integer folding, small specialized
tables, actual micro/macro tables, RFFT capped-lane layouts/padding, the diagonal
constant, and compensated float midpoint conversions. Directed MPFR intervals
certify each reference; an inconclusive interval increases the reference
precision instead of guessing. Zero signs are treated as equivalent.

Double coefficients, embedded grids, axes, and diagonals must match the reference
exactly. Other float coefficients may differ by one float ULP; the audit reports
these as `float_rounding_differences` separately from failures. This threshold
checks the chosen accuracy tradeoff and is not a correct-rounding proof.

Use `-DUSE_AVX2=ON` or `-DUSE_SSE2=ON` on an x86 machine to exercise those native
paths. The test obtains lane counts from Highway, including small RFFT capped
tags. Existing `benchmark/run.py accuracy` checks remain necessary for complete
transform behavior, all APIs, independent-library comparisons, and round trips.

## References and provenance

1. Martin Reinecke, **PocketFFT**, `sincos_2pibyn` in
   [`pocketfft_hdronly.h`, revision 4d865cba](https://github.com/mreineck/pocketfft/blob/4d865cba24359b05b37da4e75aee24498c3b78e7/pocketfft_hdronly.h#L345).
   Reference for the two-table angle-addition strategy.
2. Takeshi Ogita, Siegfried M. Rump, and Shin'ichi Oishi, **Accurate Sum and Dot
   Product**, *SIAM Journal on Scientific Computing* 26(6), 1955–1988 (2005).
   Reference for TwoSum, FastTwoSum, and TwoProduct, including the FMA variant.
3. Laurent Fousse et al., **MPFR: A Multiple-Precision Binary Floating-Point
   Library with Correct Rounding**, *ACM Transactions on Mathematical Software*
   33(2), Article 13 (2007).

MPFR and GMP are offline tool/test dependencies and retain their own licenses.
No third-party trigonometric implementation is included in the FFT headers.
