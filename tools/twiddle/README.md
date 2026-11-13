# Compensated split twiddle generation

The FFT library uses the same compensated split generator for every stage size,
for both `float` and `double`, and for the final RFFT pre/post-processing weights.
There is no separate 4,096-point lookup table and no runtime dependency on MPFR.

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
anchors for `r`, then combines the two rotations with high/low arithmetic. The
coarse/fine arrays are temporary runtime data. Compact macro fine tables request
only a prefix; their coarse factors use the smaller `N/tile_width` grid.

To construct an anchor between embedded grid points, the remaining angle is
less than `2*pi/1024`. Degree-11 sine and degree-10 cosine Taylor expansions, with
two-word coefficients, evaluate this residual. TwoSum and TwoProduct retain
rounding residuals through both the anchor construction and the final rotation.
The FMA version uses Highway's native fused operation; SSE2/SSE4 use Dekker
splitting instead of a software `fma` call.

Each dense CFFT stage shares its first-octant results across the three radix-4
harmonics using swaps and signs. RFFT emits mirrored positions directly into the
existing blocked layout. It computes `cos(theta)/2` and `(1+sin(theta))/2` while
the high/low parts are still available, so the sine is **not rounded before the
addition**. Float conversion also uses the residual when the binary64 high word
is exactly a binary32 rounding midpoint. The diagonal constant is generated
directly; taking the reciprocal of rounded `sqrt(2)` gives a different binary64
value.

The arithmetic assumes IEEE binary32/binary64 and round-to-nearest. It is a
compensated approximation, not a proof of correct rounding for every possible
FFT order. The audit below checks actual stored coefficients independently.
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
counterexample. It also checks both storage precisions, integer folding, small
specialized tables, actual micro/macro tables, RFFT capped-lane layouts/padding,
the diagonal constant, and float midpoint conversions. Directed MPFR intervals
certify each reference; an inconclusive interval increases the reference
precision instead of guessing. Zero signs are treated as equivalent.

Use `-DUSE_AVX2=ON` or `-DUSE_SSE2=ON` on an x86 machine to exercise those native
paths. The test obtains lane counts from Highway, including small RFFT capped
tags. Existing `benchmark/run.py accuracy` checks remain necessary for complete
transform behavior, all APIs, independent-library comparisons, and round trips.

## References and provenance

1. The local design study,
   [`twiddle-study/README.md`](../../twiddle-study/README.md), motivated integer
   folding, split tables, octant sharing, and carrying a sine residual through the
   RFFT affine transform. This implementation strengthens its arithmetic to keep
   two words through the residual polynomials and rotation products, and uses a
   separately reproducible MPFR constant generator.
2. Martin Reinecke, **PocketFFT**, `sincos_2pibyn` in
   [`pocketfft_hdronly.h`, revision 4d865cba](https://github.com/mreineck/pocketfft/blob/4d865cba24359b05b37da4e75aee24498c3b78e7/pocketfft_hdronly.h#L345).
   Reference for the two-table angle-addition strategy. The code here implements
   the elementary rotation identity with compensated arithmetic; it does not
   vendor PocketFFT's implementation.
3. Takeshi Ogita, Siegfried M. Rump, and Shin'ichi Oishi, **Accurate Sum and Dot
   Product**, *SIAM Journal on Scientific Computing* 26(6), 1955–1988 (2005),
   [doi:10.1137/030601818](https://doi.org/10.1137/030601818),
   [author-hosted paper](https://ogilab.w.waseda.jp/ogita/math/doc/2005_OgRuOi.pdf).
   Reference for TwoSum, FastTwoSum, and TwoProduct, including the FMA variant.
4. Laurent Fousse et al., **MPFR: A Multiple-Precision Binary Floating-Point
   Library with Correct Rounding**, *ACM Transactions on Mathematical Software*
   33(2), Article 13 (2007),
   [doi:10.1145/1236463.1236468](https://doi.org/10.1145/1236463.1236468).
   See also the [MPFR manual](https://www.mpfr.org/mpfr-current/mpfr.html) for
   `mpfr_sinpi`, `mpfr_cospi`, and directed rounding semantics.
5. [Clang floating-point pragmas](https://clang.llvm.org/docs/LanguageExtensions.html#extensions-to-specify-floating-point-flags),
   [Clang contraction options](https://releases.llvm.org/17.0.1/tools/clang/docs/ClangCommandLineReference.html#cmdoption-clang-ffp-contract),
   and [MSVC `float_control`](https://learn.microsoft.com/en-us/cpp/preprocessor/float-control)
   document the compiler controls required by the error-free transformations.

MPFR and GMP are offline tool/test dependencies and retain their own licenses.
No third-party trigonometric implementation is included in the FFT headers.
