#pragma once

#include <vector>
#include <span>
#include <complex>
#include <cmath>
#include <algorithm>
#include <numbers>
#include <cstdint>
#include <cassert>

#include <hwy/highway.h>
#include <hwy/aligned_allocator.h>

#include <cstddef>
#include <bit>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace zlfft::common {
    namespace hn = hwy::HWY_NAMESPACE;

    inline size_t get_system_page_size() {
#if defined(_WIN32)
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        return static_cast<size_t>(sysInfo.dwPageSize);
#else
        return static_cast<size_t>(sysconf(_SC_PAGESIZE));
#endif
    }

    enum class StageType {
        kRadix8FirstPass,
        kRadix4FirstPass,
        kRadix4Width4,
        kRadix4Width8,
        kRadix4,
        kRadix8,
        kRadix4LastPass,
    };

    template <typename F>
    inline void radix4(const F* __restrict in_r, const F* __restrict in_i,
                       F* __restrict out_r, F* __restrict out_i,
                       const size_t n, const size_t width,
                       const F* __restrict w_r_ptr, const F* __restrict w_i_ptr) {
        const auto quarter_n = n >> 2;
        const auto half_n = n >> 1;
        const auto three_quarter_n = quarter_n + half_n;

        const auto double_width = width << 1;
        const auto triple_width = width * 3;

        static constexpr hn::ScalableTag<F> d;
        static constexpr size_t lanes = hn::Lanes(d);

        const size_t mask = width - 1;

        for (size_t i = 0; i < quarter_n; i += lanes) {
            const auto r0 = hn::Load(d, in_r + i);
            const auto i0 = hn::Load(d, in_i + i);
            const auto r1 = hn::Load(d, in_r + quarter_n + i);
            const auto i1 = hn::Load(d, in_i + quarter_n + i);
            const auto r2 = hn::Load(d, in_r + half_n + i);
            const auto i2 = hn::Load(d, in_i + half_n + i);
            const auto r3 = hn::Load(d, in_r + three_quarter_n + i);
            const auto i3 = hn::Load(d, in_i + three_quarter_n + i);

            const size_t k = i & mask;
            const auto w1_r = hn::LoadU(d, w_r_ptr + k);
            const auto w1_i = hn::LoadU(d, w_i_ptr + k);
            const auto w2_r = hn::LoadU(d, w_r_ptr + width + k);
            const auto w2_i = hn::LoadU(d, w_i_ptr + width + k);
            const auto w3_r = hn::LoadU(d, w_r_ptr + (width << 1) + k);
            const auto w3_i = hn::LoadU(d, w_i_ptr + (width << 1) + k);

            const auto t1_r = hn::NegMulAdd(i1, w1_i, hn::Mul(r1, w1_r));
            const auto t1_i = hn::MulAdd(i1, w1_r, hn::Mul(r1, w1_i));
            const auto t3_r = hn::NegMulAdd(i3, w3_i, hn::Mul(r3, w3_r));
            const auto t3_i = hn::MulAdd(i3, w3_r, hn::Mul(r3, w3_i));

            const auto s2_r = hn::Add(t1_r, t3_r);
            const auto s2_i = hn::Add(t1_i, t3_i);
            const auto s3_r = hn::Sub(t1_r, t3_r);
            const auto s3_i = hn::Sub(t1_i, t3_i);

            const auto t2_r = hn::NegMulAdd(i2, w2_i, hn::Mul(r2, w2_r));
            const auto t2_i = hn::MulAdd(i2, w2_r, hn::Mul(r2, w2_i));

            const auto s0_r = hn::Add(r0, t2_r);
            const auto s0_i = hn::Add(i0, t2_i);
            const auto s1_r = hn::Sub(r0, t2_r);
            const auto s1_i = hn::Sub(i0, t2_i);

            const size_t j_times_4 = (i & ~mask) << 2;
            const size_t out_idx = j_times_4 + k;

            F* __restrict out_r_shift = out_r + out_idx;
            F* __restrict out_i_shift = out_i + out_idx;

            hn::Store(hn::Add(s0_r, s2_r), d, out_r_shift);
            hn::Store(hn::Add(s0_i, s2_i), d, out_i_shift);

            hn::Store(hn::Add(s1_r, s3_i), d, out_r_shift + width);
            hn::Store(hn::Sub(s1_i, s3_r), d, out_i_shift + width);

            hn::Store(hn::Sub(s0_r, s2_r), d, out_r_shift + double_width);
            hn::Store(hn::Sub(s0_i, s2_i), d, out_i_shift + double_width);

            hn::Store(hn::Sub(s1_r, s3_i), d, out_r_shift + triple_width);
            hn::Store(hn::Add(s1_i, s3_r), d, out_i_shift + triple_width);
        }
    }

    template <typename F>
    inline void radix4_first_pass_fused(const std::complex<F>* __restrict in,
                                        F* __restrict out_r, F* __restrict out_i, const size_t n) {
        const size_t quarter_n = n >> 2;
        const size_t half_n = n >> 1;
        const size_t three_over_two_n = n + half_n;

        static constexpr hn::ScalableTag<F> d;
        static constexpr size_t lanes = hn::Lanes(d);

        for (size_t j = 0; j < quarter_n; j += lanes) {
            const F* __restrict in_shift = reinterpret_cast<const F*>(in + j);

            hn::Vec<decltype(d)> x0_r, x0_i, x2_r, x2_i;

            hn::LoadInterleaved2(d, in_shift, x0_r, x0_i);
            hn::LoadInterleaved2(d, in_shift + n, x2_r, x2_i);

            const auto t0_r = hn::Add(x0_r, x2_r);
            const auto t0_i = hn::Add(x0_i, x2_i);
            const auto t1_r = hn::Sub(x0_r, x2_r);
            const auto t1_i = hn::Sub(x0_i, x2_i);

            hn::Vec<decltype(d)> x1_r, x1_i, x3_r, x3_i;

            hn::LoadInterleaved2(d, in_shift + half_n, x1_r, x1_i);
            hn::LoadInterleaved2(d, in_shift + three_over_two_n, x3_r, x3_i);

            const auto t2_r = hn::Add(x1_r, x3_r);
            const auto t2_i = hn::Add(x1_i, x3_i);
            const auto t3_r = hn::Sub(x1_r, x3_r);
            const auto t3_i = hn::Sub(x1_i, x3_i);

            const auto out0_r = hn::Add(t0_r, t2_r);
            const auto out0_i = hn::Add(t0_i, t2_i);
            const auto out2_r = hn::Sub(t0_r, t2_r);
            const auto out2_i = hn::Sub(t0_i, t2_i);

            const auto t3_neg_i_r = t3_i;
            const auto t3_neg_i_i = hn::Neg(t3_r);

            const auto out1_r = hn::Add(t1_r, t3_neg_i_r);
            const auto out1_i = hn::Add(t1_i, t3_neg_i_i);
            const auto out3_r = hn::Sub(t1_r, t3_neg_i_r);
            const auto out3_i = hn::Sub(t1_i, t3_neg_i_i);

            const size_t out_offset = j << 2;
            hn::StoreInterleaved4(out0_r, out1_r, out2_r, out3_r, d, out_r + out_offset);
            hn::StoreInterleaved4(out0_i, out1_i, out2_i, out3_i, d, out_i + out_offset);
        }
    }

    template <typename F>
    inline void radix4_width4(const F* __restrict in_r, const F* __restrict in_i,
                              F* __restrict out_r, F* __restrict out_i,
                              const size_t n,
                              const F* __restrict w_r_ptr, const F* __restrict w_i_ptr) {
        const size_t quarter_n = n >> 2;
        const size_t half_n = n >> 1;
        const size_t three_quarter_n = quarter_n * 3;

        static constexpr hn::FixedTag<F, 4> d;

        const auto w1_r = hn::LoadU(d, w_r_ptr);
        const auto w1_i = hn::LoadU(d, w_i_ptr);
        const auto w2_r = hn::LoadU(d, w_r_ptr + 4);
        const auto w2_i = hn::LoadU(d, w_i_ptr + 4);
        const auto w3_r = hn::LoadU(d, w_r_ptr + 8);
        const auto w3_i = hn::LoadU(d, w_i_ptr + 8);

        for (size_t j = 0; j < quarter_n; j += 4) {
            const F* __restrict in_i_shift = in_i + j;
            const F* __restrict in_r_shift = in_r + j;

            const auto i1 = hn::Load(d, in_i_shift + quarter_n);
            const auto r1 = hn::Load(d, in_r_shift + quarter_n);
            const auto t1_r = hn::NegMulAdd(i1, w1_i, hn::Mul(r1, w1_r));
            const auto t1_i = hn::MulAdd(i1, w1_r, hn::Mul(r1, w1_i));

            const auto i3 = hn::Load(d, in_i_shift + three_quarter_n);
            const auto r3 = hn::Load(d, in_r_shift + three_quarter_n);
            const auto t3_r = hn::NegMulAdd(i3, w3_i, hn::Mul(r3, w3_r));
            const auto t3_i = hn::MulAdd(i3, w3_r, hn::Mul(r3, w3_i));

            const auto s2_r = hn::Add(t1_r, t3_r);
            const auto s2_i = hn::Add(t1_i, t3_i);
            const auto s3_r = hn::Sub(t1_r, t3_r);
            const auto s3_i = hn::Sub(t1_i, t3_i);

            const auto i2 = hn::Load(d, in_i_shift + half_n);
            const auto r2 = hn::Load(d, in_r_shift + half_n);
            const auto t2_r = hn::NegMulAdd(i2, w2_i, hn::Mul(r2, w2_r));
            const auto t2_i = hn::MulAdd(i2, w2_r, hn::Mul(r2, w2_i));

            const auto i0 = hn::Load(d, in_i_shift);
            const auto r0 = hn::Load(d, in_r_shift);

            const auto s0_r = hn::Add(r0, t2_r);
            const auto s0_i = hn::Add(i0, t2_i);
            const auto s1_r = hn::Sub(r0, t2_r);
            const auto s1_i = hn::Sub(i0, t2_i);

            F* __restrict out_r_shift = out_r + (j << 2);
            F* __restrict out_i_shift = out_i + (j << 2);
            hn::Store(hn::Add(s0_r, s2_r), d, out_r_shift);
            hn::Store(hn::Add(s0_i, s2_i), d, out_i_shift);
            hn::Store(hn::Sub(s0_r, s2_r), d, out_r_shift + 8);
            hn::Store(hn::Sub(s0_i, s2_i), d, out_i_shift + 8);

            hn::Store(hn::Add(s1_r, s3_i), d, out_r_shift + 4);
            hn::Store(hn::Sub(s1_i, s3_r), d, out_i_shift + 4);
            hn::Store(hn::Sub(s1_r, s3_i), d, out_r_shift + 12);
            hn::Store(hn::Add(s1_i, s3_r), d, out_i_shift + 12);
        }
    }

    template <typename F>
    inline void radix4_last_pass_fused(const F* __restrict in_r, const F* __restrict in_i,
                                       std::complex<F>* __restrict out,
                                       const size_t n, const size_t width,
                                       const F* __restrict w_r_ptr, const F* __restrict w_i_ptr) {
        const size_t quarter_n = n >> 2;
        const size_t half_n = n >> 1;
        const size_t three_quarter_n = quarter_n * 3;

        static constexpr hn::ScalableTag<F> d;
        static constexpr size_t lanes = hn::Lanes(d);

        const size_t mask = width - 1;

        for (size_t i = 0; i < quarter_n; i += lanes) {
            const auto r0 = hn::Load(d, in_r + i);
            const auto i0 = hn::Load(d, in_i + i);
            const auto r1 = hn::Load(d, in_r + quarter_n + i);
            const auto i1 = hn::Load(d, in_i + quarter_n + i);
            const auto r2 = hn::Load(d, in_r + half_n + i);
            const auto i2 = hn::Load(d, in_i + half_n + i);
            const auto r3 = hn::Load(d, in_r + three_quarter_n + i);
            const auto i3 = hn::Load(d, in_i + three_quarter_n + i);

            const size_t k = i & mask;
            const auto w1_r = hn::LoadU(d, w_r_ptr + k);
            const auto w1_i = hn::LoadU(d, w_i_ptr + k);
            const auto w2_r = hn::LoadU(d, w_r_ptr + width + k);
            const auto w2_i = hn::LoadU(d, w_i_ptr + width + k);
            const auto w3_r = hn::LoadU(d, w_r_ptr + (width << 1) + k);
            const auto w3_i = hn::LoadU(d, w_i_ptr + (width << 1) + k);

            const auto t1_r = hn::NegMulAdd(i1, w1_i, hn::Mul(r1, w1_r));
            const auto t1_i = hn::MulAdd(i1, w1_r, hn::Mul(r1, w1_i));
            const auto t3_r = hn::NegMulAdd(i3, w3_i, hn::Mul(r3, w3_r));
            const auto t3_i = hn::MulAdd(i3, w3_r, hn::Mul(r3, w3_i));

            const auto s2_r = hn::Add(t1_r, t3_r);
            const auto s2_i = hn::Add(t1_i, t3_i);
            const auto s3_r = hn::Sub(t1_r, t3_r);
            const auto s3_i = hn::Sub(t1_i, t3_i);

            const auto t2_r = hn::NegMulAdd(i2, w2_i, hn::Mul(r2, w2_r));
            const auto t2_i = hn::MulAdd(i2, w2_r, hn::Mul(r2, w2_i));

            const auto s0_r = hn::Add(r0, t2_r);
            const auto s0_i = hn::Add(i0, t2_i);
            const auto s1_r = hn::Sub(r0, t2_r);
            const auto s1_i = hn::Sub(i0, t2_i);

            const size_t j_times_4 = (i & ~mask) << 2;
            const size_t out_idx = j_times_4 + k;

            hn::StoreInterleaved2(hn::Add(s0_r, s2_r), hn::Add(s0_i, s2_i), d,
                                  reinterpret_cast<F*>(out + out_idx));
            hn::StoreInterleaved2(hn::Add(s1_r, s3_i), hn::Sub(s1_i, s3_r), d,
                                  reinterpret_cast<F*>(out + out_idx + width));
            hn::StoreInterleaved2(hn::Sub(s0_r, s2_r), hn::Sub(s0_i, s2_i), d,
                                  reinterpret_cast<F*>(out + out_idx + (width << 1)));
            hn::StoreInterleaved2(hn::Sub(s1_r, s3_i), hn::Add(s1_i, s3_r), d,
                                  reinterpret_cast<F*>(out + out_idx + width * 3));
        }
    }

    template <typename F>
    inline void radix8(const F* __restrict in_r, const F* __restrict in_i,
                       F* __restrict out_r, F* __restrict out_i,
                       const size_t n, const size_t width,
                       const F* __restrict w_r_ptr, const F* __restrict w_i_ptr) {
        const size_t eighth_n = n >> 3;
        static constexpr hn::ScalableTag<F> d;
        static constexpr size_t lanes = hn::Lanes(d);
        const size_t mask = width - 1;

        static constexpr F kInvSqrt2 = static_cast<F>(1.0 / std::numbers::sqrt2);
        const auto inv_sqrt2 = hn::Set(d, kInvSqrt2);

        for (size_t i = 0; i < eighth_n; i += lanes) {
            const size_t k = i & mask;
            const size_t j_times_8 = (i & ~mask) << 3;
            const size_t out_idx = j_times_8 + k;

            auto load_twiddle_mul = [&](size_t in_offset, size_t w_mult, auto& r_out, auto& i_out) {
                const auto r_in = hn::Load(d, in_r + in_offset + i);
                const auto i_in = hn::Load(d, in_i + in_offset + i);
                const auto w_r = hn::LoadU(d, w_r_ptr + (w_mult * width) + k);
                const auto w_i = hn::LoadU(d, w_i_ptr + (w_mult * width) + k);
                r_out = hn::NegMulAdd(i_in, w_i, hn::Mul(r_in, w_r));
                i_out = hn::MulAdd(i_in, w_r, hn::Mul(r_in, w_i));
            };

            const auto r0 = hn::Load(d, in_r + i);
            const auto i0 = hn::Load(d, in_i + i);

            hn::Vec<decltype(d)> r4, i4;
            load_twiddle_mul(eighth_n * 4, 3, r4, i4);
            const auto t0_r = hn::Add(r0, r4), t0_i = hn::Add(i0, i4);
            const auto t1_r = hn::Sub(r0, r4), t1_i = hn::Sub(i0, i4);

            hn::Vec<decltype(d)> r2, i2, r6, i6;
            load_twiddle_mul(eighth_n * 2, 1, r2, i2);
            load_twiddle_mul(eighth_n * 6, 5, r6, i6);
            const auto t2_r = hn::Add(r2, r6), t2_i = hn::Add(i2, i6);
            const auto t3_r = hn::Sub(r2, r6), t3_i = hn::Sub(i2, i6);

            hn::Vec<decltype(d)> r1, i1, r5, i5;
            load_twiddle_mul(eighth_n * 1, 0, r1, i1);
            load_twiddle_mul(eighth_n * 5, 4, r5, i5);
            const auto u0_r = hn::Add(r1, r5), u0_i = hn::Add(i1, i5);
            const auto u1_r = hn::Sub(r1, r5), u1_i = hn::Sub(i1, i5);

            hn::Vec<decltype(d)> r3, i3, r7, i7;
            load_twiddle_mul(eighth_n * 3, 2, r3, i3);
            load_twiddle_mul(eighth_n * 7, 6, r7, i7);
            const auto u2_r = hn::Add(r3, r7), u2_i = hn::Add(i3, i7);
            const auto u3_r = hn::Sub(r3, r7), u3_i = hn::Sub(i3, i7);

            const auto y00_r = hn::Add(t0_r, t2_r), y00_i = hn::Add(t0_i, t2_i);
            const auto y02_r = hn::Sub(t0_r, t2_r), y02_i = hn::Sub(t0_i, t2_i);
            const auto y01_r = hn::Add(t1_r, t3_i), y01_i = hn::Sub(t1_i, t3_r);
            const auto y03_r = hn::Sub(t1_r, t3_i), y03_i = hn::Add(t1_i, t3_r);

            const auto y10_r = hn::Add(u0_r, u2_r), y10_i = hn::Add(u0_i, u2_i);
            const auto y12_r = hn::Sub(u0_r, u2_r), y12_i = hn::Sub(u0_i, u2_i);
            const auto y11_r = hn::Add(u1_r, u3_i), y11_i = hn::Sub(u1_i, u3_r);
            const auto y13_r = hn::Sub(u1_r, u3_i), y13_i = hn::Add(u1_i, u3_r);

            const auto v0_r = y10_r, v0_i = y10_i;
            const auto v1_r = hn::Mul(hn::Add(y11_r, y11_i), inv_sqrt2);
            const auto v1_i = hn::Mul(hn::Sub(y11_i, y11_r), inv_sqrt2);
            const auto v2_r = y12_i, v2_i = hn::Neg(y12_r);
            const auto v3_r = hn::Mul(hn::Sub(y13_i, y13_r), inv_sqrt2);
            const auto v3_i = hn::Mul(hn::Neg(hn::Add(y13_r, y13_i)), inv_sqrt2);

            hn::Store(hn::Add(y00_r, v0_r), d, out_r + out_idx);
            hn::Store(hn::Add(y00_i, v0_i), d, out_i + out_idx);
            hn::Store(hn::Sub(y00_r, v0_r), d, out_r + out_idx + (width << 2));
            hn::Store(hn::Sub(y00_i, v0_i), d, out_i + out_idx + (width << 2));

            hn::Store(hn::Add(y01_r, v1_r), d, out_r + out_idx + width);
            hn::Store(hn::Add(y01_i, v1_i), d, out_i + out_idx + width);
            hn::Store(hn::Sub(y01_r, v1_r), d, out_r + out_idx + width * 5);
            hn::Store(hn::Sub(y01_i, v1_i), d, out_i + out_idx + width * 5);

            hn::Store(hn::Add(y02_r, v2_r), d, out_r + out_idx + (width << 1));
            hn::Store(hn::Add(y02_i, v2_i), d, out_i + out_idx + (width << 1));
            hn::Store(hn::Sub(y02_r, v2_r), d, out_r + out_idx + width * 6);
            hn::Store(hn::Sub(y02_i, v2_i), d, out_i + out_idx + width * 6);

            hn::Store(hn::Add(y03_r, v3_r), d, out_r + out_idx + width * 3);
            hn::Store(hn::Add(y03_i, v3_i), d, out_i + out_idx + width * 3);
            hn::Store(hn::Sub(y03_r, v3_r), d, out_r + out_idx + width * 7);
            hn::Store(hn::Sub(y03_i, v3_i), d, out_i + out_idx + width * 7);
        }
    }

    template <typename F>
    inline void radix8_first_pass_fused(const std::complex<F>* __restrict in,
                                        F* __restrict out_r, F* __restrict out_i, const size_t n) {
        const size_t m = n >> 3;
        static constexpr hn::ScalableTag<F> d;
        static constexpr size_t lanes = hn::Lanes(d);

        static constexpr F kInvSqrt2 = static_cast<F>(1.0 / std::numbers::sqrt2);
        const auto inv_sqrt2 = hn::Set(d, kInvSqrt2);

        for (size_t j = 0; j + lanes <= m; j += lanes) {
            hn::Vec<decltype(d)> temp_a_r, temp_a_i, temp_b_r, temp_b_i;

            hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j), temp_a_r, temp_a_i);
            hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j + (m << 2)), temp_b_r, temp_b_i);
            const auto t0_r = hn::Add(temp_a_r, temp_b_r), t0_i = hn::Add(temp_a_i, temp_b_i);
            const auto t1_r = hn::Sub(temp_a_r, temp_b_r), t1_i = hn::Sub(temp_a_i, temp_b_i);

            hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j + (m << 1)), temp_a_r, temp_a_i);
            hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j + m * 6), temp_b_r, temp_b_i);
            const auto t2_r = hn::Add(temp_a_r, temp_b_r), t2_i = hn::Add(temp_a_i, temp_b_i);
            const auto t3_r = hn::Sub(temp_a_r, temp_b_r), t3_i = hn::Sub(temp_a_i, temp_b_i);

            const auto y00_r = hn::Add(t0_r, t2_r), y00_i = hn::Add(t0_i, t2_i);
            const auto y01_r = hn::Add(t1_r, t3_i), y01_i = hn::Sub(t1_i, t3_r);
            const auto y02_r = hn::Sub(t0_r, t2_r), y02_i = hn::Sub(t0_i, t2_i);
            const auto y03_r = hn::Sub(t1_r, t3_i), y03_i = hn::Add(t1_i, t3_r);

            hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j + m), temp_a_r, temp_a_i);
            hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j + 5 * m), temp_b_r, temp_b_i);
            const auto u0_r = hn::Add(temp_a_r, temp_b_r), u0_i = hn::Add(temp_a_i, temp_b_i);
            const auto u1_r = hn::Sub(temp_a_r, temp_b_r), u1_i = hn::Sub(temp_a_i, temp_b_i);

            hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j + 3 * m), temp_a_r, temp_a_i);
            hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + j + 7 * m), temp_b_r, temp_b_i);
            const auto u2_r = hn::Add(temp_a_r, temp_b_r), u2_i = hn::Add(temp_a_i, temp_b_i);
            const auto u3_r = hn::Sub(temp_a_r, temp_b_r), u3_i = hn::Sub(temp_a_i, temp_b_i);

            const auto y10_r = hn::Add(u0_r, u2_r), y10_i = hn::Add(u0_i, u2_i);
            const auto y11_r = hn::Add(u1_r, u3_i), y11_i = hn::Sub(u1_i, u3_r);
            const auto y12_r = hn::Sub(u0_r, u2_r), y12_i = hn::Sub(u0_i, u2_i);
            const auto y13_r = hn::Sub(u1_r, u3_i), y13_i = hn::Add(u1_i, u3_r);

            const auto v0_r = y10_r, v0_i = y10_i;
            const auto v1_r = hn::Mul(hn::Add(y11_r, y11_i), inv_sqrt2);
            const auto v1_i = hn::Mul(hn::Sub(y11_i, y11_r), inv_sqrt2);
            const auto v2_r = y12_i, v2_i = hn::Neg(y12_r);
            const auto v3_r = hn::Mul(hn::Sub(y13_i, y13_r), inv_sqrt2);
            const auto v3_i = hn::Mul(hn::Neg(hn::Add(y13_r, y13_i)), inv_sqrt2);

            const auto z00_r = hn::Add(y00_r, v0_r), z00_i = hn::Add(y00_i, v0_i);
            const auto z01_r = hn::Add(y01_r, v1_r), z01_i = hn::Add(y01_i, v1_i);
            const auto z02_r = hn::Add(y02_r, v2_r), z02_i = hn::Add(y02_i, v2_i);
            const auto z03_r = hn::Add(y03_r, v3_r), z03_i = hn::Add(y03_i, v3_i);

            const auto z10_r = hn::Sub(y00_r, v0_r), z10_i = hn::Sub(y00_i, v0_i);
            const auto z11_r = hn::Sub(y01_r, v1_r), z11_i = hn::Sub(y01_i, v1_i);
            const auto z12_r = hn::Sub(y02_r, v2_r), z12_i = hn::Sub(y02_i, v2_i);
            const auto z13_r = hn::Sub(y03_r, v3_r), z13_i = hn::Sub(y03_i, v3_i);

            const auto lower_r0 = hn::InterleaveLower(d, z00_r, z10_r);
            const auto lower_r1 = hn::InterleaveLower(d, z01_r, z11_r);
            const auto lower_r2 = hn::InterleaveLower(d, z02_r, z12_r);
            const auto lower_r3 = hn::InterleaveLower(d, z03_r, z13_r);

            const auto upper_r0 = hn::InterleaveUpper(d, z00_r, z10_r);
            const auto upper_r1 = hn::InterleaveUpper(d, z01_r, z11_r);
            const auto upper_r2 = hn::InterleaveUpper(d, z02_r, z12_r);
            const auto upper_r3 = hn::InterleaveUpper(d, z03_r, z13_r);

            const auto lower_i0 = hn::InterleaveLower(d, z00_i, z10_i);
            const auto lower_i1 = hn::InterleaveLower(d, z01_i, z11_i);
            const auto lower_i2 = hn::InterleaveLower(d, z02_i, z12_i);
            const auto lower_i3 = hn::InterleaveLower(d, z03_i, z13_i);

            const auto upper_i0 = hn::InterleaveUpper(d, z00_i, z10_i);
            const auto upper_i1 = hn::InterleaveUpper(d, z01_i, z11_i);
            const auto upper_i2 = hn::InterleaveUpper(d, z02_i, z12_i);
            const auto upper_i3 = hn::InterleaveUpper(d, z03_i, z13_i);

            const size_t out_offset = j << 3;
            if constexpr (lanes <= 4) {
                hn::StoreInterleaved4(lower_r0, lower_r1, lower_r2, lower_r3, d, out_r + out_offset);
                hn::StoreInterleaved4(upper_r0, upper_r1, upper_r2, upper_r3, d, out_r + out_offset + (lanes << 2));
                hn::StoreInterleaved4(lower_i0, lower_i1, lower_i2, lower_i3, d, out_i + out_offset);
                hn::StoreInterleaved4(upper_i0, upper_i1, upper_i2, upper_i3, d, out_i + out_offset + (lanes << 2));
            } else {
                const auto store0_r0 = hn::ConcatLowerLower(d, upper_r0, lower_r0);
                const auto store0_r1 = hn::ConcatLowerLower(d, upper_r1, lower_r1);
                const auto store0_r2 = hn::ConcatLowerLower(d, upper_r2, lower_r2);
                const auto store0_r3 = hn::ConcatLowerLower(d, upper_r3, lower_r3);

                const auto store1_r0 = hn::ConcatUpperUpper(d, upper_r0, lower_r0);
                const auto store1_r1 = hn::ConcatUpperUpper(d, upper_r1, lower_r1);
                const auto store1_r2 = hn::ConcatUpperUpper(d, upper_r2, lower_r2);
                const auto store1_r3 = hn::ConcatUpperUpper(d, upper_r3, lower_r3);

                const auto store0_i0 = hn::ConcatLowerLower(d, upper_i0, lower_i0);
                const auto store0_i1 = hn::ConcatLowerLower(d, upper_i1, lower_i1);
                const auto store0_i2 = hn::ConcatLowerLower(d, upper_i2, lower_i2);
                const auto store0_i3 = hn::ConcatLowerLower(d, upper_i3, lower_i3);

                const auto store1_i0 = hn::ConcatUpperUpper(d, upper_i0, lower_i0);
                const auto store1_i1 = hn::ConcatUpperUpper(d, upper_i1, lower_i1);
                const auto store1_i2 = hn::ConcatUpperUpper(d, upper_i2, lower_i2);
                const auto store1_i3 = hn::ConcatUpperUpper(d, upper_i3, lower_i3);

                hn::StoreInterleaved4(store0_r0, store0_r1, store0_r2, store0_r3, d, out_r + out_offset);
                hn::StoreInterleaved4(store1_r0, store1_r1, store1_r2, store1_r3, d, out_r + out_offset + (lanes << 2));

                hn::StoreInterleaved4(store0_i0, store0_i1, store0_i2, store0_i3, d, out_i + out_offset);
                hn::StoreInterleaved4(store1_i0, store1_i1, store1_i2, store1_i3, d, out_i + out_offset + (lanes << 2));
            }
        }
    }

    template <typename F>
    inline void callback_order_0(const std::complex<F>* __restrict in, std::complex<F>* __restrict out) {
        out[0] = in[0];
    }

    template <typename F>
    inline void callback_order_1(const std::complex<F>* __restrict in, std::complex<F>* __restrict out) {
        out[0] = in[0] + in[1];
        out[1] = in[0] - in[1];
    }

    template <typename F>
    inline void callback_order_2(const std::complex<F>* __restrict in, std::complex<F>* __restrict out) {
        const auto t0 = in[0] + in[2];
        const auto t1 = in[0] - in[2];
        const auto t2 = in[1] + in[3];
        const auto t3 = in[1] - in[3];
        out[0] = t0 + t2;
        out[2] = t0 - t2;
        out[1] = t1 + std::complex<F>(t3.imag(), -t3.real());
        out[3] = t1 - std::complex<F>(t3.imag(), -t3.real());
    }

    template <typename F>
    inline void callback_order_3(const std::complex<F>* __restrict in, std::complex<F>* __restrict out) {
        static constexpr F kInvSqrt2 = static_cast<F>(1.0 / std::numbers::sqrt2);
        const auto x0_r = in[0].real(), x0_i = in[0].imag();
        const auto x1_r = in[1].real(), x1_i = in[1].imag();
        const auto x2_r = in[2].real(), x2_i = in[2].imag();
        const auto x3_r = in[3].real(), x3_i = in[3].imag();
        const auto x4_r = in[4].real(), x4_i = in[4].imag();
        const auto x5_r = in[5].real(), x5_i = in[5].imag();
        const auto x6_r = in[6].real(), x6_i = in[6].imag();
        const auto x7_r = in[7].real(), x7_i = in[7].imag();

        const auto t0_r = x0_r + x4_r, t0_i = x0_i + x4_i;
        const auto t1_r = x0_r - x4_r, t1_i = x0_i - x4_i;
        const auto t2_r = x2_r + x6_r, t2_i = x2_i + x6_i;
        const auto t3_r = x2_r - x6_r, t3_i = x2_i - x6_i;

        const auto y00_r = t0_r + t2_r, y00_i = t0_i + t2_i;
        const auto y01_r = t1_r + t3_i, y01_i = t1_i - t3_r;
        const auto y02_r = t0_r - t2_r, y02_i = t0_i - t2_i;
        const auto y03_r = t1_r - t3_i, y03_i = t1_i + t3_r;

        const auto u0_r = x1_r + x5_r, u0_i = x1_i + x5_i;
        const auto u1_r = x1_r - x5_r, u1_i = x1_i - x5_i;
        const auto u2_r = x3_r + x7_r, u2_i = x3_i + x7_i;
        const auto u3_r = x3_r - x7_r, u3_i = x3_i - x7_i;

        const auto y10_r = u0_r + u2_r, y10_i = u0_i + u2_i;
        const auto y11_r = u1_r + u3_i, y11_i = u1_i - u3_r;
        const auto y12_r = u0_r - u2_r, y12_i = u0_i - u2_i;
        const auto y13_r = u1_r - u3_i, y13_i = u1_i + u3_r;

        const auto v0_r = y10_r, v0_i = y10_i;
        const auto v1_r = (y11_r + y11_i) * kInvSqrt2, v1_i = (y11_i - y11_r) * kInvSqrt2;
        const auto v2_r = y12_i, v2_i = -y12_r;
        const auto v3_r = (y13_i - y13_r) * kInvSqrt2, v3_i = -(y13_r + y13_i) * kInvSqrt2;

        out[0] = std::complex<F>(y00_r + v0_r, y00_i + v0_i);
        out[1] = std::complex<F>(y01_r + v1_r, y01_i + v1_i);
        out[2] = std::complex<F>(y02_r + v2_r, y02_i + v2_i);
        out[3] = std::complex<F>(y03_r + v3_r, y03_i + v3_i);
        out[4] = std::complex<F>(y00_r - v0_r, y00_i - v0_i);
        out[5] = std::complex<F>(y01_r - v1_r, y01_i - v1_i);
        out[6] = std::complex<F>(y02_r - v2_r, y02_i - v2_i);
        out[7] = std::complex<F>(y03_r - v3_r, y03_i - v3_i);
    }

    template <typename F>
    inline void callback_order_4(const std::complex<F>* in, std::complex<F>* out,
                                 const F* w_r_base, const F* w_i_base) {
        constexpr size_t N = HWY_MAX_BYTES / sizeof(F);

        alignas(64) F tmp_r[16];
        alignas(64) F tmp_i[16];

        if constexpr (N >= 8) {
            hn::FixedTag<F, 8> d8;
            hn::FixedTag<F, 4> d4;

            hn::Vec<decltype(d8)> v0_r, v0_i, v1_r, v1_i;
            hn::LoadInterleaved2(d8, reinterpret_cast<const F*>(in), v0_r, v0_i);
            hn::LoadInterleaved2(d8, reinterpret_cast<const F*>(in + 8), v1_r, v1_i);

            const auto t02_r = hn::Add(v0_r, v1_r), t02_i = hn::Add(v0_i, v1_i);
            const auto t13_r = hn::Sub(v0_r, v1_r), t13_i = hn::Sub(v0_i, v1_i);

            const auto A_r = hn::ConcatLowerLower(d8, t13_r, t02_r);
            const auto A_i = hn::ConcatLowerLower(d8, t13_i, t02_i);
            const auto B_r = hn::ConcatUpperUpper(d8, t13_i, t02_r);
            const auto B_i = hn::ConcatUpperUpper(d8, t13_r, t02_i);

            const auto out01_r = hn::Add(A_r, B_r);
            const auto out23_r = hn::Sub(A_r, B_r);
            const auto sum_i = hn::Add(A_i, B_i);
            const auto diff_i = hn::Sub(A_i, B_i);

            hn::Store(hn::LowerHalf(d4, out01_r), d4, tmp_r);
            hn::Store(hn::UpperHalf(d4, out01_r), d4, tmp_r + 4);
            hn::Store(hn::LowerHalf(d4, out23_r), d4, tmp_r + 8);
            hn::Store(hn::UpperHalf(d4, out23_r), d4, tmp_r + 12);

            hn::Store(hn::LowerHalf(d4, sum_i), d4, tmp_i);
            hn::Store(hn::UpperHalf(d4, diff_i), d4, tmp_i + 4);
            hn::Store(hn::LowerHalf(d4, diff_i), d4, tmp_i + 8);
            hn::Store(hn::UpperHalf(d4, sum_i), d4, tmp_i + 12);

            const auto r0 = hn::Load(d4, tmp_r), i0 = hn::Load(d4, tmp_i);
            const auto r1 = hn::Load(d4, tmp_r + 4), i1 = hn::Load(d4, tmp_i + 4);
            const auto r2 = hn::Load(d4, tmp_r + 8), i2 = hn::Load(d4, tmp_i + 8);
            const auto r3 = hn::Load(d4, tmp_r + 12), i3 = hn::Load(d4, tmp_i + 12);

            const auto w1_r = hn::Load(d4, w_r_base), w1_i = hn::Load(d4, w_i_base);
            const auto w2_r = hn::Load(d4, w_r_base + 4), w2_i = hn::Load(d4, w_i_base + 4);
            const auto w3_r = hn::Load(d4, w_r_base + 8), w3_i = hn::Load(d4, w_i_base + 8);

            const auto t1_r = hn::NegMulAdd(i1, w1_i, hn::Mul(r1, w1_r));
            const auto t1_i = hn::MulAdd(i1, w1_r, hn::Mul(r1, w1_i));
            const auto t2_r = hn::NegMulAdd(i2, w2_i, hn::Mul(r2, w2_r));
            const auto t2_i = hn::MulAdd(i2, w2_r, hn::Mul(r2, w2_i));
            const auto t3_r = hn::NegMulAdd(i3, w3_i, hn::Mul(r3, w3_r));
            const auto t3_i = hn::MulAdd(i3, w3_r, hn::Mul(r3, w3_i));

            const auto s0_r = hn::Add(r0, t2_r), s0_i = hn::Add(i0, t2_i);
            const auto s1_r = hn::Sub(r0, t2_r), s1_i = hn::Sub(i0, t2_i);
            const auto s2_r = hn::Add(t1_r, t3_r), s2_i = hn::Add(t1_i, t3_i);
            const auto s3_r = hn::Sub(t1_r, t3_r), s3_i = hn::Sub(t1_i, t3_i);

            const auto f0_r = hn::Add(s0_r, s2_r), f0_i = hn::Add(s0_i, s2_i);
            const auto f1_r = hn::Add(s1_r, s3_i), f1_i = hn::Sub(s1_i, s3_r);
            const auto f2_r = hn::Sub(s0_r, s2_r), f2_i = hn::Sub(s0_i, s2_i);
            const auto f3_r = hn::Sub(s1_r, s3_i), f3_i = hn::Add(s1_i, s3_r);

            hn::StoreInterleaved2(hn::Combine(d8, f1_r, f0_r), hn::Combine(d8, f1_i, f0_i), d8,
                                  reinterpret_cast<F*>(out));
            hn::StoreInterleaved2(hn::Combine(d8, f3_r, f2_r), hn::Combine(d8, f3_i, f2_i), d8,
                                  reinterpret_cast<F*>(out + 8));

        } else {
            hn::FixedTag<F, (N < 4 ? N : 4)> d_cap;

            for (size_t i = 0; i < 4; i += hn::Lanes(d_cap)) {
                hn::Vec<decltype(d_cap)> x0_r, x0_i, x1_r, x1_i, x2_r, x2_i, x3_r, x3_i;

                hn::LoadInterleaved2(d_cap, reinterpret_cast<const F*>(in + i), x0_r, x0_i);
                hn::LoadInterleaved2(d_cap, reinterpret_cast<const F*>(in + i + 8), x2_r, x2_i);

                const auto t0_r = hn::Add(x0_r, x2_r), t0_i = hn::Add(x0_i, x2_i);
                const auto t1_r = hn::Sub(x0_r, x2_r), t1_i = hn::Sub(x0_i, x2_i);

                hn::LoadInterleaved2(d_cap, reinterpret_cast<const F*>(in + i + 4), x1_r, x1_i);
                hn::LoadInterleaved2(d_cap, reinterpret_cast<const F*>(in + i + 12), x3_r, x3_i);

                const auto t2_r = hn::Add(x1_r, x3_r), t2_i = hn::Add(x1_i, x3_i);
                const auto t3_r = hn::Sub(x1_r, x3_r), t3_i = hn::Sub(x1_i, x3_i);

                const auto out0_r = hn::Add(t0_r, t2_r), out0_i = hn::Add(t0_i, t2_i);
                const auto out2_r = hn::Sub(t0_r, t2_r), out2_i = hn::Sub(t0_i, t2_i);
                const auto out1_r = hn::Add(t1_r, t3_i), out1_i = hn::Sub(t1_i, t3_r);
                const auto out3_r = hn::Sub(t1_r, t3_i), out3_i = hn::Add(t1_i, t3_r);

                hn::StoreInterleaved4(out0_r, out1_r, out2_r, out3_r, d_cap, tmp_r + i * 4);
                hn::StoreInterleaved4(out0_i, out1_i, out2_i, out3_i, d_cap, tmp_i + i * 4);
            }

            for (size_t i = 0; i < 4; i += hn::Lanes(d_cap)) {
                const auto i1 = hn::Load(d_cap, tmp_i + 4 + i), r1 = hn::Load(d_cap, tmp_r + 4 + i);
                const auto w1_r = hn::Load(d_cap, w_r_base + i), w1_i = hn::Load(d_cap, w_i_base + i);
                const auto t1_r = hn::NegMulAdd(i1, w1_i, hn::Mul(r1, w1_r));
                const auto t1_i = hn::MulAdd(i1, w1_r, hn::Mul(r1, w1_i));

                const auto i3 = hn::Load(d_cap, tmp_i + 12 + i), r3 = hn::Load(d_cap, tmp_r + 12 + i);
                const auto w3_r = hn::Load(d_cap, w_r_base + 8 + i), w3_i = hn::Load(d_cap, w_i_base + 8 + i);
                const auto t3_r = hn::NegMulAdd(i3, w3_i, hn::Mul(r3, w3_r));
                const auto t3_i = hn::MulAdd(i3, w3_r, hn::Mul(r3, w3_i));

                const auto s2_r = hn::Add(t1_r, t3_r), s2_i = hn::Add(t1_i, t3_i);
                const auto s3_r = hn::Sub(t1_r, t3_r), s3_i = hn::Sub(t1_i, t3_i);

                const auto i2 = hn::Load(d_cap, tmp_i + 8 + i), r2 = hn::Load(d_cap, tmp_r + 8 + i);
                const auto w2_r = hn::Load(d_cap, w_r_base + 4 + i), w2_i = hn::Load(d_cap, w_i_base + 4 + i);
                const auto t2_r = hn::NegMulAdd(i2, w2_i, hn::Mul(r2, w2_r));
                const auto t2_i = hn::MulAdd(i2, w2_r, hn::Mul(r2, w2_i));

                const auto r0 = hn::Load(d_cap, tmp_r + i), i0 = hn::Load(d_cap, tmp_i + i);
                const auto s0_r = hn::Add(r0, t2_r), s0_i = hn::Add(i0, t2_i);
                const auto s1_r = hn::Sub(r0, t2_r), s1_i = hn::Sub(i0, t2_i);

                hn::StoreInterleaved2(hn::Add(s0_r, s2_r), hn::Add(s0_i, s2_i), d_cap, reinterpret_cast<F*>(out + i));
                hn::StoreInterleaved2(hn::Add(s1_r, s3_i), hn::Sub(s1_i, s3_r), d_cap,
                                      reinterpret_cast<F*>(out + 4 + i));
                hn::StoreInterleaved2(hn::Sub(s0_r, s2_r), hn::Sub(s0_i, s2_i), d_cap,
                                      reinterpret_cast<F*>(out + 8 + i));
                hn::StoreInterleaved2(hn::Sub(s1_r, s3_i), hn::Add(s1_i, s3_r), d_cap,
                                      reinterpret_cast<F*>(out + 12 + i));
            }
        }
    }

    template <typename F>
    inline void callback_order_5(const std::complex<F>* in_ptr, std::complex<F>* out_ptr,
                                 const F* w_r_base, const F* w_i_base) {
        namespace hn = hwy::HWY_NAMESPACE;
        using D_Max = hn::ScalableTag<F>;
        static constexpr size_t N = hn::Lanes(D_Max());
        static constexpr F kInvSqrt2 = static_cast<F>(1.0 / std::numbers::sqrt2);

        alignas(64) F tmp_r[32];
        alignas(64) F tmp_i[32];

        if constexpr (N >= 8) {
            const hn::CappedTag<F, 8> d8;
            const hn::CappedTag<F, 4> d4;
            const auto inv_sqrt2 = hn::Set(d4, kInvSqrt2);

            hn::Vec<decltype(d8)> vec_in0_r, vec_in0_i, vec_in16_r, vec_in16_i;
            hn::Vec<decltype(d8)> vec_in8_r, vec_in8_i, vec_in24_r, vec_in24_i;

            hn::LoadInterleaved2(d8, reinterpret_cast<const F*>(in_ptr), vec_in0_r, vec_in0_i);
            hn::LoadInterleaved2(d8, reinterpret_cast<const F*>(in_ptr + 16), vec_in16_r, vec_in16_i);
            const auto vec_t0_r = hn::Add(vec_in0_r, vec_in16_r), vec_t0_i = hn::Add(vec_in0_i, vec_in16_i);
            const auto vec_t1_r = hn::Sub(vec_in0_r, vec_in16_r), vec_t1_i = hn::Sub(vec_in0_i, vec_in16_i);

            hn::LoadInterleaved2(d8, reinterpret_cast<const F*>(in_ptr + 8), vec_in8_r, vec_in8_i);
            hn::LoadInterleaved2(d8, reinterpret_cast<const F*>(in_ptr + 24), vec_in24_r, vec_in24_i);
            const auto vec_t2_r = hn::Add(vec_in8_r, vec_in24_r), vec_t2_i = hn::Add(vec_in8_i, vec_in24_i);
            const auto vec_t3_r = hn::Sub(vec_in8_r, vec_in24_r), vec_t3_i = hn::Sub(vec_in8_i, vec_in24_i);

            const auto vec_y00_r = hn::Add(vec_t0_r, vec_t2_r), vec_y00_i = hn::Add(vec_t0_i, vec_t2_i);
            const auto vec_y01_r = hn::Add(vec_t1_r, vec_t3_i), vec_y01_i = hn::Sub(vec_t1_i, vec_t3_r);
            const auto vec_y02_r = hn::Sub(vec_t0_r, vec_t2_r), vec_y02_i = hn::Sub(vec_t0_i, vec_t2_i);
            const auto vec_y03_r = hn::Sub(vec_t1_r, vec_t3_i), vec_y03_i = hn::Add(vec_t1_i, vec_t3_r);

            const auto y00_r = hn::LowerHalf(d4, vec_y00_r), y00_i = hn::LowerHalf(d4, vec_y00_i);
            const auto y01_r = hn::LowerHalf(d4, vec_y01_r), y01_i = hn::LowerHalf(d4, vec_y01_i);
            const auto y02_r = hn::LowerHalf(d4, vec_y02_r), y02_i = hn::LowerHalf(d4, vec_y02_i);
            const auto y03_r = hn::LowerHalf(d4, vec_y03_r), y03_i = hn::LowerHalf(d4, vec_y03_i);

            const auto y10_r = hn::UpperHalf(d4, vec_y00_r), y10_i = hn::UpperHalf(d4, vec_y00_i);
            const auto y11_r = hn::UpperHalf(d4, vec_y01_r), y11_i = hn::UpperHalf(d4, vec_y01_i);
            const auto y12_r = hn::UpperHalf(d4, vec_y02_r), y12_i = hn::UpperHalf(d4, vec_y02_i);
            const auto y13_r = hn::UpperHalf(d4, vec_y03_r), y13_i = hn::UpperHalf(d4, vec_y03_i);

            const auto v0_r = y10_r, v0_i = y10_i;
            const auto v1_r = hn::Mul(hn::Add(y11_r, y11_i), inv_sqrt2), v1_i = hn::Mul(
                hn::Sub(y11_i, y11_r), inv_sqrt2);
            const auto v2_r = y12_i, v2_i = hn::Neg(y12_r);
            const auto v3_r = hn::Mul(hn::Sub(y13_i, y13_r), inv_sqrt2), v3_i = hn::Mul(
                hn::Neg(hn::Add(y13_r, y13_i)), inv_sqrt2);

            const auto z00_r = hn::Add(y00_r, v0_r), z00_i = hn::Add(y00_i, v0_i);
            const auto z01_r = hn::Add(y01_r, v1_r), z01_i = hn::Add(y01_i, v1_i);
            const auto z02_r = hn::Add(y02_r, v2_r), z02_i = hn::Add(y02_i, v2_i);
            const auto z03_r = hn::Add(y03_r, v3_r), z03_i = hn::Add(y03_i, v3_i);

            const auto z10_r = hn::Sub(y00_r, v0_r), z10_i = hn::Sub(y00_i, v0_i);
            const auto z11_r = hn::Sub(y01_r, v1_r), z11_i = hn::Sub(y01_i, v1_i);
            const auto z12_r = hn::Sub(y02_r, v2_r), z12_i = hn::Sub(y02_i, v2_i);
            const auto z13_r = hn::Sub(y03_r, v3_r), z13_i = hn::Sub(y03_i, v3_i);

            const auto lower_r0 = hn::InterleaveLower(d4, z00_r, z10_r), upper_r0 = hn::InterleaveUpper(
                d4, z00_r, z10_r);
            const auto lower_r1 = hn::InterleaveLower(d4, z01_r, z11_r), upper_r1 = hn::InterleaveUpper(
                d4, z01_r, z11_r);
            const auto lower_r2 = hn::InterleaveLower(d4, z02_r, z12_r), upper_r2 = hn::InterleaveUpper(
                d4, z02_r, z12_r);
            const auto lower_r3 = hn::InterleaveLower(d4, z03_r, z13_r), upper_r3 = hn::InterleaveUpper(
                d4, z03_r, z13_r);

            const auto lower_i0 = hn::InterleaveLower(d4, z00_i, z10_i), upper_i0 = hn::InterleaveUpper(
                d4, z00_i, z10_i);
            const auto lower_i1 = hn::InterleaveLower(d4, z01_i, z11_i), upper_i1 = hn::InterleaveUpper(
                d4, z01_i, z11_i);
            const auto lower_i2 = hn::InterleaveLower(d4, z02_i, z12_i), upper_i2 = hn::InterleaveUpper(
                d4, z02_i, z12_i);
            const auto lower_i3 = hn::InterleaveLower(d4, z03_i, z13_i), upper_i3 = hn::InterleaveUpper(
                d4, z03_i, z13_i);

            hn::StoreInterleaved4(lower_r0, lower_r1, lower_r2, lower_r3, d4, tmp_r);
            hn::StoreInterleaved4(upper_r0, upper_r1, upper_r2, upper_r3, d4, tmp_r + 16);
            hn::StoreInterleaved4(lower_i0, lower_i1, lower_i2, lower_i3, d4, tmp_i);
            hn::StoreInterleaved4(upper_i0, upper_i1, upper_i2, upper_i3, d4, tmp_i + 16);

            const auto i1 = hn::Load(d8, tmp_i + 8), r1 = hn::Load(d8, tmp_r + 8);
            const auto w1_r = hn::Load(d8, w_r_base), w1_i = hn::Load(d8, w_i_base);
            const auto t1_r = hn::NegMulAdd(i1, w1_i, hn::Mul(r1, w1_r));
            const auto t1_i = hn::MulAdd(i1, w1_r, hn::Mul(r1, w1_i));

            const auto i3 = hn::Load(d8, tmp_i + 24), r3 = hn::Load(d8, tmp_r + 24);
            const auto w3_r = hn::Load(d8, w_r_base + 16), w3_i = hn::Load(d8, w_i_base + 16);
            const auto t3_r = hn::NegMulAdd(i3, w3_i, hn::Mul(r3, w3_r));
            const auto t3_i = hn::MulAdd(i3, w3_r, hn::Mul(r3, w3_i));

            const auto s2_r = hn::Add(t1_r, t3_r), s2_i = hn::Add(t1_i, t3_i);
            const auto s3_r = hn::Sub(t1_r, t3_r), s3_i = hn::Sub(t1_i, t3_i);

            const auto i2 = hn::Load(d8, tmp_i + 16), r2 = hn::Load(d8, tmp_r + 16);
            const auto w2_r = hn::Load(d8, w_r_base + 8), w2_i = hn::Load(d8, w_i_base + 8);
            const auto t2_r = hn::NegMulAdd(i2, w2_i, hn::Mul(r2, w2_r));
            const auto t2_i = hn::MulAdd(i2, w2_r, hn::Mul(r2, w2_i));

            const auto r0 = hn::Load(d8, tmp_r), i0 = hn::Load(d8, tmp_i);
            const auto s0_r = hn::Add(r0, t2_r), s0_i = hn::Add(i0, t2_i);
            const auto s1_r = hn::Sub(r0, t2_r), s1_i = hn::Sub(i0, t2_i);

            hn::StoreInterleaved2(hn::Add(s0_r, s2_r), hn::Add(s0_i, s2_i), d8, reinterpret_cast<F*>(out_ptr));
            hn::StoreInterleaved2(hn::Add(s1_r, s3_i), hn::Sub(s1_i, s3_r), d8, reinterpret_cast<F*>(out_ptr + 8));
            hn::StoreInterleaved2(hn::Sub(s0_r, s2_r), hn::Sub(s0_i, s2_i), d8, reinterpret_cast<F*>(out_ptr + 16));
            hn::StoreInterleaved2(hn::Sub(s1_r, s3_i), hn::Add(s1_i, s3_r), d8, reinterpret_cast<F*>(out_ptr + 24));

        } else if constexpr (N >= 4) {
            const hn::CappedTag<F, 4> d;
            {
                const auto inv_sqrt2 = hn::Set(d, kInvSqrt2);
                hn::Vec<decltype(d)> temp_a_r, temp_a_i, temp_b_r, temp_b_i;

                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in_ptr), temp_a_r, temp_a_i);
                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in_ptr + 16), temp_b_r, temp_b_i);
                const auto t0_r = hn::Add(temp_a_r, temp_b_r), t0_i = hn::Add(temp_a_i, temp_b_i);
                const auto t1_r = hn::Sub(temp_a_r, temp_b_r), t1_i = hn::Sub(temp_a_i, temp_b_i);

                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in_ptr + 8), temp_a_r, temp_a_i);
                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in_ptr + 24), temp_b_r, temp_b_i);
                const auto t2_r = hn::Add(temp_a_r, temp_b_r), t2_i = hn::Add(temp_a_i, temp_b_i);
                const auto t3_r = hn::Sub(temp_a_r, temp_b_r), t3_i = hn::Sub(temp_a_i, temp_b_i);

                const auto y00_r = hn::Add(t0_r, t2_r), y00_i = hn::Add(t0_i, t2_i);
                const auto y01_r = hn::Add(t1_r, t3_i), y01_i = hn::Sub(t1_i, t3_r);
                const auto y02_r = hn::Sub(t0_r, t2_r), y02_i = hn::Sub(t0_i, t2_i);
                const auto y03_r = hn::Sub(t1_r, t3_i), y03_i = hn::Add(t1_i, t3_r);

                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in_ptr + 4), temp_a_r, temp_a_i);
                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in_ptr + 20), temp_b_r, temp_b_i);
                const auto u0_r = hn::Add(temp_a_r, temp_b_r), u0_i = hn::Add(temp_a_i, temp_b_i);
                const auto u1_r = hn::Sub(temp_a_r, temp_b_r), u1_i = hn::Sub(temp_a_i, temp_b_i);

                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in_ptr + 12), temp_a_r, temp_a_i);
                hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in_ptr + 28), temp_b_r, temp_b_i);
                const auto u2_r = hn::Add(temp_a_r, temp_b_r), u2_i = hn::Add(temp_a_i, temp_b_i);
                const auto u3_r = hn::Sub(temp_a_r, temp_b_r), u3_i = hn::Sub(temp_a_i, temp_b_i);

                const auto y10_r = hn::Add(u0_r, u2_r), y10_i = hn::Add(u0_i, u2_i);
                const auto y11_r = hn::Add(u1_r, u3_i), y11_i = hn::Sub(u1_i, u3_r);
                const auto y12_r = hn::Sub(u0_r, u2_r), y12_i = hn::Sub(u0_i, u2_i);
                const auto y13_r = hn::Sub(u1_r, u3_i), y13_i = hn::Add(u1_i, u3_r);

                const auto v0_r = y10_r, v0_i = y10_i;
                const auto v1_r = hn::Mul(hn::Add(y11_r, y11_i), inv_sqrt2), v1_i = hn::Mul(
                    hn::Sub(y11_i, y11_r), inv_sqrt2);
                const auto v2_r = y12_i, v2_i = hn::Neg(y12_r);
                const auto v3_r = hn::Mul(hn::Sub(y13_i, y13_r), inv_sqrt2), v3_i = hn::Mul(
                    hn::Neg(hn::Add(y13_r, y13_i)), inv_sqrt2);

                const auto z00_r = hn::Add(y00_r, v0_r), z00_i = hn::Add(y00_i, v0_i);
                const auto z01_r = hn::Add(y01_r, v1_r), z01_i = hn::Add(y01_i, v1_i);
                const auto z02_r = hn::Add(y02_r, v2_r), z02_i = hn::Add(y02_i, v2_i);
                const auto z03_r = hn::Add(y03_r, v3_r), z03_i = hn::Add(y03_i, v3_i);

                const auto z10_r = hn::Sub(y00_r, v0_r), z10_i = hn::Sub(y00_i, v0_i);
                const auto z11_r = hn::Sub(y01_r, v1_r), z11_i = hn::Sub(y01_i, v1_i);
                const auto z12_r = hn::Sub(y02_r, v2_r), z12_i = hn::Sub(y02_i, v2_i);
                const auto z13_r = hn::Sub(y03_r, v3_r), z13_i = hn::Sub(y03_i, v3_i);

                const auto lower_r0 = hn::InterleaveLower(d, z00_r, z10_r), upper_r0 = hn::InterleaveUpper(
                    d, z00_r, z10_r);
                const auto lower_r1 = hn::InterleaveLower(d, z01_r, z11_r), upper_r1 = hn::InterleaveUpper(
                    d, z01_r, z11_r);
                const auto lower_r2 = hn::InterleaveLower(d, z02_r, z12_r), upper_r2 = hn::InterleaveUpper(
                    d, z02_r, z12_r);
                const auto lower_r3 = hn::InterleaveLower(d, z03_r, z13_r), upper_r3 = hn::InterleaveUpper(
                    d, z03_r, z13_r);

                const auto lower_i0 = hn::InterleaveLower(d, z00_i, z10_i), upper_i0 = hn::InterleaveUpper(
                    d, z00_i, z10_i);
                const auto lower_i1 = hn::InterleaveLower(d, z01_i, z11_i), upper_i1 = hn::InterleaveUpper(
                    d, z01_i, z11_i);
                const auto lower_i2 = hn::InterleaveLower(d, z02_i, z12_i), upper_i2 = hn::InterleaveUpper(
                    d, z02_i, z12_i);
                const auto lower_i3 = hn::InterleaveLower(d, z03_i, z13_i), upper_i3 = hn::InterleaveUpper(
                    d, z03_i, z13_i);

                hn::StoreInterleaved4(lower_r0, lower_r1, lower_r2, lower_r3, d, tmp_r);
                hn::StoreInterleaved4(upper_r0, upper_r1, upper_r2, upper_r3, d, tmp_r + 16);
                hn::StoreInterleaved4(lower_i0, lower_i1, lower_i2, lower_i3, d, tmp_i);
                hn::StoreInterleaved4(upper_i0, upper_i1, upper_i2, upper_i3, d, tmp_i + 16);
            }

            static constexpr size_t off1 = (sizeof(F) == 8) ? 16 : 8;
            static constexpr size_t off2 = (sizeof(F) == 8) ? 8 : 16;

            for (size_t k = 0; k < 8; k += 4) {
                const auto i1 = hn::Load(d, tmp_i + off1 + k), r1 = hn::Load(d, tmp_r + off1 + k);
                const auto w1_r = hn::Load(d, w_r_base + k), w1_i = hn::Load(d, w_i_base + k);
                const auto t1_r = hn::NegMulAdd(i1, w1_i, hn::Mul(r1, w1_r));
                const auto t1_i = hn::MulAdd(i1, w1_r, hn::Mul(r1, w1_i));

                const auto i3 = hn::Load(d, tmp_i + 24 + k), r3 = hn::Load(d, tmp_r + 24 + k);
                const auto w3_r = hn::Load(d, w_r_base + 16 + k), w3_i = hn::Load(d, w_i_base + 16 + k);
                const auto t3_r = hn::NegMulAdd(i3, w3_i, hn::Mul(r3, w3_r));
                const auto t3_i = hn::MulAdd(i3, w3_r, hn::Mul(r3, w3_i));

                const auto s2_r = hn::Add(t1_r, t3_r), s2_i = hn::Add(t1_i, t3_i);
                const auto s3_r = hn::Sub(t1_r, t3_r), s3_i = hn::Sub(t1_i, t3_i);

                const auto i2 = hn::Load(d, tmp_i + off2 + k), r2 = hn::Load(d, tmp_r + off2 + k);
                const auto w2_r = hn::Load(d, w_r_base + 8 + k), w2_i = hn::Load(d, w_i_base + 8 + k);
                const auto t2_r = hn::NegMulAdd(i2, w2_i, hn::Mul(r2, w2_r));
                const auto t2_i = hn::MulAdd(i2, w2_r, hn::Mul(r2, w2_i));

                const auto r0 = hn::Load(d, tmp_r + k), i0 = hn::Load(d, tmp_i + k);
                const auto s0_r = hn::Add(r0, t2_r), s0_i = hn::Add(i0, t2_i);
                const auto s1_r = hn::Sub(r0, t2_r), s1_i = hn::Sub(i0, t2_i);

                hn::StoreInterleaved2(hn::Add(s0_r, s2_r), hn::Add(s0_i, s2_i), d, reinterpret_cast<F*>(out_ptr + k));
                hn::StoreInterleaved2(hn::Add(s1_r, s3_i), hn::Sub(s1_i, s3_r), d,
                                      reinterpret_cast<F*>(out_ptr + 8 + k));
                hn::StoreInterleaved2(hn::Sub(s0_r, s2_r), hn::Sub(s0_i, s2_i), d,
                                      reinterpret_cast<F*>(out_ptr + 16 + k));
                hn::StoreInterleaved2(hn::Sub(s1_r, s3_i), hn::Add(s1_i, s3_r), d,
                                      reinterpret_cast<F*>(out_ptr + 24 + k));
            }
        } else {
            const hn::CappedTag<F, 2> d2;
            const auto inv_sqrt2 = hn::Set(d2, kInvSqrt2);

            for (size_t idx = 0; idx < 4; idx += 2) {
                hn::Vec<decltype(d2)> temp_a_r, temp_a_i, temp_b_r, temp_b_i;

                hn::LoadInterleaved2(d2, reinterpret_cast<const F*>(in_ptr + idx), temp_a_r, temp_a_i);
                hn::LoadInterleaved2(d2, reinterpret_cast<const F*>(in_ptr + 16 + idx), temp_b_r, temp_b_i);
                const auto t0_r = hn::Add(temp_a_r, temp_b_r), t0_i = hn::Add(temp_a_i, temp_b_i);
                const auto t1_r = hn::Sub(temp_a_r, temp_b_r), t1_i = hn::Sub(temp_a_i, temp_b_i);

                hn::LoadInterleaved2(d2, reinterpret_cast<const F*>(in_ptr + 8 + idx), temp_a_r, temp_a_i);
                hn::LoadInterleaved2(d2, reinterpret_cast<const F*>(in_ptr + 24 + idx), temp_b_r, temp_b_i);
                const auto t2_r = hn::Add(temp_a_r, temp_b_r), t2_i = hn::Add(temp_a_i, temp_b_i);
                const auto t3_r = hn::Sub(temp_a_r, temp_b_r), t3_i = hn::Sub(temp_a_i, temp_b_i);

                const auto y00_r = hn::Add(t0_r, t2_r), y00_i = hn::Add(t0_i, t2_i);
                const auto y01_r = hn::Add(t1_r, t3_i), y01_i = hn::Sub(t1_i, t3_r);
                const auto y02_r = hn::Sub(t0_r, t2_r), y02_i = hn::Sub(t0_i, t2_i);
                const auto y03_r = hn::Sub(t1_r, t3_i), y03_i = hn::Add(t1_i, t3_r);

                hn::LoadInterleaved2(d2, reinterpret_cast<const F*>(in_ptr + 4 + idx), temp_a_r, temp_a_i);
                hn::LoadInterleaved2(d2, reinterpret_cast<const F*>(in_ptr + 20 + idx), temp_b_r, temp_b_i);
                const auto u0_r = hn::Add(temp_a_r, temp_b_r), u0_i = hn::Add(temp_a_i, temp_b_i);
                const auto u1_r = hn::Sub(temp_a_r, temp_b_r), u1_i = hn::Sub(temp_a_i, temp_b_i);

                hn::LoadInterleaved2(d2, reinterpret_cast<const F*>(in_ptr + 12 + idx), temp_a_r, temp_a_i);
                hn::LoadInterleaved2(d2, reinterpret_cast<const F*>(in_ptr + 28 + idx), temp_b_r, temp_b_i);
                const auto u2_r = hn::Add(temp_a_r, temp_b_r), u2_i = hn::Add(temp_a_i, temp_b_i);
                const auto u3_r = hn::Sub(temp_a_r, temp_b_r), u3_i = hn::Sub(temp_a_i, temp_b_i);

                const auto y10_r = hn::Add(u0_r, u2_r), y10_i = hn::Add(u0_i, u2_i);
                const auto y11_r = hn::Add(u1_r, u3_i), y11_i = hn::Sub(u1_i, u3_r);
                const auto y12_r = hn::Sub(u0_r, u2_r), y12_i = hn::Sub(u0_i, u2_i);
                const auto y13_r = hn::Sub(u1_r, u3_i), y13_i = hn::Add(u1_i, u3_r);

                const auto v0_r = y10_r, v0_i = y10_i;
                const auto v1_r = hn::Mul(hn::Add(y11_r, y11_i), inv_sqrt2), v1_i = hn::Mul(
                    hn::Sub(y11_i, y11_r), inv_sqrt2);
                const auto v2_r = y12_i, v2_i = hn::Neg(y12_r);
                const auto v3_r = hn::Mul(hn::Sub(y13_i, y13_r), inv_sqrt2), v3_i = hn::Mul(
                    hn::Neg(hn::Add(y13_r, y13_i)), inv_sqrt2);

                const auto z00_r = hn::Add(y00_r, v0_r), z00_i = hn::Add(y00_i, v0_i);
                const auto z01_r = hn::Add(y01_r, v1_r), z01_i = hn::Add(y01_i, v1_i);
                const auto z02_r = hn::Add(y02_r, v2_r), z02_i = hn::Add(y02_i, v2_i);
                const auto z03_r = hn::Add(y03_r, v3_r), z03_i = hn::Add(y03_i, v3_i);

                const auto z10_r = hn::Sub(y00_r, v0_r), z10_i = hn::Sub(y00_i, v0_i);
                const auto z11_r = hn::Sub(y01_r, v1_r), z11_i = hn::Sub(y01_i, v1_i);
                const auto z12_r = hn::Sub(y02_r, v2_r), z12_i = hn::Sub(y02_i, v2_i);
                const auto z13_r = hn::Sub(y03_r, v3_r), z13_i = hn::Sub(y03_i, v3_i);

                alignas(16) F ar0[2], ar1[2], ar2[2], ar3[2], ar4[2], ar5[2], ar6[2], ar7[2];
                alignas(16) F ai0[2], ai1[2], ai2[2], ai3[2], ai4[2], ai5[2], ai6[2], ai7[2];

                hn::Store(z00_r, d2, ar0);
                hn::Store(z01_r, d2, ar1);
                hn::Store(z02_r, d2, ar2);
                hn::Store(z03_r, d2, ar3);
                hn::Store(z10_r, d2, ar4);
                hn::Store(z11_r, d2, ar5);
                hn::Store(z12_r, d2, ar6);
                hn::Store(z13_r, d2, ar7);
                hn::Store(z00_i, d2, ai0);
                hn::Store(z01_i, d2, ai1);
                hn::Store(z02_i, d2, ai2);
                hn::Store(z03_i, d2, ai3);
                hn::Store(z10_i, d2, ai4);
                hn::Store(z11_i, d2, ai5);
                hn::Store(z12_i, d2, ai6);
                hn::Store(z13_i, d2, ai7);

                for (size_t lane = 0; lane < 2; ++lane) {
                    size_t out_idx = (idx + lane) * 8;
                    tmp_r[out_idx + 0] = ar0[lane];
                    tmp_r[out_idx + 1] = ar1[lane];
                    tmp_r[out_idx + 2] = ar2[lane];
                    tmp_r[out_idx + 3] = ar3[lane];
                    tmp_r[out_idx + 4] = ar4[lane];
                    tmp_r[out_idx + 5] = ar5[lane];
                    tmp_r[out_idx + 6] = ar6[lane];
                    tmp_r[out_idx + 7] = ar7[lane];

                    tmp_i[out_idx + 0] = ai0[lane];
                    tmp_i[out_idx + 1] = ai1[lane];
                    tmp_i[out_idx + 2] = ai2[lane];
                    tmp_i[out_idx + 3] = ai3[lane];
                    tmp_i[out_idx + 4] = ai4[lane];
                    tmp_i[out_idx + 5] = ai5[lane];
                    tmp_i[out_idx + 6] = ai6[lane];
                    tmp_i[out_idx + 7] = ai7[lane];
                }
            }

            for (size_t k = 0; k < 8; k += 2) {
                const auto i1 = hn::Load(d2, tmp_i + 8 + k), r1 = hn::Load(d2, tmp_r + 8 + k);
                const auto w1_r = hn::Load(d2, w_r_base + k), w1_i = hn::Load(d2, w_i_base + k);
                const auto t1_r = hn::NegMulAdd(i1, w1_i, hn::Mul(r1, w1_r));
                const auto t1_i = hn::MulAdd(i1, w1_r, hn::Mul(r1, w1_i));

                const auto i3 = hn::Load(d2, tmp_i + 24 + k), r3 = hn::Load(d2, tmp_r + 24 + k);
                const auto w3_r = hn::Load(d2, w_r_base + 16 + k), w3_i = hn::Load(d2, w_i_base + 16 + k);
                const auto t3_r = hn::NegMulAdd(i3, w3_i, hn::Mul(r3, w3_r));
                const auto t3_i = hn::MulAdd(i3, w3_r, hn::Mul(r3, w3_i));

                const auto s2_r = hn::Add(t1_r, t3_r), s2_i = hn::Add(t1_i, t3_i);
                const auto s3_r = hn::Sub(t1_r, t3_r), s3_i = hn::Sub(t1_i, t3_i);

                const auto i2 = hn::Load(d2, tmp_i + 16 + k), r2 = hn::Load(d2, tmp_r + 16 + k);
                const auto w2_r = hn::Load(d2, w_r_base + 8 + k), w2_i = hn::Load(d2, w_i_base + 8 + k);
                const auto t2_r = hn::NegMulAdd(i2, w2_i, hn::Mul(r2, w2_r));
                const auto t2_i = hn::MulAdd(i2, w2_r, hn::Mul(r2, w2_i));

                const auto r0 = hn::Load(d2, tmp_r + k), i0 = hn::Load(d2, tmp_i + k);
                const auto s0_r = hn::Add(r0, t2_r), s0_i = hn::Add(i0, t2_i);
                const auto s1_r = hn::Sub(r0, t2_r), s1_i = hn::Sub(i0, t2_i);

                hn::StoreInterleaved2(hn::Add(s0_r, s2_r), hn::Add(s0_i, s2_i), d2, reinterpret_cast<F*>(out_ptr + k));
                hn::StoreInterleaved2(hn::Add(s1_r, s3_i), hn::Sub(s1_i, s3_r), d2,
                                      reinterpret_cast<F*>(out_ptr + 8 + k));
                hn::StoreInterleaved2(hn::Sub(s0_r, s2_r), hn::Sub(s0_i, s2_i), d2,
                                      reinterpret_cast<F*>(out_ptr + 16 + k));
                hn::StoreInterleaved2(hn::Sub(s1_r, s3_i), hn::Add(s1_i, s3_r), d2,
                                      reinterpret_cast<F*>(out_ptr + 24 + k));
            }
        }
    }
}
