#pragma once

#include "zlfft_common.hpp"
#include <vector>
#include <span>
#include <complex>
#include <cmath>
#include <algorithm>
#include <numbers>
#include <cassert>

#include <hwy/highway.h>
#include <hwy/cache_control.h>
#include <hwy/aligned_allocator.h>

namespace zlfft::common {
    namespace hn = hwy::HWY_NAMESPACE;

    template <class D, class V>
    inline void transpose4x4(D d, V v0, V v1, V v2, V v3, V& r0, V& r1, V& r2, V& r3) {
        hn::Repartition<uint64_t, D> d64;
        const auto t0 = hn::InterleaveLower(d, v0, v1);
        const auto t1 = hn::InterleaveLower(d, v2, v3);
        const auto t2 = hn::InterleaveUpper(d, v0, v1);
        const auto t3 = hn::InterleaveUpper(d, v2, v3);

        const auto m0 = hn::BitCast(d, hn::InterleaveLower(d64, hn::BitCast(d64, t0), hn::BitCast(d64, t1)));
        const auto m1 = hn::BitCast(d, hn::InterleaveUpper(d64, hn::BitCast(d64, t0), hn::BitCast(d64, t1)));
        const auto m2 = hn::BitCast(d, hn::InterleaveLower(d64, hn::BitCast(d64, t2), hn::BitCast(d64, t3)));
        const auto m3 = hn::BitCast(d, hn::InterleaveUpper(d64, hn::BitCast(d64, t2), hn::BitCast(d64, t3)));

        if constexpr (D().MaxBytes() > 16) {
            r0 = hn::ConcatLowerLower(d, m1, m0);
            r1 = hn::ConcatLowerLower(d, m3, m2);
            r2 = hn::ConcatUpperUpper(d, m1, m0);
            r3 = hn::ConcatUpperUpper(d, m3, m2);
        } else {
            r0 = m0;
            r1 = m1;
            r2 = m2;
            r3 = m3;
        }
    }

    template <typename F, bool use_fma = false>
    inline void radix4_aosoa(const F* __restrict in_aosoa, F* __restrict out_aosoa,
                             const size_t n, const size_t width,
                             const F* __restrict w_ptr) {
        const auto quarter_n = n >> 2;
        const auto half_n = n >> 1;
        const auto three_quarter_n = quarter_n + half_n;
        const auto three_over_two_n = three_quarter_n << 1;

        const auto double_width = width << 1;
        const auto triple_width = width * 3;
        const auto quad_width = width << 2;
        const auto sextuple_width = triple_width << 1;

        static constexpr hn::ScalableTag<F> d;
        static constexpr size_t lanes = hn::Lanes(d);

        const size_t mask = width - 1;

        for (size_t i = 0; i < quarter_n; i += lanes) {
            const F* __restrict in_shift = in_aosoa + (i << 1);
            const size_t k = i & mask;
            const size_t w_offset = k * 6;

            hn::Vec<decltype(d)> s2_r, s2_i, s3_r, s3_i;

            {
                const auto w1_r = hn::Load(d, w_ptr + w_offset);
                const auto w1_i = hn::Load(d, w_ptr + w_offset + lanes);
                const auto r1 = hn::Load(d, in_shift + half_n);
                const auto i1 = hn::Load(d, in_shift + half_n + lanes);
                const auto t1_r = hn::NegMulAdd(i1, w1_i, hn::Mul(r1, w1_r));
                const auto t1_i = hn::MulAdd(i1, w1_r, hn::Mul(r1, w1_i));

                const auto w3_r = hn::Load(d, w_ptr + w_offset + lanes * 4);
                const auto w3_i = hn::Load(d, w_ptr + w_offset + lanes * 5);
                const auto r3 = hn::Load(d, in_shift + three_over_two_n);
                const auto i3 = hn::Load(d, in_shift + three_over_two_n + lanes);
                const auto t3_r = hn::NegMulAdd(i3, w3_i, hn::Mul(r3, w3_r));
                const auto t3_i = hn::MulAdd(i3, w3_r, hn::Mul(r3, w3_i));

                s2_r = hn::Add(t1_r, t3_r);
                s2_i = hn::Add(t1_i, t3_i);
                s3_r = hn::Sub(t1_r, t3_r);
                s3_i = hn::Sub(t1_i, t3_i);
            }

            hn::Vec<decltype(d)> s0_r, s0_i, s1_r, s1_i;

            if constexpr (use_fma) {
                const auto r0 = hn::Load(d, in_shift);
                const auto i0 = hn::Load(d, in_shift + lanes);

                const auto w2_r = hn::Load(d, w_ptr + w_offset + lanes * 2);
                const auto w2_i = hn::Load(d, w_ptr + w_offset + lanes * 3);
                const auto r2 = hn::Load(d, in_shift + n);
                const auto i2 = hn::Load(d, in_shift + n + lanes);

                const auto m2_r = hn::MulAdd(r2, w2_r, r0);
                s0_r = hn::NegMulAdd(i2, w2_i, m2_r);

                const auto m2_i = hn::MulAdd(i2, w2_r, i0);
                s0_i = hn::MulAdd(r2, w2_i, m2_i);

                const auto two_r0 = hn::Add(r0, r0);
                s1_r = hn::Sub(two_r0, s0_r);

                const auto two_i0 = hn::Add(i0, i0);
                s1_i = hn::Sub(two_i0, s0_i);
            } else {
                const auto w2_r = hn::Load(d, w_ptr + w_offset + lanes * 2);
                const auto w2_i = hn::Load(d, w_ptr + w_offset + lanes * 3);
                const auto r2 = hn::Load(d, in_shift + n);
                const auto i2 = hn::Load(d, in_shift + n + lanes);
                const auto t2_r = hn::NegMulAdd(i2, w2_i, hn::Mul(r2, w2_r));
                const auto t2_i = hn::MulAdd(i2, w2_r, hn::Mul(r2, w2_i));

                const auto r0 = hn::Load(d, in_shift);
                const auto i0 = hn::Load(d, in_shift + lanes);

                s0_r = hn::Add(r0, t2_r);
                s0_i = hn::Add(i0, t2_i);
                s1_r = hn::Sub(r0, t2_r);
                s1_i = hn::Sub(i0, t2_i);
            }

            const size_t j_times_4 = (i & ~mask) << 2;
            const size_t out_idx = j_times_4 + k;
            F* __restrict out_shift = out_aosoa + (out_idx << 1);

            hn::Store(hn::Add(s0_r, s2_r), d, out_shift);
            hn::Store(hn::Add(s0_i, s2_i), d, out_shift + lanes);
            hn::Store(hn::Add(s1_r, s3_i), d, out_shift + double_width);
            hn::Store(hn::Sub(s1_i, s3_r), d, out_shift + double_width + lanes);
            hn::Store(hn::Sub(s0_r, s2_r), d, out_shift + quad_width);
            hn::Store(hn::Sub(s0_i, s2_i), d, out_shift + quad_width + lanes);
            hn::Store(hn::Sub(s1_r, s3_i), d, out_shift + sextuple_width);
            hn::Store(hn::Add(s1_i, s3_r), d, out_shift + sextuple_width + lanes);
        }
    }

    template <typename F>
    inline void radix4_first_pass_fused_aosoa(const std::complex<F>* __restrict in,
                                              F* __restrict out_aosoa, const size_t n) {
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

            const auto out1_r = hn::Add(t1_r, t3_i);
            const auto out1_i = hn::Sub(t1_i, t3_r);
            const auto out3_r = hn::Sub(t1_r, t3_i);
            const auto out3_i = hn::Add(t1_i, t3_r);

            hn::Vec<decltype(d)> r0, r1, r2, r3, i0, i1, i2, i3;
            transpose4x4(d, out0_r, out1_r, out2_r, out3_r, r0, r1, r2, r3);
            transpose4x4(d, out0_i, out1_i, out2_i, out3_i, i0, i1, i2, i3);

            F* __restrict out_shift = out_aosoa + (j << 3);
            hn::Store(r0, d, out_shift);
            hn::Store(i0, d, out_shift + lanes);
            hn::Store(r1, d, out_shift + 2 * lanes);
            hn::Store(i1, d, out_shift + 3 * lanes);
            hn::Store(r2, d, out_shift + 4 * lanes);
            hn::Store(i2, d, out_shift + 5 * lanes);
            hn::Store(r3, d, out_shift + 6 * lanes);
            hn::Store(i3, d, out_shift + 7 * lanes);
        }
    }

    template <typename F>
    inline void radix4_width4_aosoa(const F* __restrict in_aosoa, F* __restrict out_aosoa,
                                    const size_t n,
                                    const F* __restrict w_ptr) {
        const size_t quarter_n = n >> 2;
        const size_t half_n = n >> 1;
        const size_t three_quarter_n = quarter_n * 3;
        const auto three_over_two_n = three_quarter_n << 1;

        static constexpr hn::ScalableTag<F> d;
        static constexpr size_t lanes = hn::Lanes(d);

        const auto w1_r = hn::Load(d, w_ptr);
        const auto w1_i = hn::Load(d, w_ptr + lanes);
        const auto w2_r = hn::Load(d, w_ptr + lanes * 2);
        const auto w2_i = hn::Load(d, w_ptr + lanes * 3);
        const auto w3_r = hn::Load(d, w_ptr + lanes * 4);
        const auto w3_i = hn::Load(d, w_ptr + lanes * 5);

        for (size_t i = 0; i < quarter_n; i += lanes) {
            const F* __restrict in_shift = in_aosoa + (i << 1);
            const auto r1 = hn::Load(d, in_shift + half_n);
            const auto i1 = hn::Load(d, in_shift + half_n + lanes);
            const auto t1_r = hn::NegMulAdd(i1, w1_i, hn::Mul(r1, w1_r));
            const auto t1_i = hn::MulAdd(i1, w1_r, hn::Mul(r1, w1_i));

            const auto r3 = hn::Load(d, in_shift + three_over_two_n);
            const auto i3 = hn::Load(d, in_shift + three_over_two_n + lanes);
            const auto t3_r = hn::NegMulAdd(i3, w3_i, hn::Mul(r3, w3_r));
            const auto t3_i = hn::MulAdd(i3, w3_r, hn::Mul(r3, w3_i));

            const auto s2_r = hn::Add(t1_r, t3_r);
            const auto s2_i = hn::Add(t1_i, t3_i);
            const auto s3_r = hn::Sub(t1_r, t3_r);
            const auto s3_i = hn::Sub(t1_i, t3_i);

            const auto r2 = hn::Load(d, in_shift + n);
            const auto i2 = hn::Load(d, in_shift + n + lanes);
            const auto t2_r = hn::NegMulAdd(i2, w2_i, hn::Mul(r2, w2_r));
            const auto t2_i = hn::MulAdd(i2, w2_r, hn::Mul(r2, w2_i));

            const auto r0 = hn::Load(d, in_shift);
            const auto i0 = hn::Load(d, in_shift + lanes);

            const auto s0_r = hn::Add(r0, t2_r);
            const auto s0_i = hn::Add(i0, t2_i);
            const auto s1_r = hn::Sub(r0, t2_r);
            const auto s1_i = hn::Sub(i0, t2_i);

            const auto out0_r = hn::Add(s0_r, s2_r);
            const auto out0_i = hn::Add(s0_i, s2_i);
            const auto out1_r = hn::Add(s1_r, s3_i);
            const auto out1_i = hn::Sub(s1_i, s3_r);
            const auto out2_r = hn::Sub(s0_r, s2_r);
            const auto out2_i = hn::Sub(s0_i, s2_i);
            const auto out3_r = hn::Sub(s1_r, s3_i);
            const auto out3_i = hn::Add(s1_i, s3_r);

            F* __restrict out_shift = out_aosoa + (i << 3);
            if constexpr (d.MaxBytes() > 16) {
                const auto out01_r_lo = hn::ConcatLowerLower(d, out1_r, out0_r);
                const auto out01_i_lo = hn::ConcatLowerLower(d, out1_i, out0_i);
                const auto out23_r_lo = hn::ConcatLowerLower(d, out3_r, out2_r);
                const auto out23_i_lo = hn::ConcatLowerLower(d, out3_i, out2_i);

                const auto out01_r_hi = hn::ConcatUpperUpper(d, out1_r, out0_r);
                const auto out01_i_hi = hn::ConcatUpperUpper(d, out1_i, out0_i);
                const auto out23_r_hi = hn::ConcatUpperUpper(d, out3_r, out2_r);
                const auto out23_i_hi = hn::ConcatUpperUpper(d, out3_i, out2_i);

                hn::Store(out01_r_lo, d, out_shift);
                hn::Store(out01_i_lo, d, out_shift + lanes);
                hn::Store(out23_r_lo, d, out_shift + lanes * 2);
                hn::Store(out23_i_lo, d, out_shift + lanes * 3);
                hn::Store(out01_r_hi, d, out_shift + lanes * 4);
                hn::Store(out01_i_hi, d, out_shift + lanes * 5);
                hn::Store(out23_r_hi, d, out_shift + lanes * 6);
                hn::Store(out23_i_hi, d, out_shift + lanes * 7);
            } else {
                hn::Store(out0_r, d, out_shift);
                hn::Store(out0_i, d, out_shift + lanes);
                hn::Store(out1_r, d, out_shift + lanes * 2);
                hn::Store(out1_i, d, out_shift + lanes * 3);
                hn::Store(out2_r, d, out_shift + lanes * 4);
                hn::Store(out2_i, d, out_shift + lanes * 5);
                hn::Store(out3_r, d, out_shift + lanes * 6);
                hn::Store(out3_i, d, out_shift + lanes * 7);
            }
        }
    }

    template <typename F, bool use_fma = false>
inline void radix4_width8_aosoa(const F* __restrict in_aosoa, F* __restrict out_aosoa,
                                const size_t n,
                                const F* __restrict w_ptr) {
    static constexpr hn::ScalableTag<F> d;
    static constexpr size_t lanes = hn::Lanes(d);

    if constexpr (lanes > 8) {
        common::radix4_aosoa<F, use_fma>(in_aosoa, out_aosoa, n, 8, w_ptr);
        return;
    }

    const size_t quarter_n = n >> 2;
    const size_t half_n = n >> 1;
    const size_t three_over_two_n = half_n * 3;

    static constexpr size_t step = (lanes > 8) ? lanes : 8;
    static constexpr size_t vecs_per_step = step / lanes;

    hn::Vec<decltype(d)> w1_r_hoist, w1_i_hoist, w2_r_hoist, w2_i_hoist, w3_r_hoist, w3_i_hoist;

    if constexpr (vecs_per_step == 1) {
        w1_r_hoist = hn::Load(d, w_ptr);
        w1_i_hoist = hn::Load(d, w_ptr + lanes);
        w2_r_hoist = hn::Load(d, w_ptr + lanes * 2);
        w2_i_hoist = hn::Load(d, w_ptr + lanes * 3);
        w3_r_hoist = hn::Load(d, w_ptr + lanes * 4);
        w3_i_hoist = hn::Load(d, w_ptr + lanes * 5);
    }

    for (size_t i = 0; i < quarter_n; i += step) {

        #pragma clang loop unroll(full)
        for (size_t v = 0; v < vecs_per_step; ++v) {
            const size_t vec_i = i + v * lanes;
            const F* __restrict in_shift = in_aosoa + (vec_i << 1);

            hn::Vec<decltype(d)> w1_r, w1_i, w2_r, w2_i, w3_r, w3_i;

            if constexpr (vecs_per_step == 1) {
                w1_r = w1_r_hoist; w1_i = w1_i_hoist;
                w2_r = w2_r_hoist; w2_i = w2_i_hoist;
                w3_r = w3_r_hoist; w3_i = w3_i_hoist;
            } else {
                const size_t k = (v * lanes) & 7;
                const size_t w_offset = k * 6;
                w1_r = hn::Load(d, w_ptr + w_offset);
                w1_i = hn::Load(d, w_ptr + w_offset + lanes);
                w2_r = hn::Load(d, w_ptr + w_offset + lanes * 2);
                w2_i = hn::Load(d, w_ptr + w_offset + lanes * 3);
                w3_r = hn::Load(d, w_ptr + w_offset + lanes * 4);
                w3_i = hn::Load(d, w_ptr + w_offset + lanes * 5);
            }

            const auto r1 = hn::Load(d, in_shift + half_n);
            const auto i1 = hn::Load(d, in_shift + half_n + lanes);
            const auto t1_r = hn::NegMulAdd(i1, w1_i, hn::Mul(r1, w1_r));
            const auto t1_i = hn::MulAdd(i1, w1_r, hn::Mul(r1, w1_i));

            const auto r3 = hn::Load(d, in_shift + three_over_two_n);
            const auto i3 = hn::Load(d, in_shift + three_over_two_n + lanes);
            const auto t3_r = hn::NegMulAdd(i3, w3_i, hn::Mul(r3, w3_r));
            const auto t3_i = hn::MulAdd(i3, w3_r, hn::Mul(r3, w3_i));

            const auto s2_r = hn::Add(t1_r, t3_r);
            const auto s2_i = hn::Add(t1_i, t3_i);
            const auto s3_r = hn::Sub(t1_r, t3_r);
            const auto s3_i = hn::Sub(t1_i, t3_i);

            hn::Vec<decltype(d)> s0_r, s0_i, s1_r, s1_i;

            if constexpr (use_fma) {
                const auto r0 = hn::Load(d, in_shift);
                const auto i0 = hn::Load(d, in_shift + lanes);
                const auto r2 = hn::Load(d, in_shift + n);
                const auto i2 = hn::Load(d, in_shift + n + lanes);

                const auto m2_r = hn::MulAdd(r2, w2_r, r0);
                s0_r = hn::NegMulAdd(i2, w2_i, m2_r);

                const auto m2_i = hn::MulAdd(i2, w2_r, i0);
                s0_i = hn::MulAdd(r2, w2_i, m2_i);

                s1_r = hn::Sub(hn::Add(r0, r0), s0_r);
                s1_i = hn::Sub(hn::Add(i0, i0), s0_i);
            } else {
                const auto r2 = hn::Load(d, in_shift + n);
                const auto i2 = hn::Load(d, in_shift + n + lanes);
                const auto t2_r = hn::NegMulAdd(i2, w2_i, hn::Mul(r2, w2_r));
                const auto t2_i = hn::MulAdd(i2, w2_r, hn::Mul(r2, w2_i));

                const auto r0 = hn::Load(d, in_shift);
                const auto i0 = hn::Load(d, in_shift + lanes);

                s0_r = hn::Add(r0, t2_r);
                s0_i = hn::Add(i0, t2_i);
                s1_r = hn::Sub(r0, t2_r);
                s1_i = hn::Sub(i0, t2_i);
            }

            const size_t k = (v * lanes) & 7;
            F* __restrict out_shift = out_aosoa + (vec_i << 3) - k * 6;

            hn::Store(hn::Add(s0_r, s2_r), d, out_shift);
            hn::Store(hn::Add(s0_i, s2_i), d, out_shift + lanes);

            hn::Store(hn::Add(s1_r, s3_i), d, out_shift + 16);
            hn::Store(hn::Sub(s1_i, s3_r), d, out_shift + 16 + lanes);

            hn::Store(hn::Sub(s0_r, s2_r), d, out_shift + 32);
            hn::Store(hn::Sub(s0_i, s2_i), d, out_shift + 32 + lanes);

            hn::Store(hn::Sub(s1_r, s3_i), d, out_shift + 48);
            hn::Store(hn::Add(s1_i, s3_r), d, out_shift + 48 + lanes);
        }
    }
}

    template <typename F, bool use_stream = false>
    inline void radix4_last_pass_fused_aosoa(const F* __restrict in_aosoa,
                                             std::complex<F>* __restrict out,
                                             const size_t n, const size_t width,
                                             const F* __restrict w_ptr) {
        const size_t quarter_n = n >> 2;
        const size_t half_n = n >> 1;
        const size_t three_quarter_n = quarter_n * 3;

        static constexpr hn::ScalableTag<F> d;
        static constexpr size_t lanes = hn::Lanes(d);
        const size_t mask = width - 1;

        for (size_t i = 0; i < quarter_n; i += lanes) {
            const size_t k = i & mask;
            const size_t w_offset = k * 6;

            const auto w1_r = hn::Load(d, w_ptr + w_offset);
            const auto w1_i = hn::Load(d, w_ptr + w_offset + lanes);
            const auto r1 = hn::Load(d, in_aosoa + 2 * (quarter_n + i));
            const auto i1 = hn::Load(d, in_aosoa + 2 * (quarter_n + i) + lanes);
            const auto t1_r = hn::NegMulAdd(i1, w1_i, hn::Mul(r1, w1_r));
            const auto t1_i = hn::MulAdd(i1, w1_r, hn::Mul(r1, w1_i));

            const auto w3_r = hn::Load(d, w_ptr + w_offset + lanes * 4);
            const auto w3_i = hn::Load(d, w_ptr + w_offset + lanes * 5);
            const auto r3 = hn::Load(d, in_aosoa + 2 * (three_quarter_n + i));
            const auto i3 = hn::Load(d, in_aosoa + 2 * (three_quarter_n + i) + lanes);
            const auto t3_r = hn::NegMulAdd(i3, w3_i, hn::Mul(r3, w3_r));
            const auto t3_i = hn::MulAdd(i3, w3_r, hn::Mul(r3, w3_i));

            const auto s2_r = hn::Add(t1_r, t3_r);
            const auto s2_i = hn::Add(t1_i, t3_i);
            const auto s3_r = hn::Sub(t1_r, t3_r);
            const auto s3_i = hn::Sub(t1_i, t3_i);

            const auto w2_r = hn::Load(d, w_ptr + w_offset + lanes * 2);
            const auto w2_i = hn::Load(d, w_ptr + w_offset + lanes * 3);
            const auto r2 = hn::Load(d, in_aosoa + 2 * (half_n + i));
            const auto i2 = hn::Load(d, in_aosoa + 2 * (half_n + i) + lanes);
            const auto t2_r = hn::NegMulAdd(i2, w2_i, hn::Mul(r2, w2_r));
            const auto t2_i = hn::MulAdd(i2, w2_r, hn::Mul(r2, w2_i));

            const auto r0 = hn::Load(d, in_aosoa + 2 * i);
            const auto i0 = hn::Load(d, in_aosoa + 2 * i + lanes);
            const auto s0_r = hn::Add(r0, t2_r);
            const auto s0_i = hn::Add(i0, t2_i);
            const auto s1_r = hn::Sub(r0, t2_r);
            const auto s1_i = hn::Sub(i0, t2_i);

            const size_t j_times_4 = (i & ~mask) << 2;
            const size_t out_idx = j_times_4 + k;

            if constexpr (use_stream) {
                auto stream_complex = [&](const auto& real, const auto& imag, size_t idx) {
                    F* base_ptr = reinterpret_cast<F*>(out + idx);

                    const auto lower = hn::InterleaveLower(d, real, imag);
                    const auto upper = hn::InterleaveUpper(d, real, imag);

                    if constexpr (d.MaxBytes() > 16) {
                        const auto true_lower = hn::ConcatLowerLower(d, upper, lower);
                        const auto true_upper = hn::ConcatUpperUpper(d, upper, lower);

                        hn::Stream(true_lower, d, base_ptr);
                        hn::Stream(true_upper, d, base_ptr + lanes);
                    } else {
                        hn::Stream(lower, d, base_ptr);
                        hn::Stream(upper, d, base_ptr + lanes);
                    }
                };

                stream_complex(hn::Add(s0_r, s2_r), hn::Add(s0_i, s2_i), out_idx);
                stream_complex(hn::Add(s1_r, s3_i), hn::Sub(s1_i, s3_r), out_idx + width);
                stream_complex(hn::Sub(s0_r, s2_r), hn::Sub(s0_i, s2_i), out_idx + (width << 1));
                stream_complex(hn::Sub(s1_r, s3_i), hn::Add(s1_i, s3_r), out_idx + width * 3);
            } else {
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
    }

    template <typename F>
    inline void radix8_aosoa(const F* __restrict in_aosoa, F* __restrict out_aosoa,
                             const size_t n, const size_t width,
                             const F* __restrict w_ptr) {
        const size_t eighth_n = n >> 3;
        static constexpr hn::ScalableTag<F> d;
        static constexpr size_t lanes = hn::Lanes(d);
        const size_t mask = width - 1;

        static constexpr F kInvSqrt2 = static_cast<F>(1.0 / std::numbers::sqrt2);
        const auto inv_sqrt2 = hn::Set(d, kInvSqrt2);

        const size_t out_stride1 = width * 2;
        const size_t out_stride2 = out_stride1 * 2;
        const size_t out_stride3 = out_stride1 * 3;
        const size_t out_stride4 = out_stride1 * 4;
        const size_t out_stride5 = out_stride1 * 5;
        const size_t out_stride6 = out_stride1 * 6;
        const size_t out_stride7 = out_stride1 * 7;

        for (size_t i = 0; i < eighth_n; i += lanes) {
            const size_t k = i & mask;
            const size_t w_offset = k * 14;
            const size_t j_times_8 = (i & ~mask) << 3;
            const size_t out_idx = j_times_8 + k;

            F* out_ptr = out_aosoa + 2 * out_idx;

            auto load_twiddle_mul = [&](size_t in_offset, size_t m_idx, auto& r_out, auto& i_out) {
                const auto r_in = hn::Load(d, in_aosoa + 2 * (in_offset + i));
                const auto i_in = hn::Load(d, in_aosoa + 2 * (in_offset + i) + lanes);
                const auto w_r = hn::Load(d, w_ptr + w_offset + 2 * m_idx * lanes);
                const auto w_i = hn::Load(d, w_ptr + w_offset + (2 * m_idx + 1) * lanes);
                r_out = hn::NegMulAdd(i_in, w_i, hn::Mul(r_in, w_r));
                i_out = hn::MulAdd(i_in, w_r, hn::Mul(r_in, w_i));
            };

            hn::Vec<decltype(d)> tmp_r, tmp_i;

            auto r0 = hn::Load(d, in_aosoa + 2 * i);
            auto i0 = hn::Load(d, in_aosoa + 2 * i + lanes);

            load_twiddle_mul(eighth_n * 4, 0, tmp_r, tmp_i);
            auto t1_r = hn::Sub(r0, tmp_r), t1_i = hn::Sub(i0, tmp_i);
            auto t0_r = hn::Add(r0, tmp_r), t0_i = hn::Add(i0, tmp_i);

            hn::Vec<decltype(d)> r2, i2;
            load_twiddle_mul(eighth_n * 2, 1, r2, i2);
            load_twiddle_mul(eighth_n * 6, 2, tmp_r, tmp_i);
            auto t3_r = hn::Sub(r2, tmp_r), t3_i = hn::Sub(i2, tmp_i);
            auto t2_r = hn::Add(r2, tmp_r), t2_i = hn::Add(i2, tmp_i);

            auto y00_r = hn::Add(t0_r, t2_r), y00_i = hn::Add(t0_i, t2_i);
            auto y02_r = hn::Sub(t0_r, t2_r), y02_i = hn::Sub(t0_i, t2_i);
            auto y01_r = hn::Add(t1_r, t3_i), y01_i = hn::Sub(t1_i, t3_r);
            auto y03_r = hn::Sub(t1_r, t3_i), y03_i = hn::Add(t1_i, t3_r);

            hn::Vec<decltype(d)> r1, i1;
            load_twiddle_mul(eighth_n * 1, 3, r1, i1);
            load_twiddle_mul(eighth_n * 5, 4, tmp_r, tmp_i);
            auto u1_r = hn::Sub(r1, tmp_r), u1_i = hn::Sub(i1, tmp_i);
            auto u0_r = hn::Add(r1, tmp_r), u0_i = hn::Add(i1, tmp_i);

            hn::Vec<decltype(d)> r3, i3;
            load_twiddle_mul(eighth_n * 3, 5, r3, i3);
            load_twiddle_mul(eighth_n * 7, 6, tmp_r, tmp_i);
            auto u3_r = hn::Sub(r3, tmp_r), u3_i = hn::Sub(i3, tmp_i);
            auto u2_r = hn::Add(r3, tmp_r), u2_i = hn::Add(i3, tmp_i);

            auto y10_r = hn::Add(u0_r, u2_r), y10_i = hn::Add(u0_i, u2_i);
            auto y12_r = hn::Sub(u0_r, u2_r), y12_i = hn::Sub(u0_i, u2_i);
            auto y11_r = hn::Add(u1_r, u3_i), y11_i = hn::Sub(u1_i, u3_r);
            auto y13_r = hn::Sub(u1_r, u3_i), y13_i = hn::Add(u1_i, u3_r);

            {
                hn::Store(hn::Add(y00_r, y10_r), d, out_ptr);
                hn::Store(hn::Add(y00_i, y10_i), d, out_ptr + lanes);
                hn::Store(hn::Sub(y00_r, y10_r), d, out_ptr + out_stride4);
                hn::Store(hn::Sub(y00_i, y10_i), d, out_ptr + out_stride4 + lanes);
            }
            {
                const auto v1_r = hn::Mul(hn::Add(y11_r, y11_i), inv_sqrt2);
                const auto v1_i = hn::Mul(hn::Sub(y11_i, y11_r), inv_sqrt2);
                hn::Store(hn::Add(y01_r, v1_r), d, out_ptr + out_stride1);
                hn::Store(hn::Add(y01_i, v1_i), d, out_ptr + out_stride1 + lanes);
                hn::Store(hn::Sub(y01_r, v1_r), d, out_ptr + out_stride5);
                hn::Store(hn::Sub(y01_i, v1_i), d, out_ptr + out_stride5 + lanes);
            }
            {
                hn::Store(hn::Add(y02_r, y12_i), d, out_ptr + out_stride2);
                hn::Store(hn::Sub(y02_i, y12_r), d, out_ptr + out_stride2 + lanes);
                hn::Store(hn::Sub(y02_r, y12_i), d, out_ptr + out_stride6);
                hn::Store(hn::Add(y02_i, y12_r), d, out_ptr + out_stride6 + lanes);
            }
            {
                const auto v3_r = hn::Mul(hn::Sub(y13_i, y13_r), inv_sqrt2);
                const auto v3_i = hn::Mul(hn::Neg(hn::Add(y13_r, y13_i)), inv_sqrt2);
                hn::Store(hn::Add(y03_r, v3_r), d, out_ptr + out_stride3);
                hn::Store(hn::Add(y03_i, v3_i), d, out_ptr + out_stride3 + lanes);
                hn::Store(hn::Sub(y03_r, v3_r), d, out_ptr + out_stride7);
                hn::Store(hn::Sub(y03_i, v3_i), d, out_ptr + out_stride7 + lanes);
            }
        }
    }

    template <typename F>
    inline void radix8_first_pass_fused_aosoa(const std::complex<F>* __restrict in,
                                              F* __restrict out_aosoa, const size_t n) {
        const size_t one_eight_n = n >> 3;
        const size_t quarter_n = n >> 2;
        const size_t half_n = n >> 1;
        const size_t three_quarter_n = 3 * quarter_n;
        const size_t five_four_n = quarter_n + n;
        const size_t three_two_n = n + half_n;
        const size_t seven_four_n = n + three_quarter_n;
        static constexpr hn::ScalableTag<F> d;
        static constexpr size_t lanes = hn::Lanes(d);

        static constexpr F kInvSqrt2 = static_cast<F>(1.0 / std::numbers::sqrt2);
        const auto inv_sqrt2 = hn::Set(d, kInvSqrt2);

        for (size_t j = 0; j + lanes <= one_eight_n; j += lanes) {
            const F* __restrict in_shift = reinterpret_cast<const F*>(in + j);

            hn::Vec<decltype(d)> a_r, a_i, b_r, b_i;

            hn::LoadInterleaved2(d, in_shift, a_r, a_i);
            hn::LoadInterleaved2(d, in_shift + n, b_r, b_i);
            auto t0_r = hn::Add(a_r, b_r), t0_i = hn::Add(a_i, b_i);
            auto t1_r = hn::Sub(a_r, b_r), t1_i = hn::Sub(a_i, b_i);

            hn::LoadInterleaved2(d, in_shift + half_n, a_r, a_i);
            hn::LoadInterleaved2(d, in_shift + three_two_n, b_r, b_i);
            auto t2_r = hn::Add(a_r, b_r), t2_i = hn::Add(a_i, b_i);
            auto t3_r = hn::Sub(a_r, b_r), t3_i = hn::Sub(a_i, b_i);

            auto y00_r = hn::Add(t0_r, t2_r), y00_i = hn::Add(t0_i, t2_i);
            auto y02_r = hn::Sub(t0_r, t2_r), y02_i = hn::Sub(t0_i, t2_i);
            auto y01_r = hn::Add(t1_r, t3_i), y01_i = hn::Sub(t1_i, t3_r);
            auto y03_r = hn::Sub(t1_r, t3_i), y03_i = hn::Add(t1_i, t3_r);

            hn::LoadInterleaved2(d, in_shift + quarter_n, a_r, a_i);
            hn::LoadInterleaved2(d, in_shift + five_four_n, b_r, b_i);
            auto u0_r = hn::Add(a_r, b_r), u0_i = hn::Add(a_i, b_i);
            auto u1_r = hn::Sub(a_r, b_r), u1_i = hn::Sub(a_i, b_i);

            hn::LoadInterleaved2(d, in_shift + three_quarter_n, a_r, a_i);
            hn::LoadInterleaved2(d, in_shift + seven_four_n, b_r, b_i);
            auto u2_r = hn::Add(a_r, b_r), u2_i = hn::Add(a_i, b_i);
            auto u3_r = hn::Sub(a_r, b_r), u3_i = hn::Sub(a_i, b_i);

            auto y10_r = hn::Add(u0_r, u2_r), y10_i = hn::Add(u0_i, u2_i);
            auto y12_r = hn::Sub(u0_r, u2_r), y12_i = hn::Sub(u0_i, u2_i);
            auto y11_r = hn::Add(u1_r, u3_i), y11_i = hn::Sub(u1_i, u3_r);
            auto y13_r = hn::Sub(u1_r, u3_i), y13_i = hn::Add(u1_i, u3_r);

            auto v1_r = hn::Mul(hn::Add(y11_r, y11_i), inv_sqrt2);
            auto v1_i = hn::Mul(hn::Sub(y11_i, y11_r), inv_sqrt2);
            y11_r = v1_r;
            y11_i = v1_i;

            auto v2_r = y12_i;
            auto v2_i = hn::Neg(y12_r);
            y12_r = v2_r;
            y12_i = v2_i;

            auto v3_r = hn::Mul(hn::Sub(y13_i, y13_r), inv_sqrt2);
            auto v3_i = hn::Mul(hn::Neg(hn::Add(y13_r, y13_i)), inv_sqrt2);
            y13_r = v3_r;
            y13_i = v3_i;

            F* __restrict out_shift = out_aosoa + (j << 4);

            {
                auto z00 = hn::Add(y00_r, y10_r), z10 = hn::Sub(y00_r, y10_r);
                auto il_00 = hn::InterleaveLower(d, z00, z10);
                auto iu_00 = hn::InterleaveUpper(d, z00, z10);

                auto z01 = hn::Add(y01_r, y11_r), z11 = hn::Sub(y01_r, y11_r);
                auto il_01 = hn::InterleaveLower(d, z01, z11);
                auto iu_01 = hn::InterleaveUpper(d, z01, z11);

                auto z02 = hn::Add(y02_r, y12_r), z12 = hn::Sub(y02_r, y12_r);
                auto il_02 = hn::InterleaveLower(d, z02, z12);
                auto iu_02 = hn::InterleaveUpper(d, z02, z12);

                auto z03 = hn::Add(y03_r, y13_r), z13 = hn::Sub(y03_r, y13_r);
                auto il_03 = hn::InterleaveLower(d, z03, z13);
                auto iu_03 = hn::InterleaveUpper(d, z03, z13);

                hn::Vec<decltype(d)> lower_r0, lower_r1, lower_r2, lower_r3;
                transpose4x4(d, il_00, il_01, il_02, il_03, lower_r0, lower_r1, lower_r2, lower_r3);

                hn::Vec<decltype(d)> upper_r0, upper_r1, upper_r2, upper_r3;
                transpose4x4(d, iu_00, iu_01, iu_02, iu_03, upper_r0, upper_r1, upper_r2, upper_r3);

                if constexpr (d.MaxBytes() > 16) {
                    hn::Store(lower_r0, d, out_shift);
                    hn::Store(lower_r1, d, out_shift + 2 * lanes);
                    hn::Store(upper_r0, d, out_shift + 4 * lanes);
                    hn::Store(upper_r1, d, out_shift + 6 * lanes);
                    hn::Store(lower_r2, d, out_shift + 8 * lanes);
                    hn::Store(lower_r3, d, out_shift + 10 * lanes);
                    hn::Store(upper_r2, d, out_shift + 12 * lanes);
                    hn::Store(upper_r3, d, out_shift + 14 * lanes);
                } else {
                    hn::Store(lower_r0, d, out_shift);
                    hn::Store(lower_r1, d, out_shift + 2 * lanes);
                    hn::Store(lower_r2, d, out_shift + 4 * lanes);
                    hn::Store(lower_r3, d, out_shift + 6 * lanes);
                    hn::Store(upper_r0, d, out_shift + 8 * lanes);
                    hn::Store(upper_r1, d, out_shift + 10 * lanes);
                    hn::Store(upper_r2, d, out_shift + 12 * lanes);
                    hn::Store(upper_r3, d, out_shift + 14 * lanes);
                }
            }

            {
                auto z00 = hn::Add(y00_i, y10_i), z10 = hn::Sub(y00_i, y10_i);
                auto il_00 = hn::InterleaveLower(d, z00, z10);
                auto iu_00 = hn::InterleaveUpper(d, z00, z10);

                auto z01 = hn::Add(y01_i, y11_i), z11 = hn::Sub(y01_i, y11_i);
                auto il_01 = hn::InterleaveLower(d, z01, z11);
                auto iu_01 = hn::InterleaveUpper(d, z01, z11);

                auto z02 = hn::Add(y02_i, y12_i), z12 = hn::Sub(y02_i, y12_i);
                auto il_02 = hn::InterleaveLower(d, z02, z12);
                auto iu_02 = hn::InterleaveUpper(d, z02, z12);

                auto z03 = hn::Add(y03_i, y13_i), z13 = hn::Sub(y03_i, y13_i);
                auto il_03 = hn::InterleaveLower(d, z03, z13);
                auto iu_03 = hn::InterleaveUpper(d, z03, z13);

                hn::Vec<decltype(d)> lower_i0, lower_i1, lower_i2, lower_i3;
                transpose4x4(d, il_00, il_01, il_02, il_03, lower_i0, lower_i1, lower_i2, lower_i3);

                hn::Vec<decltype(d)> upper_i0, upper_i1, upper_i2, upper_i3;
                transpose4x4(d, iu_00, iu_01, iu_02, iu_03, upper_i0, upper_i1, upper_i2, upper_i3);

                if constexpr (d.MaxBytes() > 16) {
                    hn::Store(lower_i0, d, out_shift + lanes);
                    hn::Store(lower_i1, d, out_shift + 3 * lanes);
                    hn::Store(upper_i0, d, out_shift + 5 * lanes);
                    hn::Store(upper_i1, d, out_shift + 7 * lanes);
                    hn::Store(lower_i2, d, out_shift + 9 * lanes);
                    hn::Store(lower_i3, d, out_shift + 11 * lanes);
                    hn::Store(upper_i2, d, out_shift + 13 * lanes);
                    hn::Store(upper_i3, d, out_shift + 15 * lanes);
                } else {
                    hn::Store(lower_i0, d, out_shift + lanes);
                    hn::Store(lower_i1, d, out_shift + 3 * lanes);
                    hn::Store(lower_i2, d, out_shift + 5 * lanes);
                    hn::Store(lower_i3, d, out_shift + 7 * lanes);
                    hn::Store(upper_i0, d, out_shift + 9 * lanes);
                    hn::Store(upper_i1, d, out_shift + 11 * lanes);
                    hn::Store(upper_i2, d, out_shift + 13 * lanes);
                    hn::Store(upper_i3, d, out_shift + 15 * lanes);
                }
            }
        }
    }
}
