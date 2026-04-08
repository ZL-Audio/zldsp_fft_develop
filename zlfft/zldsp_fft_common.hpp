#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <numeric>
#include <span>
#include <vector>

#include <hwy/aligned_allocator.h>
#include <hwy/highway.h>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace zldsp::fft::common {
    namespace hn = hwy::HWY_NAMESPACE;

    /**
     * get system page size
     * @return
     */
    inline size_t get_system_page_size() {
#if defined(_WIN32)
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        return static_cast<size_t>(sysInfo.dwPageSize);
#else
        return static_cast<size_t>(sysconf(_SC_PAGESIZE));
#endif
    }

    template <bool is_forward, class D, typename V, typename F>
    HWY_INLINE void load_interleaved(D d, const F* __restrict ptr, V& r, V& i) {
        if constexpr (is_forward) {
            hn::LoadInterleaved2(d, ptr, r, i);
        } else {
            hn::LoadInterleaved2(d, ptr, i, r);
        }
    }

    template <bool is_forward, class D, typename V, typename F>
    HWY_INLINE void store_interleaved(const V r, const V i, D d, F* __restrict ptr) {
        if constexpr (is_forward) {
            hn::StoreInterleaved2(r, i, d, ptr);
        } else {
            hn::StoreInterleaved2(i, r, d, ptr);
        }
    }

    enum class StageType {
        kRadix8FirstPass,
        kRadix4FirstPass,
        kRadix4Width4,
        kRadix4,
        kRadix8,
        kRadix4LastPass,
    };

    /**
     * perform a 4x4 matrix transpose
     * @tparam D
     * @tparam V
     * @param d
     * @param v0
     * @param v1
     * @param v2
     * @param v3
     * @param r0
     * @param r1
     * @param r2
     * @param r3
     */
    template <class D, class V>
    inline void transpose4x4(D d, V v0, V v1, V v2, V v3, V& r0, V& r1, V& r2, V& r3) {
        const size_t lanes = hn::Lanes(d);
        using T = hn::TFromD<D>;

        const auto t0 = hn::InterleaveLower(d, v0, v1);
        const auto t1 = hn::InterleaveLower(d, v2, v3);
        const auto t2 = hn::InterleaveUpper(d, v0, v1);
        const auto t3 = hn::InterleaveUpper(d, v2, v3);

        if constexpr (lanes == 2) {
            r0 = t0;
            r1 = t1;
            r2 = t2;
            r3 = t3;
        } else if constexpr (lanes == 4) {
            if constexpr (sizeof(T) == 8) {
                r0 = hn::ConcatLowerLower(d, t1, t0);
                r1 = hn::ConcatLowerLower(d, t3, t2);
                r2 = hn::ConcatUpperUpper(d, t1, t0);
                r3 = hn::ConcatUpperUpper(d, t3, t2);
            } else {
                hn::Repartition<uint64_t, D> d64;
                r0 = hn::BitCast(d, hn::InterleaveLower(d64, hn::BitCast(d64, t0), hn::BitCast(d64, t1)));
                r1 = hn::BitCast(d, hn::InterleaveUpper(d64, hn::BitCast(d64, t0), hn::BitCast(d64, t1)));
                r2 = hn::BitCast(d, hn::InterleaveLower(d64, hn::BitCast(d64, t2), hn::BitCast(d64, t3)));
                r3 = hn::BitCast(d, hn::InterleaveUpper(d64, hn::BitCast(d64, t2), hn::BitCast(d64, t3)));
            }
        } else if constexpr (lanes == 8) {
            hn::Repartition<uint64_t, D> d64;
            const auto m0 = hn::BitCast(d, hn::InterleaveLower(d64, hn::BitCast(d64, t0), hn::BitCast(d64, t1)));
            const auto m1 = hn::BitCast(d, hn::InterleaveUpper(d64, hn::BitCast(d64, t0), hn::BitCast(d64, t1)));
            const auto m2 = hn::BitCast(d, hn::InterleaveLower(d64, hn::BitCast(d64, t2), hn::BitCast(d64, t3)));
            const auto m3 = hn::BitCast(d, hn::InterleaveUpper(d64, hn::BitCast(d64, t2), hn::BitCast(d64, t3)));
            r0 = hn::ConcatLowerLower(d, m1, m0);
            r1 = hn::ConcatLowerLower(d, m3, m2);
            r2 = hn::ConcatUpperUpper(d, m1, m0);
            r3 = hn::ConcatUpperUpper(d, m3, m2);
        } else {
            r0 = t0;
            r1 = t1;
            r2 = t2;
            r3 = t3;
        }
    }

    /**
     * perform a Stockham DIT radix-4 pass when width >= 8
     * @tparam F
     * @param in_aosoa
     * @param out_aosoa
     * @param n
     * @param width
     * @param w_ptr
     */
    template <typename F>
    inline void radix4_aosoa(const F* __restrict in_aosoa, F* __restrict out_aosoa, const size_t n, const size_t width,
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

#pragma clang loop unroll_count(2)
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

    /**
     * performs a Stockham DIT radix-4 first pass and convert data from AoS to AoSoA
     * @tparam F
     * @tparam is_forward
     * @param in
     * @param out_aosoa
     * @param n
     */
    template <typename F, bool is_forward>
    inline void radix4_first_pass_fused_aosoa(const std::complex<F>* __restrict in, F* __restrict out_aosoa,
                                              const size_t n) {
        const size_t quarter_n = n >> 2;
        const size_t half_n = n >> 1;
        const size_t three_over_two_n = n + half_n;

        static constexpr hn::ScalableTag<F> d;
        static constexpr size_t lanes = hn::Lanes(d);

#pragma clang loop unroll_count(2)
        for (size_t j = 0; j < quarter_n; j += lanes) {
            const F* __restrict in_shift = reinterpret_cast<const F*>(in + j);
            hn::Vec<decltype(d)> x0_r, x0_i, x2_r, x2_i;
            load_interleaved<is_forward>(d, in_shift, x0_r, x0_i);
            load_interleaved<is_forward>(d, in_shift + n, x2_r, x2_i);

            const auto t0_r = hn::Add(x0_r, x2_r);
            const auto t0_i = hn::Add(x0_i, x2_i);
            const auto t1_r = hn::Sub(x0_r, x2_r);
            const auto t1_i = hn::Sub(x0_i, x2_i);

            hn::Vec<decltype(d)> x1_r, x1_i, x3_r, x3_i;
            load_interleaved<is_forward>(d, in_shift + half_n, x1_r, x1_i);
            load_interleaved<is_forward>(d, in_shift + three_over_two_n, x3_r, x3_i);

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

    /**
     * perform a Stockham DIT radix-4 pass when width = 4
     * @tparam F
     * @param in_aosoa
     * @param out_aosoa
     * @param n
     * @param w_ptr
     */
    template <typename F>
    inline void radix4_width4_aosoa(const F* __restrict in_aosoa, F* __restrict out_aosoa, const size_t n,
                                    const F* __restrict w_ptr) {
        static constexpr hn::ScalableTag<F> d;
        static constexpr size_t lanes = hn::Lanes(d);

        if constexpr (lanes > 8) {
            common::radix4_aosoa<F>(in_aosoa, out_aosoa, n, 4, w_ptr);
            return;
        }

        const size_t quarter_n = n >> 2;
        const size_t half_n = n >> 1;
        const size_t three_quarter_n = quarter_n * 3;
        const auto three_over_two_n = three_quarter_n << 1;

        static constexpr size_t step = (lanes > 4) ? lanes : 4;
        static constexpr size_t vecs_per_step = step / lanes;
        static constexpr size_t width4_vec = step;

        hn::Vec<decltype(d)> w1_r_hoist, w1_i_hoist, w2_r_hoist, w2_i_hoist, w3_r_hoist, w3_i_hoist;

        if constexpr (vecs_per_step == 1) {
            w1_r_hoist = hn::Load(d, w_ptr);
            w1_i_hoist = hn::Load(d, w_ptr + width4_vec);
            w2_r_hoist = hn::Load(d, w_ptr + width4_vec * 2);
            w2_i_hoist = hn::Load(d, w_ptr + width4_vec * 3);
            w3_r_hoist = hn::Load(d, w_ptr + width4_vec * 4);
            w3_i_hoist = hn::Load(d, w_ptr + width4_vec * 5);
        }

        for (size_t i = 0; i < quarter_n; i += step) {

#pragma clang loop unroll(full)
            for (size_t v = 0; v < vecs_per_step; ++v) {
                const size_t vec_i = i + v * lanes;
                const F* __restrict in_shift = in_aosoa + (vec_i << 1);
                hn::Vec<decltype(d)> w1_r, w1_i, w2_r, w2_i, w3_r, w3_i;

                if constexpr (vecs_per_step == 1) {
                    w1_r = w1_r_hoist;
                    w1_i = w1_i_hoist;
                    w2_r = w2_r_hoist;
                    w2_i = w2_i_hoist;
                    w3_r = w3_r_hoist;
                    w3_i = w3_i_hoist;
                } else {
                    const size_t k = (v * lanes) & 3;
                    w1_r = hn::Load(d, w_ptr + k);
                    w1_i = hn::Load(d, w_ptr + k + width4_vec);
                    w2_r = hn::Load(d, w_ptr + k + width4_vec * 2);
                    w2_i = hn::Load(d, w_ptr + k + width4_vec * 3);
                    w3_r = hn::Load(d, w_ptr + k + width4_vec * 4);
                    w3_i = hn::Load(d, w_ptr + k + width4_vec * 5);
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

                const auto out0_r = hn::Add(s0_r, s2_r);
                const auto out0_i = hn::Add(s0_i, s2_i);
                const auto out1_r = hn::Add(s1_r, s3_i);
                const auto out1_i = hn::Sub(s1_i, s3_r);
                const auto out2_r = hn::Sub(s0_r, s2_r);
                const auto out2_i = hn::Sub(s0_i, s2_i);
                const auto out3_r = hn::Sub(s1_r, s3_i);
                const auto out3_i = hn::Add(s1_i, s3_r);

                if constexpr (lanes > 4) {
                    F* __restrict out_shift = out_aosoa + (i << 3);

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
                    const size_t k = (v * lanes) & 3;
                    F* __restrict out_shift = out_aosoa + (i << 3) + k * 2;

                    hn::Store(out0_r, d, out_shift);
                    hn::Store(out0_i, d, out_shift + lanes);
                    hn::Store(out1_r, d, out_shift + 8);
                    hn::Store(out1_i, d, out_shift + 8 + lanes);
                    hn::Store(out2_r, d, out_shift + 16);
                    hn::Store(out2_i, d, out_shift + 16 + lanes);
                    hn::Store(out3_r, d, out_shift + 24);
                    hn::Store(out3_i, d, out_shift + 24 + lanes);
                }
            }
        }
    }

    /**
     * perform a Stockham DIT radix-4 pass and convert data from AoSoA to AoS
     * @tparam F
     * @tparam is_forward
     * @param in_aosoa
     * @param out
     * @param n
     * @param width
     * @param w_ptr
     */
    template <typename F, bool is_forward>
    inline void radix4_last_pass_fused_aosoa(const F* __restrict in_aosoa, std::complex<F>* __restrict out,
                                             const size_t n, const size_t width, const F* __restrict w_ptr) {
        const size_t quarter_n = n >> 2;
        const size_t half_n = n >> 1;
        const size_t three_quarter_n = quarter_n * 3;

        static constexpr hn::ScalableTag<F> d;
        static constexpr size_t lanes = hn::Lanes(d);
        const size_t mask = width - 1;

#pragma clang loop unroll_count(2)
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

            store_interleaved<is_forward>(hn::Add(s0_r, s2_r), hn::Add(s0_i, s2_i), d,
                                          reinterpret_cast<F*>(out + out_idx));
            store_interleaved<is_forward>(hn::Add(s1_r, s3_i), hn::Sub(s1_i, s3_r), d,
                                          reinterpret_cast<F*>(out + out_idx + width));
            store_interleaved<is_forward>(hn::Sub(s0_r, s2_r), hn::Sub(s0_i, s2_i), d,
                                          reinterpret_cast<F*>(out + out_idx + (width << 1)));
            store_interleaved<is_forward>(hn::Sub(s1_r, s3_i), hn::Add(s1_i, s3_r), d,
                                          reinterpret_cast<F*>(out + out_idx + width * 3));
        }
    }

    /**
     * perform a Stockham DIT radix-8 pass on AoSoA when width >= 8
     * @tparam F
     * @param in_aosoa
     * @param out_aosoa
     * @param n
     * @param width
     * @param w_ptr
     */
    template <typename F>
    inline void radix8_aosoa(const F* __restrict in_aosoa, F* __restrict out_aosoa, const size_t n, const size_t width,
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

            auto load_twiddle_mul = [&](const size_t in_offset, const size_t m_idx, auto& r_out, auto& i_out) {
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

    /**
     * performs a Stockham DIT radix-8 first pass and convert data from AoS to AoSoA
     * @tparam F
     * @tparam is_forward
     * @param in
     * @param out_aosoa
     * @param n
     */
    template <typename F, bool is_forward>
    inline void radix8_first_pass_fused_aosoa(const std::complex<F>* __restrict in, F* __restrict out_aosoa,
                                              const size_t n) {
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

            load_interleaved<is_forward>(d, in_shift, a_r, a_i);
            load_interleaved<is_forward>(d, in_shift + n, b_r, b_i);
            auto t0_r = hn::Add(a_r, b_r), t0_i = hn::Add(a_i, b_i);
            auto t1_r = hn::Sub(a_r, b_r), t1_i = hn::Sub(a_i, b_i);

            load_interleaved<is_forward>(d, in_shift + half_n, a_r, a_i);
            load_interleaved<is_forward>(d, in_shift + three_two_n, b_r, b_i);
            auto t2_r = hn::Add(a_r, b_r), t2_i = hn::Add(a_i, b_i);
            auto t3_r = hn::Sub(a_r, b_r), t3_i = hn::Sub(a_i, b_i);

            auto y00_r = hn::Add(t0_r, t2_r), y00_i = hn::Add(t0_i, t2_i);
            auto y02_r = hn::Sub(t0_r, t2_r), y02_i = hn::Sub(t0_i, t2_i);
            auto y01_r = hn::Add(t1_r, t3_i), y01_i = hn::Sub(t1_i, t3_r);
            auto y03_r = hn::Sub(t1_r, t3_i), y03_i = hn::Add(t1_i, t3_r);

            load_interleaved<is_forward>(d, in_shift + quarter_n, a_r, a_i);
            load_interleaved<is_forward>(d, in_shift + five_four_n, b_r, b_i);
            auto u0_r = hn::Add(a_r, b_r), u0_i = hn::Add(a_i, b_i);
            auto u1_r = hn::Sub(a_r, b_r), u1_i = hn::Sub(a_i, b_i);

            load_interleaved<is_forward>(d, in_shift + three_quarter_n, a_r, a_i);
            load_interleaved<is_forward>(d, in_shift + seven_four_n, b_r, b_i);
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

                if constexpr (lanes > 4) {
                    hn::Store(lower_r0, d, out_shift);
                    hn::Store(lower_r1, d, out_shift + 2 * lanes);
                    hn::Store(upper_r0, d, out_shift + 4 * lanes);
                    hn::Store(upper_r1, d, out_shift + 6 * lanes);
                    hn::Store(lower_r2, d, out_shift + 8 * lanes);
                    hn::Store(lower_r3, d, out_shift + 10 * lanes);
                    hn::Store(upper_r2, d, out_shift + 12 * lanes);
                    hn::Store(upper_r3, d, out_shift + 14 * lanes);
                } else if constexpr (sizeof(F) == 8 && lanes == 4) {
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

                if constexpr (lanes == 8 || (sizeof(F) == 8 && lanes == 4)) {
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

    /**
     * hardcoded FFT for order = 0
     * @tparam F
     * @param in
     * @param out
     */
    template <typename F>
    inline void callback_order_0(const std::complex<F>* in, std::complex<F>* out) {
        out[0] = in[0];
    }

    /**
     * hardcoded FFT for order = 1
     * @tparam F
     * @param in
     * @param out
     */
    template <typename F>
    inline void callback_order_1(const std::complex<F>* in, std::complex<F>* out) {
        out[0] = in[0] + in[1];
        out[1] = in[0] - in[1];
    }

    /**
     * hardcoded FFT for order = 2
     * @tparam F
     * @tparam is_forward
     * @param in
     * @param out
     */
    template <typename F, bool is_forward>
    inline void callback_order_2(const std::complex<F>* in, std::complex<F>* out) {
        const auto t0 = in[0] + in[2];
        const auto t1 = in[0] - in[2];
        const auto t2 = in[1] + in[3];
        const auto t3 = in[1] - in[3];
        out[0] = t0 + t2;
        out[2] = t0 - t2;
        if constexpr (is_forward) {
            out[1] = t1 + std::complex<F>(t3.imag(), -t3.real());
            out[3] = t1 - std::complex<F>(t3.imag(), -t3.real());
        } else {
            out[1] = t1 + std::complex<F>(-t3.imag(), t3.real());
            out[3] = t1 - std::complex<F>(-t3.imag(), t3.real());
        }
    }

    /**
     * hardcoded FFT for order = 3
     * @tparam F
    * @tparam is_forward
     * @param in
     * @param out
     */
    template <typename F, bool is_forward>
    inline void callback_order_3(const std::complex<F>* in, std::complex<F>* out) {
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

        const auto u0_r = x1_r + x5_r, u0_i = x1_i + x5_i;
        const auto u1_r = x1_r - x5_r, u1_i = x1_i - x5_i;
        const auto u2_r = x3_r + x7_r, u2_i = x3_i + x7_i;
        const auto u3_r = x3_r - x7_r, u3_i = x3_i - x7_i;

        const auto y00_r = t0_r + t2_r, y00_i = t0_i + t2_i;
        const auto y02_r = t0_r - t2_r, y02_i = t0_i - t2_i;
        const auto y10_r = u0_r + u2_r, y10_i = u0_i + u2_i;
        const auto y12_r = u0_r - u2_r, y12_i = u0_i - u2_i;
        const auto v0_r = y10_r, v0_i = y10_i;

        F y01_r, y01_i, y03_r, y03_i;
        F y11_r, y11_i, y13_r, y13_i;
        F v1_r, v1_i, v2_r, v2_i, v3_r, v3_i;

        if constexpr (is_forward) {
            y01_r = t1_r + t3_i;
            y01_i = t1_i - t3_r;
            y03_r = t1_r - t3_i;
            y03_i = t1_i + t3_r;

            y11_r = u1_r + u3_i;
            y11_i = u1_i - u3_r;
            y13_r = u1_r - u3_i;
            y13_i = u1_i + u3_r;

            v1_r = (y11_r + y11_i) * kInvSqrt2;
            v1_i = (y11_i - y11_r) * kInvSqrt2;
            v2_r = y12_i;
            v2_i = -y12_r;
            v3_r = (y13_i - y13_r) * kInvSqrt2;
            v3_i = -(y13_r + y13_i) * kInvSqrt2;
        } else {
            y01_r = t1_r - t3_i;
            y01_i = t1_i + t3_r;
            y03_r = t1_r + t3_i;
            y03_i = t1_i - t3_r;

            y11_r = u1_r - u3_i;
            y11_i = u1_i + u3_r;
            y13_r = u1_r + u3_i;
            y13_i = u1_i - u3_r;

            v1_r = (y11_r - y11_i) * kInvSqrt2;
            v1_i = (y11_i + y11_r) * kInvSqrt2;
            v2_r = -y12_i;
            v2_i = y12_r;
            v3_r = -(y13_r + y13_i) * kInvSqrt2;
            v3_i = (y13_r - y13_i) * kInvSqrt2;
        }

        out[0] = std::complex<F>(y00_r + v0_r, y00_i + v0_i);
        out[1] = std::complex<F>(y01_r + v1_r, y01_i + v1_i);
        out[2] = std::complex<F>(y02_r + v2_r, y02_i + v2_i);
        out[3] = std::complex<F>(y03_r + v3_r, y03_i + v3_i);
        out[4] = std::complex<F>(y00_r - v0_r, y00_i - v0_i);
        out[5] = std::complex<F>(y01_r - v1_r, y01_i - v1_i);
        out[6] = std::complex<F>(y02_r - v2_r, y02_i - v2_i);
        out[7] = std::complex<F>(y03_r - v3_r, y03_i - v3_i);
    }

    /**
     * hardcoded FFT for order = 4
     * @tparam F
     * @tparam is_forward
     * @param in
     * @param out
     * @param w_r_base
     * @param w_i_base
     */
    template <typename F, bool is_forward>
    inline void callback_order_4(const std::complex<F>* in, std::complex<F>* out, const F* w_r_base,
                                 const F* w_i_base) {
        static constexpr hn::ScalableTag<F> d;
        static constexpr size_t lanes = hn::Lanes(d);

        alignas(64) F tmp_r[16];
        alignas(64) F tmp_i[16];

        if constexpr (lanes >= 8) {
            hn::FixedTag<F, 4> d4;

            hn::Vec<decltype(d)> v0_r, v0_i, v1_r, v1_i;
            load_interleaved<is_forward>(d, reinterpret_cast<const F*>(in), v0_r, v0_i);
            load_interleaved<is_forward>(d, reinterpret_cast<const F*>(in + 8), v1_r, v1_i);

            const auto t02_r = hn::Add(v0_r, v1_r), t02_i = hn::Add(v0_i, v1_i);
            const auto t13_r = hn::Sub(v0_r, v1_r), t13_i = hn::Sub(v0_i, v1_i);

            const auto A_r = hn::ConcatLowerLower(d, t13_r, t02_r);
            const auto A_i = hn::ConcatLowerLower(d, t13_i, t02_i);
            const auto B_r = hn::ConcatUpperUpper(d, t13_i, t02_r);
            const auto B_i = hn::ConcatUpperUpper(d, t13_r, t02_i);

            const auto out01_r = hn::Add(A_r, B_r);
            const auto out23_r = hn::Sub(A_r, B_r);
            const auto sum_i = hn::Add(A_i, B_i);
            const auto diff_i = hn::Sub(A_i, B_i);

            hn::StoreInterleaved4(hn::LowerHalf(d4, out01_r), hn::UpperHalf(d4, out01_r), hn::LowerHalf(d4, out23_r),
                                  hn::UpperHalf(d4, out23_r), d4, tmp_r);

            hn::StoreInterleaved4(hn::LowerHalf(d4, sum_i), hn::UpperHalf(d4, diff_i), hn::LowerHalf(d4, diff_i),
                                  hn::UpperHalf(d4, sum_i), d4, tmp_i);

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

            store_interleaved<is_forward>(hn::Combine(d, f1_r, f0_r), hn::Combine(d, f1_i, f0_i), d,
                                          reinterpret_cast<F*>(out));
            store_interleaved<is_forward>(hn::Combine(d, f3_r, f2_r), hn::Combine(d, f3_i, f2_i), d,
                                          reinterpret_cast<F*>(out + 8));

        } else {
#pragma clang loop unroll(full)
            for (size_t i = 0; i < 4; i += lanes) {
                hn::Vec<decltype(d)> x0_r, x0_i, x1_r, x1_i, x2_r, x2_i, x3_r, x3_i;

                load_interleaved<is_forward>(d, reinterpret_cast<const F*>(in + i), x0_r, x0_i);
                load_interleaved<is_forward>(d, reinterpret_cast<const F*>(in + i + 8), x2_r, x2_i);

                const auto t0_r = hn::Add(x0_r, x2_r), t0_i = hn::Add(x0_i, x2_i);
                const auto t1_r = hn::Sub(x0_r, x2_r), t1_i = hn::Sub(x0_i, x2_i);

                load_interleaved<is_forward>(d, reinterpret_cast<const F*>(in + i + 4), x1_r, x1_i);
                load_interleaved<is_forward>(d, reinterpret_cast<const F*>(in + i + 12), x3_r, x3_i);

                const auto t2_r = hn::Add(x1_r, x3_r), t2_i = hn::Add(x1_i, x3_i);
                const auto t3_r = hn::Sub(x1_r, x3_r), t3_i = hn::Sub(x1_i, x3_i);

                const auto out0_r = hn::Add(t0_r, t2_r), out0_i = hn::Add(t0_i, t2_i);
                const auto out2_r = hn::Sub(t0_r, t2_r), out2_i = hn::Sub(t0_i, t2_i);
                const auto out1_r = hn::Add(t1_r, t3_i), out1_i = hn::Sub(t1_i, t3_r);
                const auto out3_r = hn::Sub(t1_r, t3_i), out3_i = hn::Add(t1_i, t3_r);

                hn::StoreInterleaved4(out0_r, out1_r, out2_r, out3_r, d, tmp_r + i * 4);
                hn::StoreInterleaved4(out0_i, out1_i, out2_i, out3_i, d, tmp_i + i * 4);
            }
#pragma clang loop unroll(full)
            for (size_t i = 0; i < 4; i += lanes) {
                const auto i1 = hn::Load(d, tmp_i + 4 + i), r1 = hn::Load(d, tmp_r + 4 + i);
                const auto w1_r = hn::Load(d, w_r_base + i), w1_i = hn::Load(d, w_i_base + i);
                const auto t1_r = hn::NegMulAdd(i1, w1_i, hn::Mul(r1, w1_r));
                const auto t1_i = hn::MulAdd(i1, w1_r, hn::Mul(r1, w1_i));

                const auto i3 = hn::Load(d, tmp_i + 12 + i), r3 = hn::Load(d, tmp_r + 12 + i);
                const auto w3_r = hn::Load(d, w_r_base + 8 + i), w3_i = hn::Load(d, w_i_base + 8 + i);
                const auto t3_r = hn::NegMulAdd(i3, w3_i, hn::Mul(r3, w3_r));
                const auto t3_i = hn::MulAdd(i3, w3_r, hn::Mul(r3, w3_i));

                const auto s2_r = hn::Add(t1_r, t3_r), s2_i = hn::Add(t1_i, t3_i);
                const auto s3_r = hn::Sub(t1_r, t3_r), s3_i = hn::Sub(t1_i, t3_i);

                const auto i2 = hn::Load(d, tmp_i + 8 + i), r2 = hn::Load(d, tmp_r + 8 + i);
                const auto w2_r = hn::Load(d, w_r_base + 4 + i), w2_i = hn::Load(d, w_i_base + 4 + i);
                const auto t2_r = hn::NegMulAdd(i2, w2_i, hn::Mul(r2, w2_r));
                const auto t2_i = hn::MulAdd(i2, w2_r, hn::Mul(r2, w2_i));

                const auto r0 = hn::Load(d, tmp_r + i), i0 = hn::Load(d, tmp_i + i);
                const auto s0_r = hn::Add(r0, t2_r), s0_i = hn::Add(i0, t2_i);
                const auto s1_r = hn::Sub(r0, t2_r), s1_i = hn::Sub(i0, t2_i);

                store_interleaved<is_forward>(hn::Add(s0_r, s2_r), hn::Add(s0_i, s2_i), d,
                                              reinterpret_cast<F*>(out + i));
                store_interleaved<is_forward>(hn::Add(s1_r, s3_i), hn::Sub(s1_i, s3_r), d,
                                              reinterpret_cast<F*>(out + 4 + i));
                store_interleaved<is_forward>(hn::Sub(s0_r, s2_r), hn::Sub(s0_i, s2_i), d,
                                              reinterpret_cast<F*>(out + 8 + i));
                store_interleaved<is_forward>(hn::Sub(s1_r, s3_i), hn::Add(s1_i, s3_r), d,
                                              reinterpret_cast<F*>(out + 12 + i));
            }
        }
    }

    /**
     * hardcoded FFT for order = 5
     * @tparam F
     * @tparam is_forward
     * @param in
     * @param out
     * @param w_r_base
     * @param w_i_base
     */
    template <typename F, bool is_forward>
    inline void callback_order_5(const std::complex<F>* in, std::complex<F>* out, const F* w_r_base,
                                 const F* w_i_base) {
        static constexpr hn::ScalableTag<F> d;
        static constexpr size_t lanes = hn::Lanes(d);
        static constexpr F kInvSqrt2 = static_cast<F>(1.0 / std::numbers::sqrt2);

        alignas(64) F tmp_r[32];
        alignas(64) F tmp_i[32];

        if constexpr (lanes >= 8) {
            const hn::FixedTag<F, 4> d4;
            const auto inv_sqrt2 = hn::Set(d4, kInvSqrt2);

            hn::Vec<decltype(d)> vec_in0_r, vec_in0_i, vec_in16_r, vec_in16_i;
            hn::Vec<decltype(d)> vec_in8_r, vec_in8_i, vec_in24_r, vec_in24_i;

            load_interleaved<is_forward>(d, reinterpret_cast<const F*>(in), vec_in0_r, vec_in0_i);
            load_interleaved<is_forward>(d, reinterpret_cast<const F*>(in + 16), vec_in16_r, vec_in16_i);

            const auto vec_t0_r = hn::Add(vec_in0_r, vec_in16_r), vec_t0_i = hn::Add(vec_in0_i, vec_in16_i);
            const auto vec_t1_r = hn::Sub(vec_in0_r, vec_in16_r), vec_t1_i = hn::Sub(vec_in0_i, vec_in16_i);

            load_interleaved<is_forward>(d, reinterpret_cast<const F*>(in + 8), vec_in8_r, vec_in8_i);
            load_interleaved<is_forward>(d, reinterpret_cast<const F*>(in + 24), vec_in24_r, vec_in24_i);

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
            const auto v1_r = hn::Mul(hn::Add(y11_r, y11_i), inv_sqrt2),
                v1_i = hn::Mul(hn::Sub(y11_i, y11_r), inv_sqrt2);
            const auto v2_r = y12_i, v2_i = hn::Neg(y12_r);
            const auto v3_r = hn::Mul(hn::Sub(y13_i, y13_r), inv_sqrt2),
                v3_i = hn::Mul(hn::Neg(hn::Add(y13_r, y13_i)), inv_sqrt2);

            const auto z00_r = hn::Add(y00_r, v0_r), z00_i = hn::Add(y00_i, v0_i);
            const auto z01_r = hn::Add(y01_r, v1_r), z01_i = hn::Add(y01_i, v1_i);
            const auto z02_r = hn::Add(y02_r, v2_r), z02_i = hn::Add(y02_i, v2_i);
            const auto z03_r = hn::Add(y03_r, v3_r), z03_i = hn::Add(y03_i, v3_i);

            const auto z10_r = hn::Sub(y00_r, v0_r), z10_i = hn::Sub(y00_i, v0_i);
            const auto z11_r = hn::Sub(y01_r, v1_r), z11_i = hn::Sub(y01_i, v1_i);
            const auto z12_r = hn::Sub(y02_r, v2_r), z12_i = hn::Sub(y02_i, v2_i);
            const auto z13_r = hn::Sub(y03_r, v3_r), z13_i = hn::Sub(y03_i, v3_i);

            const auto lower_r0 = hn::InterleaveLower(d4, z00_r, z10_r);
            const auto upper_r0 = hn::InterleaveUpper(d4, z00_r, z10_r);
            const auto lower_r1 = hn::InterleaveLower(d4, z01_r, z11_r);
            const auto upper_r1 = hn::InterleaveUpper(d4, z01_r, z11_r);
            const auto lower_r2 = hn::InterleaveLower(d4, z02_r, z12_r);
            const auto upper_r2 = hn::InterleaveUpper(d4, z02_r, z12_r);
            const auto lower_r3 = hn::InterleaveLower(d4, z03_r, z13_r);
            const auto upper_r3 = hn::InterleaveUpper(d4, z03_r, z13_r);

            const auto lower_i0 = hn::InterleaveLower(d4, z00_i, z10_i);
            const auto upper_i0 = hn::InterleaveUpper(d4, z00_i, z10_i);
            const auto lower_i1 = hn::InterleaveLower(d4, z01_i, z11_i);
            const auto upper_i1 = hn::InterleaveUpper(d4, z01_i, z11_i);
            const auto lower_i2 = hn::InterleaveLower(d4, z02_i, z12_i);
            const auto upper_i2 = hn::InterleaveUpper(d4, z02_i, z12_i);
            const auto lower_i3 = hn::InterleaveLower(d4, z03_i, z13_i);
            const auto upper_i3 = hn::InterleaveUpper(d4, z03_i, z13_i);

            hn::StoreInterleaved4(lower_r0, lower_r1, lower_r2, lower_r3, d4, tmp_r);
            hn::StoreInterleaved4(upper_r0, upper_r1, upper_r2, upper_r3, d4, tmp_r + 16);
            hn::StoreInterleaved4(lower_i0, lower_i1, lower_i2, lower_i3, d4, tmp_i);
            hn::StoreInterleaved4(upper_i0, upper_i1, upper_i2, upper_i3, d4, tmp_i + 16);

            const auto i1 = hn::Load(d, tmp_i + 8), r1 = hn::Load(d, tmp_r + 8);
            const auto w1_r = hn::Load(d, w_r_base), w1_i = hn::Load(d, w_i_base);
            const auto t1_r = hn::NegMulAdd(i1, w1_i, hn::Mul(r1, w1_r));
            const auto t1_i = hn::MulAdd(i1, w1_r, hn::Mul(r1, w1_i));

            const auto i3 = hn::Load(d, tmp_i + 24), r3 = hn::Load(d, tmp_r + 24);
            const auto w3_r = hn::Load(d, w_r_base + 16), w3_i = hn::Load(d, w_i_base + 16);
            const auto t3_r = hn::NegMulAdd(i3, w3_i, hn::Mul(r3, w3_r));
            const auto t3_i = hn::MulAdd(i3, w3_r, hn::Mul(r3, w3_i));

            const auto s2_r = hn::Add(t1_r, t3_r), s2_i = hn::Add(t1_i, t3_i);
            const auto s3_r = hn::Sub(t1_r, t3_r), s3_i = hn::Sub(t1_i, t3_i);

            const auto i2 = hn::Load(d, tmp_i + 16), r2 = hn::Load(d, tmp_r + 16);
            const auto w2_r = hn::Load(d, w_r_base + 8), w2_i = hn::Load(d, w_i_base + 8);
            const auto t2_r = hn::NegMulAdd(i2, w2_i, hn::Mul(r2, w2_r));
            const auto t2_i = hn::MulAdd(i2, w2_r, hn::Mul(r2, w2_i));

            const auto r0 = hn::Load(d, tmp_r), i0 = hn::Load(d, tmp_i);
            const auto s0_r = hn::Add(r0, t2_r), s0_i = hn::Add(i0, t2_i);
            const auto s1_r = hn::Sub(r0, t2_r), s1_i = hn::Sub(i0, t2_i);

            store_interleaved<is_forward>(hn::Add(s0_r, s2_r), hn::Add(s0_i, s2_i), d, reinterpret_cast<F*>(out));
            store_interleaved<is_forward>(hn::Add(s1_r, s3_i), hn::Sub(s1_i, s3_r), d, reinterpret_cast<F*>(out + 8));
            store_interleaved<is_forward>(hn::Sub(s0_r, s2_r), hn::Sub(s0_i, s2_i), d, reinterpret_cast<F*>(out + 16));
            store_interleaved<is_forward>(hn::Sub(s1_r, s3_i), hn::Add(s1_i, s3_r), d, reinterpret_cast<F*>(out + 24));

        } else if constexpr (lanes >= 4) {
            {
                const auto inv_sqrt2 = hn::Set(d, kInvSqrt2);
                hn::Vec<decltype(d)> temp_a_r, temp_a_i, temp_b_r, temp_b_i;

                load_interleaved<is_forward>(d, reinterpret_cast<const F*>(in), temp_a_r, temp_a_i);
                load_interleaved<is_forward>(d, reinterpret_cast<const F*>(in + 16), temp_b_r, temp_b_i);
                const auto t0_r = hn::Add(temp_a_r, temp_b_r), t0_i = hn::Add(temp_a_i, temp_b_i);
                const auto t1_r = hn::Sub(temp_a_r, temp_b_r), t1_i = hn::Sub(temp_a_i, temp_b_i);

                load_interleaved<is_forward>(d, reinterpret_cast<const F*>(in + 8), temp_a_r, temp_a_i);
                load_interleaved<is_forward>(d, reinterpret_cast<const F*>(in + 24), temp_b_r, temp_b_i);
                const auto t2_r = hn::Add(temp_a_r, temp_b_r), t2_i = hn::Add(temp_a_i, temp_b_i);
                const auto t3_r = hn::Sub(temp_a_r, temp_b_r), t3_i = hn::Sub(temp_a_i, temp_b_i);

                const auto y00_r = hn::Add(t0_r, t2_r), y00_i = hn::Add(t0_i, t2_i);
                const auto y01_r = hn::Add(t1_r, t3_i), y01_i = hn::Sub(t1_i, t3_r);
                const auto y02_r = hn::Sub(t0_r, t2_r), y02_i = hn::Sub(t0_i, t2_i);
                const auto y03_r = hn::Sub(t1_r, t3_i), y03_i = hn::Add(t1_i, t3_r);

                load_interleaved<is_forward>(d, reinterpret_cast<const F*>(in + 4), temp_a_r, temp_a_i);
                load_interleaved<is_forward>(d, reinterpret_cast<const F*>(in + 20), temp_b_r, temp_b_i);
                const auto u0_r = hn::Add(temp_a_r, temp_b_r), u0_i = hn::Add(temp_a_i, temp_b_i);
                const auto u1_r = hn::Sub(temp_a_r, temp_b_r), u1_i = hn::Sub(temp_a_i, temp_b_i);

                load_interleaved<is_forward>(d, reinterpret_cast<const F*>(in + 12), temp_a_r, temp_a_i);
                load_interleaved<is_forward>(d, reinterpret_cast<const F*>(in + 28), temp_b_r, temp_b_i);
                const auto u2_r = hn::Add(temp_a_r, temp_b_r), u2_i = hn::Add(temp_a_i, temp_b_i);
                const auto u3_r = hn::Sub(temp_a_r, temp_b_r), u3_i = hn::Sub(temp_a_i, temp_b_i);

                const auto y10_r = hn::Add(u0_r, u2_r), y10_i = hn::Add(u0_i, u2_i);
                const auto y11_r = hn::Add(u1_r, u3_i), y11_i = hn::Sub(u1_i, u3_r);
                const auto y12_r = hn::Sub(u0_r, u2_r), y12_i = hn::Sub(u0_i, u2_i);
                const auto y13_r = hn::Sub(u1_r, u3_i), y13_i = hn::Add(u1_i, u3_r);

                const auto v0_r = y10_r, v0_i = y10_i;
                const auto v1_r = hn::Mul(hn::Add(y11_r, y11_i), inv_sqrt2),
                    v1_i = hn::Mul(hn::Sub(y11_i, y11_r), inv_sqrt2);
                const auto v2_r = y12_i, v2_i = hn::Neg(y12_r);
                const auto v3_r = hn::Mul(hn::Sub(y13_i, y13_r), inv_sqrt2),
                    v3_i = hn::Mul(hn::Neg(hn::Add(y13_r, y13_i)), inv_sqrt2);

                const auto z00_r = hn::Add(y00_r, v0_r), z00_i = hn::Add(y00_i, v0_i);
                const auto z01_r = hn::Add(y01_r, v1_r), z01_i = hn::Add(y01_i, v1_i);
                const auto z02_r = hn::Add(y02_r, v2_r), z02_i = hn::Add(y02_i, v2_i);
                const auto z03_r = hn::Add(y03_r, v3_r), z03_i = hn::Add(y03_i, v3_i);

                const auto z10_r = hn::Sub(y00_r, v0_r), z10_i = hn::Sub(y00_i, v0_i);
                const auto z11_r = hn::Sub(y01_r, v1_r), z11_i = hn::Sub(y01_i, v1_i);
                const auto z12_r = hn::Sub(y02_r, v2_r), z12_i = hn::Sub(y02_i, v2_i);
                const auto z13_r = hn::Sub(y03_r, v3_r), z13_i = hn::Sub(y03_i, v3_i);

                const auto lower_r0 = hn::InterleaveLower(d, z00_r, z10_r);
                const auto upper_r0 = hn::InterleaveUpper(d, z00_r, z10_r);
                const auto lower_r1 = hn::InterleaveLower(d, z01_r, z11_r);
                const auto upper_r1 = hn::InterleaveUpper(d, z01_r, z11_r);
                const auto lower_r2 = hn::InterleaveLower(d, z02_r, z12_r);
                const auto upper_r2 = hn::InterleaveUpper(d, z02_r, z12_r);
                const auto lower_r3 = hn::InterleaveLower(d, z03_r, z13_r);
                const auto upper_r3 = hn::InterleaveUpper(d, z03_r, z13_r);

                const auto lower_i0 = hn::InterleaveLower(d, z00_i, z10_i);
                const auto upper_i0 = hn::InterleaveUpper(d, z00_i, z10_i);
                const auto lower_i1 = hn::InterleaveLower(d, z01_i, z11_i);
                const auto upper_i1 = hn::InterleaveUpper(d, z01_i, z11_i);
                const auto lower_i2 = hn::InterleaveLower(d, z02_i, z12_i);
                const auto upper_i2 = hn::InterleaveUpper(d, z02_i, z12_i);
                const auto lower_i3 = hn::InterleaveLower(d, z03_i, z13_i);
                const auto upper_i3 = hn::InterleaveUpper(d, z03_i, z13_i);

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

                store_interleaved<is_forward>(hn::Add(s0_r, s2_r), hn::Add(s0_i, s2_i), d,
                                              reinterpret_cast<F*>(out + k));
                store_interleaved<is_forward>(hn::Add(s1_r, s3_i), hn::Sub(s1_i, s3_r), d,
                                              reinterpret_cast<F*>(out + 8 + k));
                store_interleaved<is_forward>(hn::Sub(s0_r, s2_r), hn::Sub(s0_i, s2_i), d,
                                              reinterpret_cast<F*>(out + 16 + k));
                store_interleaved<is_forward>(hn::Sub(s1_r, s3_i), hn::Add(s1_i, s3_r), d,
                                              reinterpret_cast<F*>(out + 24 + k));
            }
        } else {
            const auto inv_sqrt2 = hn::Set(d, kInvSqrt2);

            for (size_t idx = 0; idx < 4; idx += 2) {
                hn::Vec<decltype(d)> temp_a_r, temp_a_i, temp_b_r, temp_b_i;

                load_interleaved<is_forward>(d, reinterpret_cast<const F*>(in + idx), temp_a_r, temp_a_i);
                load_interleaved<is_forward>(d, reinterpret_cast<const F*>(in + 16 + idx), temp_b_r, temp_b_i);
                const auto t0_r = hn::Add(temp_a_r, temp_b_r), t0_i = hn::Add(temp_a_i, temp_b_i);
                const auto t1_r = hn::Sub(temp_a_r, temp_b_r), t1_i = hn::Sub(temp_a_i, temp_b_i);

                load_interleaved<is_forward>(d, reinterpret_cast<const F*>(in + 8 + idx), temp_a_r, temp_a_i);
                load_interleaved<is_forward>(d, reinterpret_cast<const F*>(in + 24 + idx), temp_b_r, temp_b_i);
                const auto t2_r = hn::Add(temp_a_r, temp_b_r), t2_i = hn::Add(temp_a_i, temp_b_i);
                const auto t3_r = hn::Sub(temp_a_r, temp_b_r), t3_i = hn::Sub(temp_a_i, temp_b_i);

                const auto y00_r = hn::Add(t0_r, t2_r), y00_i = hn::Add(t0_i, t2_i);
                const auto y01_r = hn::Add(t1_r, t3_i), y01_i = hn::Sub(t1_i, t3_r);
                const auto y02_r = hn::Sub(t0_r, t2_r), y02_i = hn::Sub(t0_i, t2_i);
                const auto y03_r = hn::Sub(t1_r, t3_i), y03_i = hn::Add(t1_i, t3_r);

                load_interleaved<is_forward>(d, reinterpret_cast<const F*>(in + 4 + idx), temp_a_r, temp_a_i);
                load_interleaved<is_forward>(d, reinterpret_cast<const F*>(in + 20 + idx), temp_b_r, temp_b_i);
                const auto u0_r = hn::Add(temp_a_r, temp_b_r), u0_i = hn::Add(temp_a_i, temp_b_i);
                const auto u1_r = hn::Sub(temp_a_r, temp_b_r), u1_i = hn::Sub(temp_a_i, temp_b_i);

                load_interleaved<is_forward>(d, reinterpret_cast<const F*>(in + 12 + idx), temp_a_r, temp_a_i);
                load_interleaved<is_forward>(d, reinterpret_cast<const F*>(in + 28 + idx), temp_b_r, temp_b_i);
                const auto u2_r = hn::Add(temp_a_r, temp_b_r), u2_i = hn::Add(temp_a_i, temp_b_i);
                const auto u3_r = hn::Sub(temp_a_r, temp_b_r), u3_i = hn::Sub(temp_a_i, temp_b_i);

                const auto y10_r = hn::Add(u0_r, u2_r), y10_i = hn::Add(u0_i, u2_i);
                const auto y11_r = hn::Add(u1_r, u3_i), y11_i = hn::Sub(u1_i, u3_r);
                const auto y12_r = hn::Sub(u0_r, u2_r), y12_i = hn::Sub(u0_i, u2_i);
                const auto y13_r = hn::Sub(u1_r, u3_i), y13_i = hn::Add(u1_i, u3_r);

                const auto v0_r = y10_r, v0_i = y10_i;
                const auto v1_r = hn::Mul(hn::Add(y11_r, y11_i), inv_sqrt2),
                    v1_i = hn::Mul(hn::Sub(y11_i, y11_r), inv_sqrt2);
                const auto v2_r = y12_i, v2_i = hn::Neg(y12_r);
                const auto v3_r = hn::Mul(hn::Sub(y13_i, y13_r), inv_sqrt2),
                    v3_i = hn::Mul(hn::Neg(hn::Add(y13_r, y13_i)), inv_sqrt2);

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

                hn::Store(z00_r, d, ar0);
                hn::Store(z01_r, d, ar1);
                hn::Store(z02_r, d, ar2);
                hn::Store(z03_r, d, ar3);
                hn::Store(z10_r, d, ar4);
                hn::Store(z11_r, d, ar5);
                hn::Store(z12_r, d, ar6);
                hn::Store(z13_r, d, ar7);
                hn::Store(z00_i, d, ai0);
                hn::Store(z01_i, d, ai1);
                hn::Store(z02_i, d, ai2);
                hn::Store(z03_i, d, ai3);
                hn::Store(z10_i, d, ai4);
                hn::Store(z11_i, d, ai5);
                hn::Store(z12_i, d, ai6);
                hn::Store(z13_i, d, ai7);

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
                const auto i1 = hn::Load(d, tmp_i + 8 + k), r1 = hn::Load(d, tmp_r + 8 + k);
                const auto w1_r = hn::Load(d, w_r_base + k), w1_i = hn::Load(d, w_i_base + k);
                const auto t1_r = hn::NegMulAdd(i1, w1_i, hn::Mul(r1, w1_r));
                const auto t1_i = hn::MulAdd(i1, w1_r, hn::Mul(r1, w1_i));

                const auto i3 = hn::Load(d, tmp_i + 24 + k), r3 = hn::Load(d, tmp_r + 24 + k);
                const auto w3_r = hn::Load(d, w_r_base + 16 + k), w3_i = hn::Load(d, w_i_base + 16 + k);
                const auto t3_r = hn::NegMulAdd(i3, w3_i, hn::Mul(r3, w3_r));
                const auto t3_i = hn::MulAdd(i3, w3_r, hn::Mul(r3, w3_i));

                const auto s2_r = hn::Add(t1_r, t3_r), s2_i = hn::Add(t1_i, t3_i);
                const auto s3_r = hn::Sub(t1_r, t3_r), s3_i = hn::Sub(t1_i, t3_i);

                const auto i2 = hn::Load(d, tmp_i + 16 + k), r2 = hn::Load(d, tmp_r + 16 + k);
                const auto w2_r = hn::Load(d, w_r_base + 8 + k), w2_i = hn::Load(d, w_i_base + 8 + k);
                const auto t2_r = hn::NegMulAdd(i2, w2_i, hn::Mul(r2, w2_r));
                const auto t2_i = hn::MulAdd(i2, w2_r, hn::Mul(r2, w2_i));

                const auto r0 = hn::Load(d, tmp_r + k), i0 = hn::Load(d, tmp_i + k);
                const auto s0_r = hn::Add(r0, t2_r), s0_i = hn::Add(i0, t2_i);
                const auto s1_r = hn::Sub(r0, t2_r), s1_i = hn::Sub(i0, t2_i);

                store_interleaved<is_forward>(hn::Add(s0_r, s2_r), hn::Add(s0_i, s2_i), d,
                                              reinterpret_cast<F*>(out + k));
                store_interleaved<is_forward>(hn::Add(s1_r, s3_i), hn::Sub(s1_i, s3_r), d,
                                              reinterpret_cast<F*>(out + 8 + k));
                store_interleaved<is_forward>(hn::Sub(s0_r, s2_r), hn::Sub(s0_i, s2_i), d,
                                              reinterpret_cast<F*>(out + 16 + k));
                store_interleaved<is_forward>(hn::Sub(s1_r, s3_i), hn::Add(s1_i, s3_r), d,
                                              reinterpret_cast<F*>(out + 24 + k));
            }
        }
    }

    /**
     * generate twiddles for order = 4 or order = 5
     * @tparam F
     * @param order
     * @param twiddles_r
     * @param twiddles_i
     */
    template <typename F>
    inline void generate_order_4_5_twiddles(const size_t order, hwy::AlignedFreeUniquePtr<F[]>& twiddles_r,
                                            hwy::AlignedFreeUniquePtr<F[]>& twiddles_i) {
        const size_t num_twiddles = (order == 4 ? 60 : 120);

        twiddles_r = hwy::AllocateAligned<F>(num_twiddles);
        twiddles_i = hwy::AllocateAligned<F>(num_twiddles);

        size_t offset = 0;
        size_t width = (order == 4 ? 4 : 8);
        for (size_t i = 0; i < 2; ++i) {
            const double angle_step = -2.0 * std::numbers::pi / static_cast<double>(width << 2);
            for (int mul = 1; mul < 4; ++mul) {
                const auto step = angle_step * static_cast<double>(mul);
                for (size_t k = 0; k < width; ++k, ++offset) {
                    const double angle = static_cast<double>(k) * step;
                    twiddles_r[offset] = static_cast<F>(std::cos(angle));
                    twiddles_i[offset] = static_cast<F>(std::sin(angle));
                }
            }
            width = width << 2;
        }
    }

    /**
     * generates twiddles for order > 5
     * @tparam F
     * @param stages
     * @param twiddles_shift
     * @param twiddles_aosoa
     */
    template <typename F>
    inline void generate_general_twiddles(std::vector<StageType>& stages, std::vector<size_t>& twiddles_shift,
                                          hwy::AlignedFreeUniquePtr<F[]>& twiddles_aosoa) {
        static constexpr size_t lanes = hn::Lanes(hn::ScalableTag<F>());
        static constexpr size_t width4_vec = std::max(static_cast<size_t>(4), lanes);
        // calculate twiddle shift for each stage
        {
            size_t width = (stages[0] == StageType::kRadix4FirstPass) ? 4 : 8;
            for (size_t i = 1; i < stages.size(); ++i) {
                const auto stage = stages[i];
                if (stage == StageType::kRadix4Width4) {
                    twiddles_shift[i] = 6 * width4_vec;
                    width = width << 2;
                } else if (stage == StageType::kRadix4 || stage == StageType::kRadix4LastPass) {
                    const size_t num_blocks = std::max<size_t>(1, width / lanes);
                    twiddles_shift[i] = num_blocks * 6 * lanes;
                    width = width << 2;
                } else if (stage == StageType::kRadix8) {
                    const size_t num_blocks = std::max<size_t>(1, width / lanes);
                    twiddles_shift[i] = num_blocks * 14 * lanes;
                    width = width << 3;
                }
            }
        }
        // allocate twiddle
        {
            const auto num_twiddles =
                std::accumulate(twiddles_shift.begin(), twiddles_shift.end(), static_cast<size_t>(0));
            twiddles_aosoa = hwy::AllocateAligned<F>(num_twiddles);
        }
        // calculate twiddle values
        {
            size_t offset = 0;
            size_t width = (stages[0] == StageType::kRadix4FirstPass) ? 4 : 8;
            for (size_t i = 1; i < stages.size(); ++i) {
                const auto stage = stages[i];
                if (stage == StageType::kRadix4Width4) {
                    const double angle_step = -2.0 * std::numbers::pi / static_cast<double>(width << 2);
                    for (size_t l = 0; l < width4_vec; ++l) {
                        const auto angle = static_cast<double>(l % 4) * angle_step;
                        static constexpr int muls[3] = {1, 2, 3};
                        for (int m = 0; m < 3; ++m) {
                            const auto a = angle * muls[m];
                            twiddles_aosoa[offset + 2 * m * width4_vec + l] = static_cast<F>(std::cos(a));
                            twiddles_aosoa[offset + (2 * m + 1) * width4_vec + l] = static_cast<F>(std::sin(a));
                        }
                    }
                    offset += twiddles_shift[i];
                    width = width << 2;
                } else if (stage == StageType::kRadix4 || stage == StageType::kRadix4LastPass) {
                    const size_t num_blocks = std::max<size_t>(1, width / lanes);
                    const double angle_step = -2.0 * std::numbers::pi / static_cast<double>(width << 2);
                    for (size_t b = 0; b < num_blocks; ++b) {
                        for (size_t l = 0; l < lanes; ++l) {
                            const size_t idx = (b * lanes + l) % width;
                            const auto angle = static_cast<double>(idx) * angle_step;
                            static constexpr int muls[3] = {1, 2, 3};
                            for (int m = 0; m < 3; ++m) {
                                const auto a = angle * muls[m];
                                twiddles_aosoa[offset + 2 * m * lanes + l] = static_cast<F>(std::cos(a));
                                twiddles_aosoa[offset + (2 * m + 1) * lanes + l] = static_cast<F>(std::sin(a));
                            }
                        }
                        offset += 6 * lanes;
                    }
                    width = width << 2;
                } else if (stage == StageType::kRadix8) {
                    const size_t num_blocks = std::max<size_t>(1, width / lanes);
                    const double angle_step = -2.0 * std::numbers::pi / static_cast<double>(width << 3);
                    for (size_t b = 0; b < num_blocks; ++b) {
                        for (size_t l = 0; l < lanes; ++l) {
                            const size_t idx = (b * lanes + l) % width;
                            const auto angle = static_cast<double>(idx) * angle_step;
                            static constexpr int muls[7] = {4, 2, 6, 1, 5, 3, 7};
                            for (int m = 0; m < 7; ++m) {
                                const auto a = angle * muls[m];
                                twiddles_aosoa[offset + 2 * m * lanes + l] = static_cast<F>(std::cos(a));
                                twiddles_aosoa[offset + (2 * m + 1) * lanes + l] = static_cast<F>(std::sin(a));
                            }
                        }
                        offset += 14 * lanes;
                    }
                    width = width << 3;
                }
            }
        }
    }
}
