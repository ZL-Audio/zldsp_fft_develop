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

#if defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <unistd.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

namespace zlfft::common {
    inline size_t get_l1d_cache_size() {
        size_t l1d_size = 32768;
#if defined(__APPLE__)
        size_t size = sizeof(l1d_size);
        sysctlbyname("hw.l1dcachesize", &l1d_size, &size, nullptr, 0);
#elif defined(__linux__)
        long size = sysconf(_SC_LEVEL1_DCACHE_SIZE);
        if (size > 0) {
            l1d_size = static_cast<size_t>(size);
        }
#elif defined(_WIN32)
        DWORD buffer_size = 0;
        GetLogicalProcessorInformation(nullptr, &buffer_size);
        if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
            std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(
                buffer_size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
            if (GetLogicalProcessorInformation(buffer.data(), &buffer_size)) {
                for (const auto& info : buffer) {
                    if (info.Relationship == RelationCache) {
                        if (info.Cache.Level == 1 && (info.Cache.Type == CacheData || info.Cache.Type ==
                            CacheUnified)) {
                            l1d_size = info.Cache.Size;
                            break;
                        }
                    }
                }
            }
        }
#endif
        return l1d_size;
    }

    inline size_t get_l2_cache_size() {
        size_t l2_size = 262144;
#if defined(__APPLE__)
        size_t size = sizeof(l2_size);
        sysctlbyname("hw.l2cachesize", &l2_size, &size, nullptr, 0);
#elif defined(__linux__)
        long size = sysconf(_SC_LEVEL2_CACHE_SIZE);
        if (size > 0) {
            l2_size = static_cast<size_t>(size);
        }
#elif defined(_WIN32)
        DWORD buffer_size = 0;
        GetLogicalProcessorInformation(nullptr, &buffer_size);
        if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
            std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(
                buffer_size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
            if (GetLogicalProcessorInformation(buffer.data(), &buffer_size)) {
                for (const auto& info : buffer) {
                    if (info.Relationship == RelationCache) {
                        if (info.Cache.Level == 2 &&
                            (info.Cache.Type == CacheData || info.Cache.Type == CacheUnified)) {
                            l2_size = info.Cache.Size;
                            break;
                        }
                    }
                }
            }
        }
#endif
        return l2_size;
    }

    inline size_t get_l3_cache_size() {
        size_t l3_size = 0;
#if defined(__APPLE__)
        l3_size = 0;
#elif defined(__linux__)
        long size = sysconf(_SC_LEVEL3_CACHE_SIZE);
        if (size > 0) {
            l3_size = static_cast<size_t>(size);
        }
#elif defined(_WIN32)
        DWORD buffer_size = 0;
        GetLogicalProcessorInformation(nullptr, &buffer_size);
        if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
            std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(
                buffer_size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
            if (GetLogicalProcessorInformation(buffer.data(), &buffer_size)) {
                for (const auto& info : buffer) {
                    if (info.Relationship == RelationCache) {
                        if (info.Cache.Level == 3 &&
                            (info.Cache.Type == CacheData || info.Cache.Type == CacheUnified)) {
                            l3_size = info.Cache.Size;
                            break;
                        }
                    }
                }
            }
        }
#endif
        return l3_size;
    }

    template <typename F>
    inline void radix4_first_pass_dif_aosoa(const std::complex<F>* __restrict in,
                                            F* __restrict out_aosoa,
                                            const size_t n,
                                            const F* __restrict w_ptr) {
        static constexpr hn::ScalableTag<F> d;
        static constexpr size_t lanes = hn::Lanes(d);

        const size_t width = n >> 2;
        const size_t double_width = width << 1;
        const size_t triple_width = width * 3;

        const F* local_w_ptr = w_ptr;

        for (size_t i = 0; i < width; i += lanes) {
            hn::Vec<decltype(d)> r0, i0, r1, i1, r2, i2, r3, i3;
            hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + i), r0, i0);
            hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + i + width), r1, i1);
            hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + i + double_width), r2, i2);
            hn::LoadInterleaved2(d, reinterpret_cast<const F*>(in + i + triple_width), r3, i3);

            const auto t0_r = hn::Add(r0, r2), t0_i = hn::Add(i0, i2);
            const auto t1_r = hn::Sub(r0, r2), t1_i = hn::Sub(i0, i2);
            const auto t2_r = hn::Add(r1, r3), t2_i = hn::Add(i1, i3);
            const auto t3_r = hn::Sub(r1, r3), t3_i = hn::Sub(i1, i3);

            const auto y0_r = hn::Add(t0_r, t2_r), y0_i = hn::Add(t0_i, t2_i);
            const auto y1_r = hn::Add(t1_r, t3_i), y1_i = hn::Sub(t1_i, t3_r);
            const auto y2_r = hn::Sub(t0_r, t2_r), y2_i = hn::Sub(t0_i, t2_i);
            const auto y3_r = hn::Sub(t1_r, t3_i), y3_i = hn::Add(t1_i, t3_r);

            const auto w1_r = hn::Load(d, local_w_ptr);
            const auto w1_i = hn::Load(d, local_w_ptr + lanes);
            const auto w2_r = hn::Load(d, local_w_ptr + lanes * 2);
            const auto w2_i = hn::Load(d, local_w_ptr + lanes * 3);
            const auto w3_r = hn::Load(d, local_w_ptr + lanes * 4);
            const auto w3_i = hn::Load(d, local_w_ptr + lanes * 5);
            local_w_ptr += lanes * 6;

            const auto out1_r = hn::NegMulAdd(y1_i, w1_i, hn::Mul(y1_r, w1_r));
            const auto out1_i = hn::MulAdd(y1_i, w1_r, hn::Mul(y1_r, w1_i));
            const auto out2_r = hn::NegMulAdd(y2_i, w2_i, hn::Mul(y2_r, w2_r));
            const auto out2_i = hn::MulAdd(y2_i, w2_r, hn::Mul(y2_r, w2_i));
            const auto out3_r = hn::NegMulAdd(y3_i, w3_i, hn::Mul(y3_r, w3_r));
            const auto out3_i = hn::MulAdd(y3_i, w3_r, hn::Mul(y3_r, w3_i));

            hn::Store(y0_r, d, out_aosoa + (i << 1));
            hn::Store(y0_i, d, out_aosoa + (i << 1) + lanes);

            hn::Store(out1_r, d, out_aosoa + ((i + width) << 1));
            hn::Store(out1_i, d, out_aosoa + ((i + width) << 1) + lanes);

            hn::Store(out2_r, d, out_aosoa + ((i + double_width) << 1));
            hn::Store(out2_i, d, out_aosoa + ((i + double_width) << 1) + lanes);

            hn::Store(out3_r, d, out_aosoa + ((i + triple_width) << 1));
            hn::Store(out3_i, d, out_aosoa + ((i + triple_width) << 1) + lanes);
        }
    }

    template <typename F>
    inline void radix4_dif_aosoa_inplace(F* __restrict workspace,
                                         const size_t n, const size_t width,
                                         const F* __restrict w_ptr) {
        static constexpr hn::ScalableTag<F> d;
        static constexpr size_t lanes = hn::Lanes(d);

        const size_t sub_n = width << 2;
        const size_t double_width = width << 1;
        const size_t triple_width = width * 3;

        for (size_t block = 0; block < n; block += sub_n) {
            F* __restrict in_ptr = workspace + (block << 1);

            const F* __restrict local_w_ptr = w_ptr;

            for (size_t i = 0; i < width; i += lanes) {
                const auto r0 = hn::Load(d, in_ptr + (i << 1));
                const auto i0 = hn::Load(d, in_ptr + (i << 1) + lanes);
                const auto r1 = hn::Load(d, in_ptr + ((i + width) << 1));
                const auto i1 = hn::Load(d, in_ptr + ((i + width) << 1) + lanes);
                const auto r2 = hn::Load(d, in_ptr + ((i + double_width) << 1));
                const auto i2 = hn::Load(d, in_ptr + ((i + double_width) << 1) + lanes);
                const auto r3 = hn::Load(d, in_ptr + ((i + triple_width) << 1));
                const auto i3 = hn::Load(d, in_ptr + ((i + triple_width) << 1) + lanes);

                const auto t0_r = hn::Add(r0, r2), t0_i = hn::Add(i0, i2);
                const auto t1_r = hn::Sub(r0, r2), t1_i = hn::Sub(i0, i2);
                const auto t2_r = hn::Add(r1, r3), t2_i = hn::Add(i1, i3);
                const auto t3_r = hn::Sub(r1, r3), t3_i = hn::Sub(i1, i3);

                const auto y0_r = hn::Add(t0_r, t2_r), y0_i = hn::Add(t0_i, t2_i);
                const auto y1_r = hn::Add(t1_r, t3_i), y1_i = hn::Sub(t1_i, t3_r);
                const auto y2_r = hn::Sub(t0_r, t2_r), y2_i = hn::Sub(t0_i, t2_i);
                const auto y3_r = hn::Sub(t1_r, t3_i), y3_i = hn::Add(t1_i, t3_r);

                const auto w1_r = hn::Load(d, local_w_ptr);
                const auto w1_i = hn::Load(d, local_w_ptr + lanes);
                const auto w2_r = hn::Load(d, local_w_ptr + lanes * 2);
                const auto w2_i = hn::Load(d, local_w_ptr + lanes * 3);
                const auto w3_r = hn::Load(d, local_w_ptr + lanes * 4);
                const auto w3_i = hn::Load(d, local_w_ptr + lanes * 5);
                local_w_ptr += lanes * 6;

                const auto out1_r = hn::NegMulAdd(y1_i, w1_i, hn::Mul(y1_r, w1_r));
                const auto out1_i = hn::MulAdd(y1_i, w1_r, hn::Mul(y1_r, w1_i));

                const auto out2_r = hn::NegMulAdd(y2_i, w2_i, hn::Mul(y2_r, w2_r));
                const auto out2_i = hn::MulAdd(y2_i, w2_r, hn::Mul(y2_r, w2_i));

                const auto out3_r = hn::NegMulAdd(y3_i, w3_i, hn::Mul(y3_r, w3_r));
                const auto out3_i = hn::MulAdd(y3_i, w3_r, hn::Mul(y3_r, w3_i));

                hn::Store(y0_r, d, in_ptr + (i << 1));
                hn::Store(y0_i, d, in_ptr + (i << 1) + lanes);

                hn::Store(out1_r, d, in_ptr + ((i + width) << 1));
                hn::Store(out1_i, d, in_ptr + ((i + width) << 1) + lanes);

                hn::Store(out2_r, d, in_ptr + ((i + double_width) << 1));
                hn::Store(out2_i, d, in_ptr + ((i + double_width) << 1) + lanes);

                hn::Store(out3_r, d, in_ptr + ((i + triple_width) << 1));
                hn::Store(out3_i, d, in_ptr + ((i + triple_width) << 1) + lanes);
            }
        }
    }
}
