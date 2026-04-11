#pragma once

#include "zlfft_common.hpp"
#include "zlfft_common_aosoa.hpp"
#include "zlfft_common_high_order.hpp"
#include "simd_low_order_aosoa1.hpp"
#include <memory>
#include <vector>
#include <iostream>
#include <hwy/cache_control.h>

#if defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <unistd.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

namespace zlfft {
    namespace hn = hwy::HWY_NAMESPACE;

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

    template <typename F>
    void FastComplexTranspose(const std::complex<F>* aos_matrix,
                              std::complex<F>* out_buffer,
                              size_t l_, size_t M, size_t M_padded) {

        static constexpr hn::ScalableTag<F> d;
        static constexpr size_t lanes = hn::Lanes(d);
        static constexpr size_t c_per_vec = lanes / 2; // Number of complex elements per vector

        // Macro-tiles tuned to fit comfortably within L1/L2 caches
        static constexpr size_t MACRO_TILE_C = (sizeof(F) == 4) ? 128 : 64;
        static constexpr size_t MACRO_TILE_K = (sizeof(F) == 4) ? 128 : 64;

        const F* ptr_in = reinterpret_cast<const F*>(aos_matrix);
        F* ptr_out = reinterpret_cast<F*>(out_buffer);

        for (size_t c_macro = 0; c_macro < l_; c_macro += MACRO_TILE_C) {
            for (size_t k_macro = 0; k_macro < M; k_macro += MACRO_TILE_K) {

                const size_t c_max = std::min(c_macro + MACRO_TILE_C, l_);
                const size_t k_max = std::min(k_macro + MACRO_TILE_K, M);

                size_t c = c_macro;
                for (; c + c_per_vec <= c_max; c += c_per_vec) {
                    size_t k = k_macro;

                    for (; k + c_per_vec <= k_max; k += c_per_vec) {

                        // --- 1. Software Prefetching ---
                        // Fetch cache lines 2 iterations ahead to hide strided load latency
                        if (k + c_per_vec * 2 < k_max) {
                            for (size_t p = 0; p < c_per_vec; ++p) {
                                hwy::Prefetch(&ptr_in[((c + p) * M_padded + (k + c_per_vec * 2)) * 2]);
                            }
                        }

                        // --- 2. In-Register Block Transpose Networks ---

                        // CASE A: 1 Complex Number per Vector (e.g., SSE2 Double, 128-bit)
                        if constexpr (c_per_vec == 1) {
                            auto v0 = hn::LoadU(d, &ptr_in[((c + 0) * M_padded + k) * 2]);
                            hn::Stream(v0, d, &ptr_out[((k + 0) * l_ + c) * 2]);
                        }

                        // CASE B: 2 Complex Numbers per Vector (e.g., NEON/SSE2 Float or AVX2 Double)
                        else if constexpr (c_per_vec == 2 && sizeof(F) == 4) {
                            // Float, 128-bit: Treat as 64-bit blocks
                            auto v0 = hn::LoadU(d, &ptr_in[((c + 0) * M_padded + k) * 2]);
                            auto v1 = hn::LoadU(d, &ptr_in[((c + 1) * M_padded + k) * 2]);

                            const hn::Repartition<uint64_t, decltype(d)> d64;
                            auto v0_64 = hn::BitCast(d64, v0);
                            auto v1_64 = hn::BitCast(d64, v1);

                            auto tr0 = hn::BitCast(d, hn::InterleaveLower(d64, v0_64, v1_64));
                            auto tr1 = hn::BitCast(d, hn::InterleaveUpper(d64, v0_64, v1_64));

                            hn::Stream(tr0, d, &ptr_out[((k + 0) * l_ + c) * 2]);
                            hn::Stream(tr1, d, &ptr_out[((k + 1) * l_ + c) * 2]);
                        } else if constexpr (c_per_vec == 2 && sizeof(F) == 8) {
                            // Double, 256-bit: Combine half-vectors
                            auto v0 = hn::LoadU(d, &ptr_in[((c + 0) * M_padded + k) * 2]);
                            auto v1 = hn::LoadU(d, &ptr_in[((c + 1) * M_padded + k) * 2]);

                            // Combine puts arg2 in the upper 128-bits and arg1 in the lower 128-bits
                            auto tr0 = hn::Combine(d, hn::LowerHalf(d, v1), hn::LowerHalf(d, v0));
                            auto tr1 = hn::Combine(d, hn::UpperHalf(d, v1), hn::UpperHalf(d, v0));

                            hn::Stream(tr0, d, &ptr_out[((k + 0) * l_ + c) * 2]);
                            hn::Stream(tr1, d, &ptr_out[((k + 1) * l_ + c) * 2]);
                        }

                        // CASE C: 4 Complex Numbers per Vector (e.g., AVX2 Float, 256-bit)
                        else if constexpr (c_per_vec == 4 && sizeof(F) == 4) {
                            auto v0 = hn::LoadU(d, &ptr_in[((c + 0) * M_padded + k) * 2]);
                            auto v1 = hn::LoadU(d, &ptr_in[((c + 1) * M_padded + k) * 2]);
                            auto v2 = hn::LoadU(d, &ptr_in[((c + 2) * M_padded + k) * 2]);
                            auto v3 = hn::LoadU(d, &ptr_in[((c + 3) * M_padded + k) * 2]);

                            const hn::Repartition<uint64_t, decltype(d)> d64;
                            auto v0_64 = hn::BitCast(d64, v0);
                            auto v1_64 = hn::BitCast(d64, v1);
                            auto v2_64 = hn::BitCast(d64, v2);
                            auto v3_64 = hn::BitCast(d64, v3);

                            // Stage 1: Interleave pairs (operates on 128-bit lane boundaries in AVX2)
                            auto t0 = hn::InterleaveLower(d64, v0_64, v1_64);
                            auto t1 = hn::InterleaveUpper(d64, v0_64, v1_64);
                            auto t2 = hn::InterleaveLower(d64, v2_64, v3_64);
                            auto t3 = hn::InterleaveUpper(d64, v2_64, v3_64);

                            // Stage 2: Combine 128-bit halves to finalize the 4x4 matrix
                            auto tr0 = hn::BitCast(d, hn::Combine(d64, hn::LowerHalf(d64, t2), hn::LowerHalf(d64, t0)));
                            auto tr1 = hn::BitCast(d, hn::Combine(d64, hn::LowerHalf(d64, t3), hn::LowerHalf(d64, t1)));
                            auto tr2 = hn::BitCast(d, hn::Combine(d64, hn::UpperHalf(d64, t2), hn::UpperHalf(d64, t0)));
                            auto tr3 = hn::BitCast(d, hn::Combine(d64, hn::UpperHalf(d64, t3), hn::UpperHalf(d64, t1)));

                            hn::Stream(tr0, d, &ptr_out[((k + 0) * l_ + c) * 2]);
                            hn::Stream(tr1, d, &ptr_out[((k + 1) * l_ + c) * 2]);
                            hn::Stream(tr2, d, &ptr_out[((k + 2) * l_ + c) * 2]);
                            hn::Stream(tr3, d, &ptr_out[((k + 3) * l_ + c) * 2]);
                        } else {
                            for (size_t i = 0; i < c_per_vec; ++i) {
                                for (size_t j = 0; j < c_per_vec; ++j) {
                                    out_buffer[(k + j) * l_ + (c + i)] = aos_matrix[(c + i) * M_padded + (k + j)];
                                }
                            }
                        }
                    }

                    for (; k < k_max; ++k) {
                        for (size_t i = 0; i < c_per_vec; ++i) {
                            out_buffer[k * l_ + (c + i)] = aos_matrix[(c + i) * M_padded + k];
                        }
                    }
                }

                for (; c < c_max; ++c) {
                    for (size_t k = k_macro; k < k_max; ++k) {
                        out_buffer[k * l_ + c] = aos_matrix[c * M_padded + k];
                    }
                }
            }
        }
    }

    template <typename F>
    class HybridAoSoA2 {
        using C = std::complex<F>;

    private:
        static constexpr hn::ScalableTag<F> d{};
        static constexpr size_t lanes = hn::Lanes(d);
        static constexpr size_t width4_vec = std::max(static_cast<size_t>(4), lanes);

        size_t order_;
        size_t n_;
        size_t m_;
        size_t l_;
        size_t max_m_;
        size_t stride_;
        size_t p_;

        std::vector<common::StageType> micro_stages_;
        std::vector<size_t> digit_rev_4_;

        hwy::AlignedFreeUniquePtr<F[]> macro_twiddles_;
        hwy::AlignedFreeUniquePtr<F[]> micro_twiddles_;
        hwy::AlignedFreeUniquePtr<F[]> workspace_;

        std::unique_ptr<SIMDLowOrderAOSOA1<F>> low_order_fft_;

    public:
        explicit HybridAoSoA2(const size_t order) :
            order_(order) {
            n_ = static_cast<size_t>(1) << order_;

            size_t l1d = get_l1d_cache_size();
            size_t working_set_per_item = 6 * sizeof(F);
            size_t max_m_val = (l1d / 2) / working_set_per_item;
            max_m_ = (max_m_val == 0) ? 0 : std::bit_width(max_m_val) - 1;

            if (order_ <= max_m_ + 4 || order_ <= 5) {
                low_order_fft_ = std::make_unique<SIMDLowOrderAOSOA1<F>>(order_);
                return;
            }

            m_ = max_m_;
            if ((order_ - m_) % 2 != 0) {
                m_--;
            }

            p_ = (order_ - m_) / 2;
            l_ = static_cast<size_t>(1) << (order_ - m_);
            size_t m_pow = static_cast<size_t>(1) << m_;

            if (m_ % 2 == 1) {
                micro_stages_.emplace_back(common::StageType::kRadix8FirstPass);
                for (size_t i = 3; i < m_ - 2; i += 2) {
                    micro_stages_.emplace_back(common::StageType::kRadix4);
                }
            } else {
                micro_stages_.emplace_back(common::StageType::kRadix4FirstPass);
                micro_stages_.emplace_back(common::StageType::kRadix4Width4);
                for (size_t i = 4; i < m_ - 2; i += 2) {
                    micro_stages_.emplace_back(common::StageType::kRadix4);
                }
            }
            micro_stages_.emplace_back(common::StageType::kRadix4LastPass);

            const auto pad = (64 / sizeof(F)) + 16;
            stride_ = n_ + (8 * l_) + pad;
            workspace_ = hwy::AllocateAligned<F>(hwy::RoundUpTo(4 * stride_ + 4 * m_pow + pad, lanes));

            size_t num_macro_tw_elements = 0;
            for (size_t p = 0; p < p_; ++p) {
                size_t sub_n = n_ >> (2 * p);
                size_t width = sub_n >> 2;
                size_t num_blocks = std::max<size_t>(1, width / lanes);
                num_macro_tw_elements += num_blocks * 6 * lanes;
            }
            macro_twiddles_ = hwy::AllocateAligned<F>(num_macro_tw_elements);

            size_t offset = 0;
            for (size_t p = 0; p < p_; ++p) {
                size_t sub_n = n_ >> (2 * p);
                size_t width = sub_n >> 2;
                size_t num_blocks = std::max<size_t>(1, width / lanes);
                const double angle_step = -2.0 * std::numbers::pi / static_cast<double>(sub_n);

                for (size_t b = 0; b < num_blocks; ++b) {
                    for (size_t l = 0; l < lanes; ++l) {
                        const size_t idx = (b * lanes + l) % width;
                        const double angle = static_cast<double>(idx) * angle_step;
                        macro_twiddles_[offset + l] = static_cast<F>(std::cos(angle * 1));
                        macro_twiddles_[offset + lanes + l] = static_cast<F>(std::sin(angle * 1));
                        macro_twiddles_[offset + 2 * lanes + l] = static_cast<F>(std::cos(angle * 2));
                        macro_twiddles_[offset + 3 * lanes + l] = static_cast<F>(std::sin(angle * 2));
                        macro_twiddles_[offset + 4 * lanes + l] = static_cast<F>(std::cos(angle * 3));
                        macro_twiddles_[offset + 5 * lanes + l] = static_cast<F>(std::sin(angle * 3));
                    }
                    offset += 6 * lanes;
                }
            }

            size_t num_twiddle_elements = 0;
            size_t sim_width = (micro_stages_[0] == common::StageType::kRadix4FirstPass) ? 4 : 8;

            for (size_t i = 1; i < micro_stages_.size(); ++i) {
                const auto stage = micro_stages_[i];
                if (stage == common::StageType::kRadix4Width4) {
                    num_twiddle_elements += 6 * width4_vec;
                    sim_width = sim_width << 2;
                } else if (stage == common::StageType::kRadix4 || stage == common::StageType::kRadix4LastPass) {
                    size_t num_blocks = std::max<size_t>(1, sim_width / lanes);
                    num_twiddle_elements += num_blocks * 6 * lanes;
                    sim_width = sim_width << 2;
                } else if (stage == common::StageType::kRadix8) {
                    size_t num_blocks = std::max<size_t>(1, sim_width / lanes);
                    num_twiddle_elements += num_blocks * 14 * lanes;
                    sim_width = sim_width << 3;
                }
            }

            micro_twiddles_ = hwy::AllocateAligned<F>(num_twiddle_elements);
            size_t micro_offset = 0;
            size_t gen_width = (micro_stages_[0] == common::StageType::kRadix4FirstPass) ? 4 : 8;

            for (size_t i = 1; i < micro_stages_.size(); ++i) {
                const auto stage = micro_stages_[i];
                if (stage == common::StageType::kRadix4Width4) {
                    const double angle_step = -2.0 * std::numbers::pi / static_cast<double>(gen_width << 2);
                    for (size_t l = 0; l < width4_vec; ++l) {
                        const double angle = static_cast<double>(l % 4) * angle_step;
                        micro_twiddles_[micro_offset + l] = static_cast<F>(std::cos(angle * 1));
                        micro_twiddles_[micro_offset + width4_vec + l] = static_cast<F>(std::sin(angle * 1));
                        micro_twiddles_[micro_offset + width4_vec * 2 + l] = static_cast<F>(std::cos(angle * 2));
                        micro_twiddles_[micro_offset + width4_vec * 3 + l] = static_cast<F>(std::sin(angle * 2));
                        micro_twiddles_[micro_offset + width4_vec * 4 + l] = static_cast<F>(std::cos(angle * 3));
                        micro_twiddles_[micro_offset + width4_vec * 5 + l] = static_cast<F>(std::sin(angle * 3));
                    }
                    micro_offset += 6 * width4_vec;
                    gen_width = gen_width << 2;
                } else if (stage == common::StageType::kRadix4 || stage == common::StageType::kRadix4LastPass) {
                    size_t num_blocks = std::max<size_t>(1, gen_width / lanes);
                    const double angle_step = -2.0 * std::numbers::pi / static_cast<double>(gen_width << 2);

                    for (size_t b = 0; b < num_blocks; ++b) {
                        for (size_t l = 0; l < lanes; ++l) {
                            const size_t idx = (b * lanes + l) % gen_width;
                            const double angle = static_cast<double>(idx) * angle_step;
                            micro_twiddles_[micro_offset + l] = static_cast<F>(std::cos(angle * 1));
                            micro_twiddles_[micro_offset + lanes + l] = static_cast<F>(std::sin(angle * 1));
                            micro_twiddles_[micro_offset + 2 * lanes + l] = static_cast<F>(std::cos(angle * 2));
                            micro_twiddles_[micro_offset + 3 * lanes + l] = static_cast<F>(std::sin(angle * 2));
                            micro_twiddles_[micro_offset + 4 * lanes + l] = static_cast<F>(std::cos(angle * 3));
                            micro_twiddles_[micro_offset + 5 * lanes + l] = static_cast<F>(std::sin(angle * 3));
                        }
                        micro_offset += 6 * lanes;
                    }
                    gen_width = gen_width << 2;
                } else if (stage == common::StageType::kRadix8) {
                    size_t num_blocks = std::max<size_t>(1, gen_width / lanes);
                    const double angle_step = -2.0 * std::numbers::pi / static_cast<double>(gen_width << 3);

                    for (size_t b = 0; b < num_blocks; ++b) {
                        for (size_t l = 0; l < lanes; ++l) {
                            const size_t idx = (b * lanes + l) % gen_width;
                            const double angle = static_cast<double>(idx) * angle_step;
                            const int muls[7] = {3, 1, 5, 0, 4, 2, 6};
                            for (int m_val = 0; m_val < 7; ++m_val) {
                                micro_twiddles_[micro_offset + 2 * m_val * lanes + l] = static_cast<F>(std::cos(
                                    angle * muls[m_val]));
                                micro_twiddles_[micro_offset + (2 * m_val + 1) * lanes + l] = static_cast<F>(std::sin(
                                    angle * muls[m_val]));
                            }
                        }
                        micro_offset += 14 * lanes;
                    }
                    gen_width = gen_width << 3;
                }
            }

            digit_rev_4_.resize(l_);
            for (size_t i = 0; i < l_; ++i) {
                size_t rev = 0;
                size_t temp = i;
                for (size_t d_idx = 0; d_idx < p_; ++d_idx) {
                    size_t digit = temp & 3;
                    temp >>= 2;
                    rev = (rev << 2) | digit;
                }
                digit_rev_4_[i] = rev;
            }
        }

        void forward(std::span<C> in_buffer, std::span<C> out_buffer) {
            if (low_order_fft_) {
                low_order_fft_->forward(in_buffer, out_buffer);
                return;
            }

            F* __restrict buf0 = workspace_.get();
            std::complex<F>* aos_matrix = reinterpret_cast<std::complex<F>*>(workspace_.get() + 2 * stride_);

            const F* __restrict w_ptr = macro_twiddles_.get();

            radix4_first_pass_dif_aosoa(in_buffer.data(), buf0, n_, w_ptr);
            size_t w_offset = std::max<size_t>(1, (n_ >> 2) / lanes) * 6 * lanes;
            w_ptr += w_offset;

            for (size_t p = 1; p < p_; ++p) {
                size_t sub_n = n_ >> (2 * p);
                size_t width = sub_n >> 2;
                radix4_dif_aosoa_inplace(buf0, n_, width, w_ptr);
                w_ptr += std::max<size_t>(1, width / lanes) * 6 * lanes;
            }

            size_t M = static_cast<size_t>(1) << m_;
            const size_t M_padded = M + 8;

            F* __restrict l1_buf_A = workspace_.get() + 4 * stride_;
            F* __restrict l1_buf_B = workspace_.get() + 4 * stride_ + 2 * M;

            static constexpr size_t MACRO_TILE_C = 64;
            static constexpr size_t MACRO_TILE_K = 64;
            static constexpr size_t c_per_vec = lanes / 2;

            for (size_t c_macro = 0; c_macro < l_; c_macro += MACRO_TILE_C) {
                const size_t c_max = std::min(c_macro + MACRO_TILE_C, l_);
                const size_t c_chunk_size = c_max - c_macro;

                for (size_t c_offset = 0; c_offset < c_chunk_size; ++c_offset) {
                    const size_t reversed_c = c_macro + c_offset;
                    const size_t l_idx = digit_rev_4_[reversed_c];

                    F* __restrict main_mem_in = buf0 + 2 * l_idx * M;
                    F* current_in = main_mem_in;
                    F* current_out = l1_buf_A;

                    if (micro_stages_[0] == common::StageType::kRadix4FirstPass) {
                        common::radix4_first_pass_aosoa(current_in, current_out, M);
                    } else {
                        common::radix8_first_pass_aosoa(current_in, current_out, M);
                    }

                    current_in = l1_buf_A;
                    current_out = l1_buf_B;
                    const F* __restrict uw_ptr = micro_twiddles_.get();

                    size_t width = (micro_stages_[0] == common::StageType::kRadix4FirstPass) ? 4 : 8;
                    for (size_t i = 1; i < micro_stages_.size() - 1; ++i) {
                        const auto stage = micro_stages_[i];
                        switch (stage) {
                        case common::StageType::kRadix4Width4: {
                            common::radix4_width4_aosoa(current_in, current_out, M, uw_ptr);
                            uw_ptr += 6 * width4_vec;
                            width = width << 2;
                            break;
                        }
                        case common::StageType::kRadix4: {
                            common::radix4_aosoa(current_in, current_out, M, width, uw_ptr);
                            const size_t num_blocks = std::max<size_t>(1, width / lanes);
                            uw_ptr += num_blocks * 6 * lanes;
                            width = width << 2;
                            break;
                        }
                        case common::StageType::kRadix8: {
                            common::radix8_aosoa(current_in, current_out, M, width, uw_ptr);
                            const size_t num_blocks = std::max<size_t>(1, width / lanes);
                            uw_ptr += num_blocks * 14 * lanes;
                            width = width << 3;
                            break;
                        }
                        default:
                            break;
                        }
                        std::swap(current_in, current_out);
                    }

                    std::complex<F>* out_aos = aos_matrix + c_offset * M_padded;
                    common::radix4_last_pass_fused_aosoa(current_in, out_aos, M, width, uw_ptr);
                }

                for (size_t k_macro = 0; k_macro < M; k_macro += MACRO_TILE_K) {
                    const size_t k_max = std::min(k_macro + MACRO_TILE_K, M);
                    for (size_t k = k_macro; k < k_max; ++k) {
                        size_t c = 0;
                        const size_t out_shift = k * l_ + c_macro;
                        for (; c + c_per_vec <= c_chunk_size; c += c_per_vec) {
                            alignas(64) std::complex<F> tmp[32];
                            for (size_t i = 0; i < c_per_vec; ++i) {
                                tmp[i] = aos_matrix[(c + i) * M_padded + k];
                            }
                            auto v = hn::Load(d, reinterpret_cast<const F*>(tmp));
                            hn::Stream(v, d, reinterpret_cast<F*>(out_buffer.data() + out_shift + c));
                        }
                        for (; c < c_chunk_size; ++c) {
                            out_buffer[out_shift + c] = aos_matrix[c * M_padded + k];
                        }
                    }
                }
            }
        }
    };
}
