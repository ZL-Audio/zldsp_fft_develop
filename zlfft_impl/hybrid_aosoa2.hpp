#pragma once

#include "zlfft_common.hpp"
#include "zlfft_common_aosoa.hpp"
#include "zlfft_common_aosoa_hybrid.hpp"
#include "zlfft_common_high_order.hpp"
#include "simd_low_order_aosoa1.hpp"
#include <memory>
#include <vector>
#include <iostream>
#include <hwy/cache_control.h>

namespace zlfft {
    namespace hn = hwy::HWY_NAMESPACE;

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

            size_t l1d = common::get_l1d_cache_size();
            size_t working_set_per_item = 6 * sizeof(F);
            size_t max_m_val = (l1d / 2) / working_set_per_item;
            max_m_ = (max_m_val == 0) ? 0 : std::bit_width(max_m_val) - 1;

            if (order_ <= max_m_ + 5 || order_ <= 5) {
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

            common::radix4_first_pass_dif_aosoa(in_buffer.data(), buf0, n_, w_ptr);
            size_t w_offset = std::max<size_t>(1, (n_ >> 2) / lanes) * 6 * lanes;
            w_ptr += w_offset;

            for (size_t p = 1; p < p_; ++p) {
                size_t sub_n = n_ >> (2 * p);
                size_t width = sub_n >> 2;
                common::radix4_dif_aosoa_inplace(buf0, n_, width, w_ptr);
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
                    if (c_offset + 2 < c_chunk_size) {
                        const size_t future_c = c_macro + c_offset + 2;
                        const size_t future_l_idx = digit_rev_4_[future_c];
                        const F* future_in = buf0 + 2 * future_l_idx * M;

                        hwy::Prefetch(future_in);
                        hwy::Prefetch(future_in + 16);
                    }

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
