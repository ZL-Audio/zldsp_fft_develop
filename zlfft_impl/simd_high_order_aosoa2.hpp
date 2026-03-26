#pragma once

#include "zlfft_common_aosoa.hpp"
#include "zlfft_common_high_order.hpp"

namespace zlfft {
    namespace hn = hwy::HWY_NAMESPACE;

    template <typename F>
    class SIMDHighOrderAOSOA2 {
        using C = std::complex<F>;

    public:
        explicit SIMDHighOrderAOSOA2(const size_t order) : order_(order) {
            if (order < 10) return;

            const bool is_even = (order % 2 == 0);
            sub_order_ = is_even ? (order / 2) : ((order - 1) / 2);
            sub_n_ = static_cast<size_t>(1) << sub_order_;
            padded_sub_n_ = sub_n_ + 16;

            if (sub_order_ == 4) {
                stages_ = {common::StageType::kRadix4FirstPass, common::StageType::kRadix4};
            } else if (sub_order_ == 5) {
                stages_ = {common::StageType::kRadix8FirstPass, common::StageType::kRadix4};
            } else if (sub_order_ == 6) {
                stages_ = {common::StageType::kRadix4FirstPass, common::StageType::kRadix4Width4, common::StageType::kRadix4};
            } else {
                if (sub_order_ % 2 == 1) {
                    stages_.emplace_back(common::StageType::kRadix8FirstPass);
                    for (size_t i = 3; i < sub_order_; i += 2) {
                        stages_.emplace_back(common::StageType::kRadix4);
                    }
                } else {
                    stages_.emplace_back(common::StageType::kRadix4FirstPass);
                    stages_.emplace_back(common::StageType::kRadix4Width4);
                    for (size_t i = 4; i < sub_order_; i += 2) {
                        stages_.emplace_back(common::StageType::kRadix4);
                    }
                }
            }

            size_t num_twiddle_elements = 0;
            size_t sim_width = (stages_[0] == common::StageType::kRadix4FirstPass) ? 4 : 8;

            for (size_t i = 1; i < stages_.size(); ++i) {
                const auto stage = stages_[i];
                if (stage == common::StageType::kRadix4Width4) {
                    num_twiddle_elements += 6 * width4_vec;
                    sim_width = sim_width << 2;
                } else if (stage == common::StageType::kRadix4) {
                    size_t num_blocks = std::max<size_t>(1, sim_width / lanes);
                    num_twiddle_elements += num_blocks * 6 * lanes;
                    sim_width = sim_width << 2;
                } else if (stage == common::StageType::kRadix8) {
                    size_t num_blocks = std::max<size_t>(1, sim_width / lanes);
                    num_twiddle_elements += num_blocks * 14 * lanes;
                    sim_width = sim_width << 3;
                }
            }

            sub_twiddles_aosoa_ = hwy::AllocateAligned<F>(num_twiddle_elements);
            size_t offset = 0;
            size_t gen_width = (stages_[0] == common::StageType::kRadix4FirstPass) ? 4 : 8;

            for (size_t i = 1; i < stages_.size(); ++i) {
                const auto stage = stages_[i];
                if (stage == common::StageType::kRadix4Width4) {
                    const double angle_step = -2.0 * std::numbers::pi / static_cast<double>(gen_width << 2);
                    for (size_t l = 0; l < width4_vec; ++l) {
                        const double angle = static_cast<double>(l % 4) * angle_step;
                        sub_twiddles_aosoa_[offset + l] = static_cast<F>(std::cos(angle * 1));
                        sub_twiddles_aosoa_[offset + width4_vec + l] = static_cast<F>(std::sin(angle * 1));
                        sub_twiddles_aosoa_[offset + width4_vec * 2 + l] = static_cast<F>(std::cos(angle * 2));
                        sub_twiddles_aosoa_[offset + width4_vec * 3 + l] = static_cast<F>(std::sin(angle * 2));
                        sub_twiddles_aosoa_[offset + width4_vec * 4 + l] = static_cast<F>(std::cos(angle * 3));
                        sub_twiddles_aosoa_[offset + width4_vec * 5 + l] = static_cast<F>(std::sin(angle * 3));
                    }
                    offset += 6 * width4_vec;
                    gen_width = gen_width << 2;
                } else if (stage == common::StageType::kRadix4) {
                    size_t num_blocks = std::max<size_t>(1, gen_width / lanes);
                    const double angle_step = -2.0 * std::numbers::pi / static_cast<double>(gen_width << 2);
                    for (size_t b = 0; b < num_blocks; ++b) {
                        for (size_t l = 0; l < lanes; ++l) {
                            const size_t idx = (b * lanes + l) % gen_width;
                            const double angle = static_cast<double>(idx) * angle_step;
                            sub_twiddles_aosoa_[offset + l] = static_cast<F>(std::cos(angle));
                            sub_twiddles_aosoa_[offset + lanes + l] = static_cast<F>(std::sin(angle));
                            sub_twiddles_aosoa_[offset + 2 * lanes + l] = static_cast<F>(std::cos(angle * 2));
                            sub_twiddles_aosoa_[offset + 3 * lanes + l] = static_cast<F>(std::sin(angle * 2));
                            sub_twiddles_aosoa_[offset + 4 * lanes + l] = static_cast<F>(std::cos(angle * 3));
                            sub_twiddles_aosoa_[offset + 5 * lanes + l] = static_cast<F>(std::sin(angle * 3));
                        }
                        offset += 6 * lanes;
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
                            for (int m = 0; m < 7; ++m) {
                                sub_twiddles_aosoa_[offset + 2 * m * lanes + l] = static_cast<F>(std::cos(angle * muls[m]));
                                sub_twiddles_aosoa_[offset + (2 * m + 1) * lanes + l] = static_cast<F>(std::sin(angle * muls[m]));
                            }
                        }
                        offset += 14 * lanes;
                    }
                    gen_width = gen_width << 3;
                }
            }

            size_t n_sq = sub_n_ * sub_n_;
            const double macro_angle_step = -2.0 * std::numbers::pi / static_cast<double>(n_sq);

            macro_twiddles_ = hwy::AllocateAligned<F>(n_sq * 2);

            static constexpr hn::ScalableTag<F> d;
            const size_t quarter_n = sub_n_ >> 2;

            for (size_t r = 0; r < sub_n_; ++r) {
                F* row_twiddles = macro_twiddles_.get() + r * (sub_n_ * 2);
                offset = 0;

                for (size_t i = 0; i < quarter_n; i += lanes) {
                    alignas(64) F t_r[4][lanes];
                    alignas(64) F t_i[4][lanes];

                    for (size_t k = 0; k < lanes; ++k) {
                        size_t c_idx = i + k;
                        double a0 = macro_angle_step * static_cast<double>(r * c_idx);
                        double a1 = macro_angle_step * static_cast<double>(r * (c_idx + quarter_n));
                        double a2 = macro_angle_step * static_cast<double>(r * (c_idx + 2 * quarter_n));
                        double a3 = macro_angle_step * static_cast<double>(r * (c_idx + 3 * quarter_n));

                        t_r[0][k] = static_cast<F>(std::cos(a0)); t_i[0][k] = static_cast<F>(std::sin(a0));
                        t_r[1][k] = static_cast<F>(std::cos(a1)); t_i[1][k] = static_cast<F>(std::sin(a1));
                        t_r[2][k] = static_cast<F>(std::cos(a2)); t_i[2][k] = static_cast<F>(std::sin(a2));
                        t_r[3][k] = static_cast<F>(std::cos(a3)); t_i[3][k] = static_cast<F>(std::sin(a3));
                    }
                    hn::Store(hn::Load(d, t_r[0]), d, row_twiddles + offset);
                    hn::Store(hn::Load(d, t_i[0]), d, row_twiddles + offset + lanes);
                    offset += 2 * lanes;
                    hn::Store(hn::Load(d, t_r[1]), d, row_twiddles + offset);
                    hn::Store(hn::Load(d, t_i[1]), d, row_twiddles + offset + lanes);
                    offset += 2 * lanes;
                    hn::Store(hn::Load(d, t_r[2]), d, row_twiddles + offset);
                    hn::Store(hn::Load(d, t_i[2]), d, row_twiddles + offset + lanes);
                    offset += 2 * lanes;
                    hn::Store(hn::Load(d, t_r[3]), d, row_twiddles + offset);
                    hn::Store(hn::Load(d, t_i[3]), d, row_twiddles + offset + lanes);
                    offset += 2 * lanes;
                }
            }

            if (order_ % 2 != 0) {
                const size_t half_n = (static_cast<size_t>(1) << order_) >> 1;
                radix2_twiddles_ = hwy::AllocateAligned<F>(half_n * 2);
                const double angle_step = -2.0 * std::numbers::pi / static_cast<double>(half_n << 1);

                for (size_t i = 0; i < half_n; i += lanes) {
                    alignas(64) F w_r_arr[lanes];
                    alignas(64) F w_i_arr[lanes];
                    for (size_t k = 0; k < lanes; ++k) {
                        const double angle = static_cast<double>(i + k) * angle_step;
                        w_r_arr[k] = static_cast<F>(std::cos(angle));
                        w_i_arr[k] = static_cast<F>(std::sin(angle));
                    }
                    hn::StoreInterleaved2(hn::Load(d, w_r_arr), hn::Load(d, w_i_arr), d, radix2_twiddles_.get() + 2 * i);
                }
            }

            workspace_ = hwy::AllocateAligned<F>(4 * sub_n_ * padded_sub_n_);
        }

        void forward(std::span<C> in_span, std::span<C> out_span) {
            if (order_ < 10) return;

            auto in_buffer = in_span.data();
            auto out_buffer = out_span.data();
            const size_t n = static_cast<size_t>(1) << order_;

            if (order_ % 2 == 0) {
                process_even_order(in_buffer, out_buffer);
            } else {
                static constexpr hn::ScalableTag<F> d;
                const size_t half_n = n >> 1;
                common::radix2_peel_aos(in_buffer, out_buffer, n, radix2_twiddles_.get());

                process_even_order(out_buffer, out_buffer);
                process_even_order(out_buffer + half_n, out_buffer + half_n);

                auto* workspace_c = reinterpret_cast<std::complex<F>*>(workspace_.get());
                std::memcpy(workspace_c, out_buffer, n * sizeof(std::complex<F>));

                static constexpr hn::ScalableTag<F> tag;
                for (size_t i = 0; i < half_n; i += lanes) {
                    hn::Vec<decltype(tag)> r0, i0, r1, i1;
                    hn::LoadInterleaved2(tag, reinterpret_cast<const F*>(workspace_c + i), r0, i0);
                    hn::LoadInterleaved2(tag, reinterpret_cast<const F*>(workspace_c + i + half_n), r1, i1);

                    const auto r_lo = hn::InterleaveLower(tag, r0, r1);
                    const auto i_lo = hn::InterleaveLower(tag, i0, i1);

                    const auto r_hi = hn::InterleaveUpper(tag, r0, r1);
                    const auto i_hi = hn::InterleaveUpper(tag, i0, i1);

                    F* out_ptr = reinterpret_cast<F*>(out_buffer + 2 * i);
                    hn::StoreInterleaved2(r_lo, i_lo, tag, out_ptr);
                    hn::StoreInterleaved2(r_hi, i_hi, tag, out_ptr + 2 * lanes);
                }
            }
        }

    private:
        static constexpr size_t lanes = hn::Lanes(hn::ScalableTag<F>());
        static constexpr size_t width4_vec = std::max(static_cast<size_t>(4), lanes);

        size_t order_;
        size_t sub_order_;
        size_t sub_n_;
        size_t padded_sub_n_;

        std::vector<common::StageType> stages_;
        hwy::AlignedFreeUniquePtr<F[]> sub_twiddles_aosoa_;
        hwy::AlignedFreeUniquePtr<F[]> macro_twiddles_;
        hwy::AlignedFreeUniquePtr<F[]> workspace_;
        hwy::AlignedFreeUniquePtr<F[]> radix2_twiddles_;

        void process_even_order(const std::complex<F>* in, std::complex<F>* out) {
            static constexpr hn::ScalableTag<F> d;
            const size_t n1 = sub_n_;
            const size_t n_sq = sub_n_ * sub_n_;

            F* __restrict in_aosoa = workspace_.get();
            F* __restrict out_aosoa = workspace_.get() + ((sub_n_ * padded_sub_n_) << 1);

            common::transpose_aos_to_aosoa(in, in_aosoa, n1, padded_sub_n_);

            for (size_t r = 0; r < n1; ++r) {
                F* row_in = in_aosoa + r * (padded_sub_n_ << 1);
                F* row_out = out_aosoa + r * (padded_sub_n_ << 1);

                F* temp_in = row_in;
                F* temp_out = row_out;

                const F* w_ptr = sub_twiddles_aosoa_.get();
                if (stages_[0] == common::StageType::kRadix4FirstPass) {
                    common::radix4_first_pass_aosoa(temp_in, temp_out, n1);
                } else {
                    common::radix8_first_pass_aosoa(temp_in, temp_out, n1);
                }

                size_t width = (stages_[0] == common::StageType::kRadix4FirstPass) ? 4 : 8;
                for (size_t i = 1; i < stages_.size() - 1; ++i) {
                    std::swap(temp_in, temp_out);
                    const auto stage = stages_[i];
                    if (stage == common::StageType::kRadix4Width4) {
                        common::radix4_width4_aosoa(temp_in, temp_out, n1, w_ptr);
                        w_ptr += 6 * width4_vec;
                        width = width << 2;
                    } else if (stage == common::StageType::kRadix4) {
                        common::radix4_aosoa(temp_in, temp_out, n1, width, w_ptr);
                        const size_t num_blocks = std::max<size_t>(1, width / lanes);
                        w_ptr += num_blocks * 6 * lanes;
                        width = width << 2;
                    } else if (stage == common::StageType::kRadix8) {
                        common::radix8_aosoa(temp_in, temp_out, n1, width, w_ptr);
                        const size_t num_blocks = std::max<size_t>(1, width / lanes);
                        w_ptr += num_blocks * 14 * lanes;
                        width = width << 3;
                    }
                }
                std::swap(temp_in, temp_out);
                const F* row_macro_tw = macro_twiddles_.get() + r * (sub_n_ * 2);
                common::radix4_aosoa_macrotwiddle<F, true>(temp_in, temp_out, n1, width, w_ptr, row_macro_tw);
            }

            const bool stages_is_even = (stages_.size() % 2 == 0);

            F* const row_result = stages_is_even ? in_aosoa : out_aosoa;
            F* const trans_out  = stages_is_even ? out_aosoa : in_aosoa;
            common::transpose_aosoa_square_pure(row_result, trans_out, n1, padded_sub_n_);

            for (size_t r = 0; r < n1; ++r) {
                F* temp_in = trans_out + r * (padded_sub_n_ << 1);
                F* temp_out = row_result + r * (padded_sub_n_ << 1);

                const F* w_ptr = sub_twiddles_aosoa_.get();
                if (stages_[0] == common::StageType::kRadix4FirstPass) {
                    common::radix4_first_pass_aosoa(temp_in, temp_out, n1);
                } else {
                    common::radix8_first_pass_aosoa(temp_in, temp_out, n1);
                }

                size_t width = (stages_[0] == common::StageType::kRadix4FirstPass) ? 4 : 8;
                for (size_t i = 1; i < stages_.size(); ++i) {
                    std::swap(temp_in, temp_out);
                    const auto stage = stages_[i];
                    if (stage == common::StageType::kRadix4Width4) {
                        common::radix4_width4_aosoa(temp_in, temp_out, n1, w_ptr);
                        w_ptr += 6 * width4_vec;
                        width = width << 2;
                    } else if (stage == common::StageType::kRadix4) {
                        common::radix4_aosoa(temp_in, temp_out, n1, width, w_ptr);
                        const size_t num_blocks = std::max<size_t>(1, width / lanes);
                        w_ptr += num_blocks * 6 * lanes;
                        width = width << 2;
                    } else if (stage == common::StageType::kRadix8) {
                        common::radix8_aosoa(temp_in, temp_out, n1, width, w_ptr);
                        const size_t num_blocks = std::max<size_t>(1, width / lanes);
                        w_ptr += num_blocks * 14 * lanes;
                        width = width << 3;
                    }
                }
            }

            F* const final_result = stages_is_even ? trans_out : row_result;
            common::transpose_aosoa_to_aos(final_result, out, n1, padded_sub_n_);
        }
    };
}
