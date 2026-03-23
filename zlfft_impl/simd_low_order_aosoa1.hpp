#pragma once

#include "zlfft_common_aosoa.hpp"
#include <algorithm>

namespace zlfft {
    namespace hn = hwy::HWY_NAMESPACE;

    template <typename F>
    class SIMDLowOrderAOSOA1 {
        using C = std::complex<F>;

    private:
        std::vector<common::StageType> stages_;

    public:
        explicit SIMDLowOrderAOSOA1(const size_t order) :
            order_(order) {
            if (order < 4) {
                return;
            }
            if (order == 4) {
                stages_ = {common::StageType::kRadix4FirstPass, common::StageType::kRadix4LastPass};
            } else if (order == 5) {
                stages_ = {common::StageType::kRadix8FirstPass, common::StageType::kRadix4LastPass};
            } else if (order == 6) {
                stages_ = {common::StageType::kRadix4FirstPass, common::StageType::kRadix4Width4,
                           common::StageType::kRadix4LastPass};
            } else {
                const auto mod_result = order % 2;
                if (mod_result == 1) {
                    stages_.emplace_back(common::StageType::kRadix8FirstPass);
                    for (size_t i = 3; i < order - 2; i += 2) {
                        stages_.emplace_back(common::StageType::kRadix4);
                    }
                } else {
                    stages_.emplace_back(common::StageType::kRadix4FirstPass);
                    stages_.emplace_back(common::StageType::kRadix4Width4);
                    for (size_t i = 4; i < order - 2; i += 2) {
                        stages_.emplace_back(common::StageType::kRadix4);
                    }
                }
                stages_.emplace_back(common::StageType::kRadix4LastPass);
            }

            if (order <= 5) {
                size_t num_twiddles = 0;
                size_t width = (stages_[0] == common::StageType::kRadix4FirstPass) ? 4 : 8;
                for (const auto stage : stages_) {
                    num_twiddles += 3 * width;
                    width = width << 2;
                }

                twiddles_r_ = hwy::AllocateAligned<F>(num_twiddles);
                twiddles_i_ = hwy::AllocateAligned<F>(num_twiddles);

                size_t offset = 0;
                width = (stages_[0] == common::StageType::kRadix4FirstPass) ? 4 : 8;
                for (const auto stage : stages_) {
                    const double angle_step = -2.0 * std::numbers::pi / static_cast<double>(width << 2);
                    for (int mul = 1; mul < 4; ++mul) {
                        const auto step = angle_step * static_cast<double>(mul);
                        for (size_t k = 0; k < width; ++k, ++offset) {
                            const double angle = static_cast<double>(k) * step;
                            twiddles_r_[offset] = static_cast<F>(std::cos(angle));
                            twiddles_i_[offset] = static_cast<F>(std::sin(angle));
                        }
                    }
                    width = width << 2;
                }
            } else {
                size_t num_twiddle_elements = 0;
                size_t sim_width = (stages_[0] == common::StageType::kRadix4FirstPass) ? 4 : 8;

                for (size_t i = 1; i < stages_.size(); ++i) {
                    const auto stage = stages_[i];
                    if (stage == common::StageType::kRadix4Width4) {
                        num_twiddle_elements += 6 * std::max(static_cast<size_t>(4), vlanes);
                        sim_width = sim_width << 2;
                    } else if (stage == common::StageType::kRadix4 || stage == common::StageType::kRadix4LastPass) {
                        size_t num_blocks = std::max<size_t>(1, sim_width / vlanes);
                        num_twiddle_elements += num_blocks * 6 * vlanes;
                        sim_width = sim_width << 2;
                    } else if (stage == common::StageType::kRadix8) {
                        size_t num_blocks = std::max<size_t>(1, sim_width / vlanes);
                        num_twiddle_elements += num_blocks * 14 * vlanes;
                        sim_width = sim_width << 3;
                    }
                }

                twiddles_aosoa_ = hwy::AllocateAligned<F>(num_twiddle_elements);
                size_t offset = 0;
                size_t gen_width = (stages_[0] == common::StageType::kRadix4FirstPass) ? 4 : 8;

                for (size_t i = 1; i < stages_.size(); ++i) {
                    const auto stage = stages_[i];
                    if (stage == common::StageType::kRadix4Width4) {
                        const double angle_step = -2.0 * std::numbers::pi / static_cast<double>(gen_width << 2);

                        for (int l = 0; l < std::max(static_cast<size_t>(4), vlanes); ++l) {
                            const double angle = static_cast<double>(l) * angle_step;
                            twiddles_aosoa_[offset + l]      = static_cast<F>(std::cos(angle * 1));
                            twiddles_aosoa_[offset + 4 + l]  = static_cast<F>(std::sin(angle * 1));
                            twiddles_aosoa_[offset + 8 + l]  = static_cast<F>(std::cos(angle * 2));
                            twiddles_aosoa_[offset + 12 + l] = static_cast<F>(std::sin(angle * 2));
                            twiddles_aosoa_[offset + 16 + l] = static_cast<F>(std::cos(angle * 3));
                            twiddles_aosoa_[offset + 20 + l] = static_cast<F>(std::sin(angle * 3));
                        }
                        offset += 24;
                        gen_width = gen_width << 2;
                    } else if (stage == common::StageType::kRadix4 || stage == common::StageType::kRadix4LastPass) {
                        size_t num_blocks = std::max<size_t>(1, gen_width / vlanes);
                        const double angle_step = -2.0 * std::numbers::pi / static_cast<double>(gen_width << 2);

                        for (size_t b = 0; b < num_blocks; ++b) {
                            for (size_t l = 0; l < vlanes; ++l) {
                                const size_t idx = (b * vlanes + l) % gen_width;
                                const double angle = static_cast<double>(idx) * angle_step;
                                twiddles_aosoa_[offset + l]              = static_cast<F>(std::cos(angle * 1));
                                twiddles_aosoa_[offset + vlanes + l]     = static_cast<F>(std::sin(angle * 1));
                                twiddles_aosoa_[offset + 2 * vlanes + l] = static_cast<F>(std::cos(angle * 2));
                                twiddles_aosoa_[offset + 3 * vlanes + l] = static_cast<F>(std::sin(angle * 2));
                                twiddles_aosoa_[offset + 4 * vlanes + l] = static_cast<F>(std::cos(angle * 3));
                                twiddles_aosoa_[offset + 5 * vlanes + l] = static_cast<F>(std::sin(angle * 3));
                            }
                            offset += 6 * vlanes;
                        }
                        gen_width = gen_width << 2;
                    } else if (stage == common::StageType::kRadix8) {
                        size_t num_blocks = std::max<size_t>(1, gen_width / vlanes);
                        const double angle_step = -2.0 * std::numbers::pi / static_cast<double>(gen_width << 3);

                        for (size_t b = 0; b < num_blocks; ++b) {
                            for (size_t l = 0; l < vlanes; ++l) {
                                const size_t idx = (b * vlanes + l) % gen_width;
                                const double angle = static_cast<double>(idx) * angle_step;
                                const int muls[7] = {3, 1, 5, 0, 4, 2, 6};
                                for (int m = 0; m < 7; ++m) {
                                    twiddles_aosoa_[offset + 2 * m * vlanes + l]       = static_cast<F>(std::cos(angle * muls[m]));
                                    twiddles_aosoa_[offset + (2 * m + 1) * vlanes + l] = static_cast<F>(std::sin(angle * muls[m]));
                                }
                            }
                            offset += 14 * vlanes;
                        }
                        gen_width = gen_width << 3;
                    }
                }
            }

            const auto n = static_cast<size_t>(1) << order;
            const auto pad = (64 / sizeof(F)) + 16;
            stride_ = n + pad;
            workspace_ = hwy::AllocateAligned<F>(4 * stride_);
        }

        void forward(std::span<C> in_buffer, std::span<C> out_buffer) {
            switch (order_) {
            case 0:
                common::callback_order_0(in_buffer.data(), out_buffer.data());
                return;
            case 1:
                common::callback_order_1(in_buffer.data(), out_buffer.data());
                return;
            case 2:
                common::callback_order_2(in_buffer.data(), out_buffer.data());
                return;
            case 3:
                common::callback_order_3(in_buffer.data(), out_buffer.data());
                return;
            case 4:
                common::callback_order_4(in_buffer.data(), out_buffer.data(),
                                         twiddles_r_.get(), twiddles_i_.get());
                return;
            case 5:
                common::callback_order_5(in_buffer.data(), out_buffer.data(),
                                         twiddles_r_.get(), twiddles_i_.get());
                return;
            default:
                break;
            }

            const auto n = in_buffer.size();

            F* __restrict in_aosoa = workspace_.get();
            F* __restrict out_aosoa = workspace_.get() + 2 * stride_;

            const F* __restrict w_ptr = twiddles_aosoa_.get();

            if (stages_[0] == common::StageType::kRadix4FirstPass) {
                common::radix4_first_pass_fused_aosoa(in_buffer.data(), out_aosoa, n);
            } else {
                common::radix8_first_pass_fused_aosoa(in_buffer.data(), out_aosoa, n);
            }

            std::swap(in_aosoa, out_aosoa);

            size_t width = (stages_[0] == common::StageType::kRadix4FirstPass) ? 4 : 8;
            for (size_t i = 1; i < stages_.size() - 1; ++i) {
                const auto stage = stages_[i];
                switch (stage) {
                case common::StageType::kRadix4Width4: {
                    common::radix4_width4_aosoa(in_aosoa, out_aosoa, n, w_ptr);
                    w_ptr += 24;
                    width = width << 2;
                    break;
                }
                case common::StageType::kRadix4: {
                    common::radix4_aosoa(in_aosoa, out_aosoa, n, width, w_ptr);
                    const size_t vlanes = hn::Lanes(hn::ScalableTag<F>());
                    const size_t num_blocks = std::max<size_t>(1, width / vlanes);
                    w_ptr += num_blocks * 6 * vlanes;
                    width = width << 2;
                    break;
                }
                case common::StageType::kRadix8: {
                    common::radix8_aosoa(in_aosoa, out_aosoa, n, width, w_ptr);
                    const size_t vlanes = hn::Lanes(hn::ScalableTag<F>());
                    const size_t num_blocks = std::max<size_t>(1, width / vlanes);
                    w_ptr += num_blocks * 14 * vlanes;
                    width = width << 3;
                    break;
                }
                default:
                    break;
                }
                std::swap(in_aosoa, out_aosoa);
            }

            common::radix4_last_pass_fused_aosoa(in_aosoa, out_buffer.data(), n, width, w_ptr);
        }

    private:
        size_t order_;
        size_t stride_;
        hwy::AlignedFreeUniquePtr<F[]> twiddles_r_;
        hwy::AlignedFreeUniquePtr<F[]> twiddles_i_;
        hwy::AlignedFreeUniquePtr<F[]> twiddles_aosoa_;
        hwy::AlignedFreeUniquePtr<F[]> workspace_;
    };
}