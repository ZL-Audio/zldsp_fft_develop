#pragma once

#include "zlfft_common_aosoa.hpp"
#include <algorithm>

namespace zlfft {
    namespace hn = hwy::HWY_NAMESPACE;

    template <typename F>
    class SIMDLowOrderAOSOA2 {
        using C = std::complex<F>;

    private:
        std::vector<common::StageType> stages_;

    public:
        explicit SIMDLowOrderAOSOA2(const size_t order) :
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
                const auto mod_result = order % 3;
                stages_.emplace_back(common::StageType::kRadix8FirstPass);
                if (mod_result == 1) {
                    for (size_t i = 3; i < order - 4; i += 3) {
                        stages_.emplace_back(common::StageType::kRadix8);
                    }
                    stages_.emplace_back(common::StageType::kRadix4);
                } else if (mod_result == 2) {
                    for (size_t i = 3; i < order - 2; i += 3) {
                        stages_.emplace_back(common::StageType::kRadix8);
                    }
                } else {
                    for (size_t i = 3; i < order - 6; i += 3) {
                        stages_.emplace_back(common::StageType::kRadix8);
                    }
                    stages_.emplace_back(common::StageType::kRadix4);
                    stages_.emplace_back(common::StageType::kRadix4);
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

                twiddles_aosoa_ = hwy::AllocateAligned<F>(num_twiddle_elements);
                size_t offset = 0;
                size_t gen_width = (stages_[0] == common::StageType::kRadix4FirstPass) ? 4 : 8;

                for (size_t i = 1; i < stages_.size(); ++i) {
                    const auto stage = stages_[i];
                    if (stage == common::StageType::kRadix4Width4) {
                        const double angle_step = -2.0 * std::numbers::pi / static_cast<double>(gen_width << 2);
                        for (size_t l = 0; l < width4_vec; ++l) {
                            const double angle = static_cast<double>(l % 4) * angle_step;
                            twiddles_aosoa_[offset + l] = static_cast<F>(std::cos(angle * 1));
                            twiddles_aosoa_[offset + width4_vec + l] = static_cast<F>(std::sin(angle * 1));
                            twiddles_aosoa_[offset + width4_vec * 2 + l] = static_cast<F>(std::cos(angle * 2));
                            twiddles_aosoa_[offset + width4_vec * 3 + l] = static_cast<F>(std::sin(angle * 2));
                            twiddles_aosoa_[offset + width4_vec * 4 + l] = static_cast<F>(std::cos(angle * 3));
                            twiddles_aosoa_[offset + width4_vec * 5 + l] = static_cast<F>(std::sin(angle * 3));
                        }
                        offset += 6 * width4_vec;
                        gen_width = gen_width << 2;
                    } else if (stage == common::StageType::kRadix4 || stage == common::StageType::kRadix4LastPass) {
                        size_t num_blocks = std::max<size_t>(1, gen_width / lanes);
                        const double angle_step = -2.0 * std::numbers::pi / static_cast<double>(gen_width << 2);

                        for (size_t b = 0; b < num_blocks; ++b) {
                            for (size_t l = 0; l < lanes; ++l) {
                                const size_t idx = (b * lanes + l) % gen_width;
                                const double angle = static_cast<double>(idx) * angle_step;
                                twiddles_aosoa_[offset + l] = static_cast<F>(std::cos(angle * 1));
                                twiddles_aosoa_[offset + lanes + l] = static_cast<F>(std::sin(angle * 1));
                                twiddles_aosoa_[offset + 2 * lanes + l] = static_cast<F>(std::cos(angle * 2));
                                twiddles_aosoa_[offset + 3 * lanes + l] = static_cast<F>(std::sin(angle * 2));
                                twiddles_aosoa_[offset + 4 * lanes + l] = static_cast<F>(std::cos(angle * 3));
                                twiddles_aosoa_[offset + 5 * lanes + l] = static_cast<F>(std::sin(angle * 3));
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
                                static constexpr int muls[7] = {4, 2, 6, 1, 5, 3, 7};
                                for (int m = 0; m < 7; ++m) {
                                    twiddles_aosoa_[offset + 2 * m * lanes + l] = static_cast<F>(std::cos(
                                        angle * muls[m]));
                                    twiddles_aosoa_[offset + (2 * m + 1) * lanes + l] = static_cast<F>(std::sin(
                                        angle * muls[m]));
                                }
                            }
                            offset += 14 * lanes;
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
                    w_ptr += 6 * width4_vec;
                    width = width << 2;
                    break;
                }
                case common::StageType::kRadix4: {
                    common::radix4_aosoa(in_aosoa, out_aosoa, n, width, w_ptr);
                    const size_t num_blocks = std::max<size_t>(1, width / lanes);
                    w_ptr += num_blocks * 6 * lanes;
                    width = width << 2;
                    break;
                }
                case common::StageType::kRadix8: {
                    common::radix8_aosoa(in_aosoa, out_aosoa, n, width, w_ptr);
                    const size_t num_blocks = std::max<size_t>(1, width / lanes);
                    w_ptr += num_blocks * 14 * lanes;
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

        size_t num_stages() const { return stages_.size(); }

        void forward_stages(std::span<C> in_buffer, std::span<C> out_buffer, size_t num_stages_to_run) {
            if (order_ <= 5 || num_stages_to_run == stages_.size()) {
                forward(in_buffer, out_buffer);
                return;
            }

            const auto n = in_buffer.size();
            num_stages_to_run = std::min(num_stages_to_run, stages_.size());

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

            for (size_t i = 1; i < num_stages_to_run; ++i) {
                const auto stage = stages_[i];
                switch (stage) {
                case common::StageType::kRadix4Width4: {
                    common::radix4_width4_aosoa(in_aosoa, out_aosoa, n, w_ptr);
                    w_ptr += 6 * width4_vec;
                    width = width << 2;
                    break;
                }
                case common::StageType::kRadix4: {
                    common::radix4_aosoa(in_aosoa, out_aosoa, n, width, w_ptr);
                    const size_t num_blocks = std::max<size_t>(1, width / lanes);
                    w_ptr += num_blocks * 6 * lanes;
                    width = width << 2;
                    break;
                }
                case common::StageType::kRadix8: {
                    common::radix8_aosoa(in_aosoa, out_aosoa, n, width, w_ptr);
                    const size_t num_blocks = std::max<size_t>(1, width / lanes);
                    w_ptr += num_blocks * 14 * lanes;
                    width = width << 3;
                    break;
                }
                default:
                    break;
                }
                std::swap(in_aosoa, out_aosoa);
            }
        }

    private:
        static constexpr size_t lanes = hn::Lanes(hn::ScalableTag<F>());
        static constexpr size_t width4_vec = std::max(static_cast<size_t>(4), lanes);

        size_t order_;
        size_t stride_;
        hwy::AlignedFreeUniquePtr<F[]> twiddles_r_;
        hwy::AlignedFreeUniquePtr<F[]> twiddles_i_;
        hwy::AlignedFreeUniquePtr<F[]> twiddles_aosoa_;
        hwy::AlignedFreeUniquePtr<F[]> workspace_;
    };
}
