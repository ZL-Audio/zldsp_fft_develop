#pragma once

#include "zldsp_fft_common.hpp"
#include <algorithm>

namespace zldsp::fft {
    namespace hn = hwy::HWY_NAMESPACE;

    template <typename F>
    struct SplitSpan {
        std::span<F> real;
        std::span<F> imag;
        [[nodiscard]] size_t size() const { return real.size(); }
    };

    template <typename F>
    class CFFT {
        using C = std::complex<F>;

    private:
        size_t order_;
        size_t stride_;
        hwy::AlignedFreeUniquePtr<F[]> workspace_;
        hwy::AlignedFreeUniquePtr<F[]> twiddles_r_;
        hwy::AlignedFreeUniquePtr<F[]> twiddles_i_;
        hwy::AlignedFreeUniquePtr<F[]> twiddles_aosoa_;
        std::vector<size_t> twiddles_shift_;
        std::vector<common::StageType> stages_;

    public:
        explicit CFFT(const size_t order) :
            order_(order) {
            if (order < 4) {
                return;
            }
            if (order < 6) {
                common::generate_order_4_5_twiddles(order, twiddles_r_, twiddles_i_);
                return;
            }
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

            twiddles_shift_.resize(stages_.size());
            twiddles_shift_[0] = 0;

            common::generate_general_twiddles(stages_, twiddles_shift_, twiddles_aosoa_);

            const auto n = static_cast<size_t>(1) << order;
            const auto pad = (64 / sizeof(F)) + 16;
            stride_ = n + pad;
            workspace_ = hwy::AllocateAligned<F>(4 * stride_);
        }

        void forward(std::span<C> in_buffer, std::span<C> out_buffer) {
            execute<true>(in_buffer, out_buffer);
        }

        void backward(std::span<C> in_buffer, std::span<C> out_buffer) {
            execute<false>(in_buffer, out_buffer);
        }

    private:
        template <bool is_forward>
        void execute(std::span<C> in_buffer, std::span<C> out_buffer) {
            F* in = reinterpret_cast<F*>(in_buffer.data());
            F* out = reinterpret_cast<F*>(out_buffer.data());
            switch (order_) {
            case 0:
                common::callback_order_0<is_forward, F>(common::AoSPtr<F>{in}, common::AoSPtr<F>{out});
                return;
            case 1:
                common::callback_order_1<is_forward, F>(common::AoSPtr<F>{in}, common::AoSPtr<F>{out});
                return;
            case 2:
                common::callback_order_2<is_forward, F>(common::AoSPtr<F>{in}, common::AoSPtr<F>{out});
                return;
            case 3:
                common::callback_order_3<is_forward, F>(common::AoSPtr<F>{in}, common::AoSPtr<F>{out});
                return;
            case 4:
                common::callback_order_4<is_forward, F>(common::AoSPtr<F>{in}, common::AoSPtr<F>{out},
                                                        twiddles_r_.get(), twiddles_i_.get());
                return;
            case 5:
                common::callback_order_5<is_forward, F>(common::AoSPtr<F>{in}, common::AoSPtr<F>{out},
                                                        twiddles_r_.get(), twiddles_i_.get());
                return;
            case 6: {
                F* __restrict in_aosoa = workspace_.get();
                F* __restrict out_aosoa = workspace_.get() + 2 * stride_;
                const F* __restrict w0 = twiddles_aosoa_.get();
                common::radix4_first_pass_fused_aosoa<is_forward>(common::AoSPtr<F>{in}, out_aosoa, 64);
                common::radix4_width4_aosoa(out_aosoa, in_aosoa, 64, w0);
                const F* __restrict w1 = w0 + twiddles_shift_[1];
                common::radix4_last_pass_fused_aosoa<is_forward>(in_aosoa, common::AoSPtr<F>{out}, 64, 16, w1);
                return;
            }
            case 7: {
                F* __restrict in_aosoa = workspace_.get();
                F* __restrict out_aosoa = workspace_.get() + 2 * stride_;
                const F* __restrict w0 = twiddles_aosoa_.get();
                common::radix8_first_pass_fused_aosoa<is_forward>(common::AoSPtr<F>{in}, out_aosoa, 128);
                common::radix4_aosoa(out_aosoa, in_aosoa, 128, 8, w0);
                const F* __restrict w1 = w0 + twiddles_shift_[1];
                common::radix4_last_pass_fused_aosoa<is_forward>(in_aosoa, common::AoSPtr<F>{out}, 128, 32, w1);
                return;
            }
            case 8: {
                F* __restrict in_aosoa = workspace_.get();
                F* __restrict out_aosoa = workspace_.get() + 2 * stride_;
                const F* __restrict w0 = twiddles_aosoa_.get();
                common::radix4_first_pass_fused_aosoa<is_forward>(common::AoSPtr<F>{in}, out_aosoa, 256);
                common::radix4_width4_aosoa(out_aosoa, in_aosoa, 256, w0);
                const F* __restrict w1 = w0 + twiddles_shift_[1];
                common::radix4_aosoa(in_aosoa, out_aosoa, 256, 16, w1);
                const F* __restrict w2 = w1 + twiddles_shift_[2];
                common::radix4_last_pass_fused_aosoa<is_forward>(out_aosoa, common::AoSPtr<F>{out}, 256, 64, w2);
                return;
            }
            case 9: {
                F* __restrict in_aosoa = workspace_.get();
                F* __restrict out_aosoa = workspace_.get() + 2 * stride_;
                const F* __restrict w0 = twiddles_aosoa_.get();
                common::radix8_first_pass_fused_aosoa<is_forward>(common::AoSPtr<F>{in}, out_aosoa, 512);
                common::radix4_aosoa(out_aosoa, in_aosoa, 512, 8, w0);
                const F* __restrict w1 = w0 + twiddles_shift_[1];
                common::radix4_aosoa(in_aosoa, out_aosoa, 512, 32, w1);
                const F* __restrict w2 = w1 + twiddles_shift_[2];
                common::radix4_last_pass_fused_aosoa<is_forward>(out_aosoa, common::AoSPtr<F>{out}, 512, 128, w2);
                return;
            }
            case 10: {
                F* __restrict in_aosoa = workspace_.get();
                F* __restrict out_aosoa = workspace_.get() + 2 * stride_;
                const F* __restrict w0 = twiddles_aosoa_.get();
                common::radix4_first_pass_fused_aosoa<is_forward>(common::AoSPtr<F>{in}, out_aosoa, 1024);
                common::radix4_width4_aosoa(out_aosoa, in_aosoa, 1024, w0);
                const F* __restrict w1 = w0 + twiddles_shift_[1];
                common::radix4_aosoa(in_aosoa, out_aosoa, 1024, 16, w1);
                const F* __restrict w2 = w1 + twiddles_shift_[2];
                common::radix4_aosoa(out_aosoa, in_aosoa, 1024, 64, w2);
                const F* __restrict w3 = w2 + twiddles_shift_[3];
                common::radix4_last_pass_fused_aosoa<is_forward>(in_aosoa, common::AoSPtr<F>{out}, 1024, 256, w3);
                return;
            }
            default:
                break;
            }

            const auto n = in_buffer.size();

            F* __restrict in_aosoa = workspace_.get();
            F* __restrict out_aosoa = workspace_.get() + 2 * stride_;

            const F* __restrict w_ptr = twiddles_aosoa_.get();

            if (stages_[0] == common::StageType::kRadix4FirstPass) {
                common::radix4_first_pass_fused_aosoa<is_forward>(common::AoSPtr<F>{in}, out_aosoa, n);
            } else {
                common::radix8_first_pass_fused_aosoa<is_forward>(common::AoSPtr<F>{in}, out_aosoa, n);
            }

            std::swap(in_aosoa, out_aosoa);

            size_t width = (stages_[0] == common::StageType::kRadix4FirstPass) ? 4 : 8;
            for (size_t i = 1; i < stages_.size() - 1; ++i) {
                switch (stages_[i]) {
                case common::StageType::kRadix4Width4: {
                    common::radix4_width4_aosoa(in_aosoa, out_aosoa, n, w_ptr);
                    width = width << 2;
                    break;
                }
                case common::StageType::kRadix4: {
                    common::radix4_aosoa(in_aosoa, out_aosoa, n, width, w_ptr);
                    width = width << 2;
                    break;
                }
                case common::StageType::kRadix8: {
                    common::radix8_aosoa(in_aosoa, out_aosoa, n, width, w_ptr);
                    width = width << 3;
                    break;
                }
                default:
                    break;
                }
                w_ptr += twiddles_shift_[i];
                std::swap(in_aosoa, out_aosoa);
            }
            common::radix4_last_pass_fused_aosoa<is_forward>(in_aosoa, common::AoSPtr<F>{out}, n, width, w_ptr);
        }
    };
}
