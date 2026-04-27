#pragma once

#include "zldsp_fft_common.hpp"
#include <algorithm>
#include <array>

namespace zldsp::fft {
    namespace hn = hwy::HWY_NAMESPACE;

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
            if (order < 6) {
                if (order < 4) {
                    return;
                }
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

        /**
         * AoS to AoS forward
         * @param in_buffer
         * @param out_buffer
         */
        void forward(std::span<C> in_buffer, std::span<C> out_buffer) {
            execute<true>(make_aos(in_buffer), make_aos(out_buffer), in_buffer.size());
        }

        /**
         * AoS to AoS backward
         * @param in_buffer
         * @param out_buffer
         */
        void backward(std::span<C> in_buffer, std::span<C> out_buffer) {
            execute<false>(make_aos(in_buffer), make_aos(out_buffer), in_buffer.size());
        }

        /**
         * AoS to SoA forward
         * @param in_buffer
         * @param out_buffer
         */
        void forward(std::span<C> in_buffer, std::array<std::span<F>, 2> out_buffer) {
            execute<true>(make_aos(in_buffer), make_soa(out_buffer), in_buffer.size());
        }

        /**
         * AoS to SoA backward
         * @param in_buffer
         * @param out_buffer
         */
        void backward(std::span<C> in_buffer, std::array<std::span<F>, 2> out_buffer) {
            execute<false>(make_aos(in_buffer), make_soa(out_buffer), in_buffer.size());
        }

        /**
         * SoA to AoS forward
         * @param in_buffer
         * @param out_buffer
         */
        void forward(std::array<std::span<F>, 2> in_buffer, std::span<C> out_buffer) {
            execute<true>(make_soa(in_buffer), make_aos(out_buffer), in_buffer[0].size());
        }

        /**
         * SoA to AoS backward
         * @param in_buffer
         * @param out_buffer
         */
        void backward(std::array<std::span<F>, 2> in_buffer, std::span<C> out_buffer) {
            execute<false>(make_soa(in_buffer), make_aos(out_buffer), in_buffer[0].size());
        }

        /**
         * SoA to SoA forward
         * @param in_buffer
         * @param out_buffer
         */
        void forward(std::array<std::span<F>, 2> in_buffer, std::array<std::span<F>, 2> out_buffer) {
            execute<true>(make_soa(in_buffer), make_soa(out_buffer), in_buffer[0].size());
        }

        /**
         * SoA to SoA backward
         * @param in_buffer
         * @param out_buffer
         */
        void backward(std::array<std::span<F>, 2> in_buffer, std::array<std::span<F>, 2> out_buffer) {
            execute<false>(make_soa(in_buffer), make_soa(out_buffer), in_buffer[0].size());
        }

    private:
        static common::AoSPtr<F> make_aos(std::span<C> span) {
            return common::AoSPtr<F>{reinterpret_cast<F*>(span.data())};
        }

        static common::SoAPtr<F> make_soa(std::array<std::span<F>, 2> span) {
            return common::SoAPtr<F>{span[0].data(), span[1].data()};
        }

        template <bool is_forward, typename InPtr, typename OutPtr>
        void execute(InPtr in_ptr, OutPtr out_ptr, const size_t n) {
            switch (order_) {
            case 0:
                common::callback_order_0<is_forward, F>(in_ptr, out_ptr);
                return;
            case 1:
                common::callback_order_1<is_forward, F>(in_ptr, out_ptr);
                return;
            case 2:
                common::callback_order_2<is_forward, F>(in_ptr, out_ptr);
                return;
            case 3:
                common::callback_order_3<is_forward, F>(in_ptr, out_ptr);
                return;
            case 4:
                common::callback_order_4<is_forward, F>(in_ptr, out_ptr, twiddles_r_.get(), twiddles_i_.get());
                return;
            case 5:
                common::callback_order_5<is_forward, F>(in_ptr, out_ptr, twiddles_r_.get(), twiddles_i_.get());
                return;
            case 6: {
                F* HWY_RESTRICT in_aosoa = workspace_.get();
                F* HWY_RESTRICT out_aosoa = workspace_.get() + 2 * stride_;
                const F* HWY_RESTRICT w0 = twiddles_aosoa_.get();
                common::radix4_first_pass_fused_aosoa<is_forward>(in_ptr, out_aosoa, 64);
                common::radix4_width4_aosoa(out_aosoa, in_aosoa, 64, w0);
                const F* HWY_RESTRICT w1 = w0 + twiddles_shift_[1];
                common::radix4_last_pass_fused_aosoa<is_forward>(in_aosoa, out_ptr, 64, 16, w1);
                return;
            }
            case 7: {
                F* HWY_RESTRICT in_aosoa = workspace_.get();
                F* HWY_RESTRICT out_aosoa = workspace_.get() + 2 * stride_;
                const F* HWY_RESTRICT w0 = twiddles_aosoa_.get();
                common::radix8_first_pass_fused_aosoa<is_forward>(in_ptr, out_aosoa, 128);
                common::radix4_aosoa(out_aosoa, in_aosoa, 128, 8, w0);
                const F* HWY_RESTRICT w1 = w0 + twiddles_shift_[1];
                common::radix4_last_pass_fused_aosoa<is_forward>(in_aosoa, out_ptr, 128, 32, w1);
                return;
            }
            case 8: {
                F* HWY_RESTRICT in_aosoa = workspace_.get();
                F* HWY_RESTRICT out_aosoa = workspace_.get() + 2 * stride_;
                const F* HWY_RESTRICT w0 = twiddles_aosoa_.get();
                common::radix4_first_pass_fused_aosoa<is_forward>(in_ptr, out_aosoa, 256);
                common::radix4_width4_aosoa(out_aosoa, in_aosoa, 256, w0);
                const F* HWY_RESTRICT w1 = w0 + twiddles_shift_[1];
                common::radix4_aosoa(in_aosoa, out_aosoa, 256, 16, w1);
                const F* HWY_RESTRICT w2 = w1 + twiddles_shift_[2];
                common::radix4_last_pass_fused_aosoa<is_forward>(out_aosoa, out_ptr, 256, 64, w2);
                return;
            }
            case 9: {
                F* HWY_RESTRICT in_aosoa = workspace_.get();
                F* HWY_RESTRICT out_aosoa = workspace_.get() + 2 * stride_;
                const F* HWY_RESTRICT w0 = twiddles_aosoa_.get();
                common::radix8_first_pass_fused_aosoa<is_forward>(in_ptr, out_aosoa, 512);
                common::radix4_aosoa(out_aosoa, in_aosoa, 512, 8, w0);
                const F* HWY_RESTRICT w1 = w0 + twiddles_shift_[1];
                common::radix4_aosoa(in_aosoa, out_aosoa, 512, 32, w1);
                const F* HWY_RESTRICT w2 = w1 + twiddles_shift_[2];
                common::radix4_last_pass_fused_aosoa<is_forward>(out_aosoa, out_ptr, 512, 128, w2);
                return;
            }
            case 10: {
                F* HWY_RESTRICT in_aosoa = workspace_.get();
                F* HWY_RESTRICT out_aosoa = workspace_.get() + 2 * stride_;
                const F* HWY_RESTRICT w0 = twiddles_aosoa_.get();
                common::radix4_first_pass_fused_aosoa<is_forward>(in_ptr, out_aosoa, 1024);
                common::radix4_width4_aosoa(out_aosoa, in_aosoa, 1024, w0);
                const F* HWY_RESTRICT w1 = w0 + twiddles_shift_[1];
                common::radix4_aosoa(in_aosoa, out_aosoa, 1024, 16, w1);
                const F* HWY_RESTRICT w2 = w1 + twiddles_shift_[2];
                common::radix4_aosoa(out_aosoa, in_aosoa, 1024, 64, w2);
                const F* HWY_RESTRICT w3 = w2 + twiddles_shift_[3];
                common::radix4_last_pass_fused_aosoa<is_forward>(in_aosoa, out_ptr, 1024, 256, w3);
                return;
            }
            default:
                break;
            }

            F* HWY_RESTRICT in_aosoa = workspace_.get();
            F* HWY_RESTRICT out_aosoa = workspace_.get() + 2 * stride_;
            const F* HWY_RESTRICT w_ptr = twiddles_aosoa_.get();

            if (stages_[0] == common::StageType::kRadix4FirstPass) {
                common::radix4_first_pass_fused_aosoa<is_forward>(in_ptr, out_aosoa, n);
            } else {
                common::radix8_first_pass_fused_aosoa<is_forward>(in_ptr, out_aosoa, n);
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
                default:
                    break;
                }
                w_ptr += twiddles_shift_[i];
                std::swap(in_aosoa, out_aosoa);
            }
            common::radix4_last_pass_fused_aosoa<is_forward>(in_aosoa, out_ptr, n, width, w_ptr);
        }
    };
}
