#pragma once

#include "zldsp_fft_common.hpp"
#include <algorithm>
#include <array>

namespace zldsp::fft {
    namespace hn = hwy::HWY_NAMESPACE;



    template <typename F>
    class RFFT {
        using C = std::complex<F>;

    private:
        size_t order_;
        size_t stride_;
        hwy::AlignedFreeUniquePtr<F[]> workspace_;
        hwy::AlignedFreeUniquePtr<F[]> twiddles_r_;
        hwy::AlignedFreeUniquePtr<F[]> twiddles_i_;
        hwy::AlignedFreeUniquePtr<F[]> twiddles_aosoa_;
        hwy::AlignedFreeUniquePtr<F[]> rfft_twiddles_;
        std::vector<size_t> twiddles_shift_;
        std::vector<common::StageType> stages_;

    public:
        explicit RFFT(const size_t order) :
            order_(std::max(order, static_cast<size_t>(1)) - 1) {
            
            if (order_ >= 4 && order_ < 6) {
                common::generate_order_4_5_twiddles(order_, twiddles_r_, twiddles_i_);
            }

            if (order_ >= 6) {
                const auto mod_result = order_ % 2;
                if (mod_result == 1) {
                    stages_.emplace_back(common::StageType::kRadix8FirstPass);
                    for (size_t i = 3; i < order_ - 2; i += 2) {
                        stages_.emplace_back(common::StageType::kRadix4);
                    }
                } else {
                    stages_.emplace_back(common::StageType::kRadix4FirstPass);
                    stages_.emplace_back(common::StageType::kRadix4Width4);
                    for (size_t i = 4; i < order_ - 2; i += 2) {
                        stages_.emplace_back(common::StageType::kRadix4);
                    }
                }
                stages_.emplace_back(common::StageType::kRadix4LastPass);

                twiddles_shift_.resize(stages_.size());
                twiddles_shift_[0] = 0;

                common::generate_general_twiddles(stages_, twiddles_shift_, twiddles_aosoa_);

                const auto n = static_cast<size_t>(1) << order_;
                const auto pad = (64 / sizeof(F)) + 16;
                stride_ = n + pad;
                workspace_ = hwy::AllocateAligned<F>(4 * stride_);
            } else {
                const auto n = static_cast<size_t>(1) << order;
                stride_ = n;
                workspace_ = hwy::AllocateAligned<F>(4 * stride_);
            }
            generate_rfft_twiddles();
        }

        /**
         * real to AoS forward
         * @param in_buffer
         * @param out_buffer
         */
        void forward(std::span<const F> in_buffer, std::span<C> out_buffer) {
            const size_t M = static_cast<size_t>(1) << order_;
            auto in_c = std::span<const C>(reinterpret_cast<const C*>(in_buffer.data()), M);
            
            F* final_out = workspace_.get();
            if (order_ + 1 >= 6) {
                final_out = ((stages_.size() - 1) % 2 == 1) ? workspace_.get() : workspace_.get() + 2 * stride_;
            }
            
            std::array<std::span<F>, 2> temp_soa = {
                std::span<F>(final_out, M),
                std::span<F>(final_out + stride_, M)
            };
            
            execute<true>(make_aos(in_c), make_soa(temp_soa), M);
            
            if (order_ + 1 >= 6) {
                simd_post_process_forward(hwy::HWY_NAMESPACE::ScalableTag<F>(), make_soa(temp_soa), make_aos(out_buffer), M);
            } else {
                switch (order_ + 1) {
                    case 1: simd_post_process_forward(hwy::HWY_NAMESPACE::CappedTag<F, 1>(), make_soa(temp_soa), make_aos(out_buffer), M); break;
                    case 2: simd_post_process_forward(hwy::HWY_NAMESPACE::CappedTag<F, 1>(), make_soa(temp_soa), make_aos(out_buffer), M); break;
                    case 3: simd_post_process_forward(hwy::HWY_NAMESPACE::CappedTag<F, 2>(), make_soa(temp_soa), make_aos(out_buffer), M); break;
                    case 4: simd_post_process_forward(hwy::HWY_NAMESPACE::CappedTag<F, 4>(), make_soa(temp_soa), make_aos(out_buffer), M); break;
                    case 5: simd_post_process_forward(hwy::HWY_NAMESPACE::CappedTag<F, 8>(), make_soa(temp_soa), make_aos(out_buffer), M); break;
                }
            }
        }

        /**
         * real to SoA forward
         * @param in_buffer
         * @param out_buffer
         */
        void forward(std::span<const F> in_buffer, std::array<std::span<F>, 2> out_buffer) {
            const size_t M = static_cast<size_t>(1) << order_;
            auto in_c = std::span<const C>(reinterpret_cast<const C*>(in_buffer.data()), M);
            
            F* final_out = workspace_.get();
            if (order_ + 1 >= 6) {
                final_out = ((stages_.size() - 1) % 2 == 1) ? workspace_.get() : workspace_.get() + 2 * stride_;
            }
            
            std::array<std::span<F>, 2> temp_soa = {
                std::span<F>(final_out, M),
                std::span<F>(final_out + stride_, M)
            };
            
            execute<true>(make_aos(in_c), make_soa(temp_soa), M);
            
            if (order_ + 1 >= 6) {
                simd_post_process_forward(hwy::HWY_NAMESPACE::ScalableTag<F>(), make_soa(temp_soa), make_soa(out_buffer), M);
            } else {
                switch (order_ + 1) {
                    case 1: simd_post_process_forward(hwy::HWY_NAMESPACE::CappedTag<F, 1>(), make_soa(temp_soa), make_soa(out_buffer), M); break;
                    case 2: simd_post_process_forward(hwy::HWY_NAMESPACE::CappedTag<F, 1>(), make_soa(temp_soa), make_soa(out_buffer), M); break;
                    case 3: simd_post_process_forward(hwy::HWY_NAMESPACE::CappedTag<F, 2>(), make_soa(temp_soa), make_soa(out_buffer), M); break;
                    case 4: simd_post_process_forward(hwy::HWY_NAMESPACE::CappedTag<F, 4>(), make_soa(temp_soa), make_soa(out_buffer), M); break;
                    case 5: simd_post_process_forward(hwy::HWY_NAMESPACE::CappedTag<F, 8>(), make_soa(temp_soa), make_soa(out_buffer), M); break;
                }
            }
        }

        void backward(std::span<const C> in_buffer, std::span<F> out_buffer) {
            const size_t M = static_cast<size_t>(1) << order_;
            F* temp = workspace_.get();
            std::array<std::span<F>, 2> temp_soa = {
                std::span<F>(temp, M),
                std::span<F>(temp + stride_, M)
            };
            
            if (order_ + 1 >= 6) {
                simd_pre_process_backward(hwy::HWY_NAMESPACE::ScalableTag<F>(), make_aos(in_buffer), make_soa(temp_soa), M);
            } else {
                switch (order_ + 1) {
                    case 1: simd_pre_process_backward(hwy::HWY_NAMESPACE::CappedTag<F, 1>(), make_aos(in_buffer), make_soa(temp_soa), M); break;
                    case 2: simd_pre_process_backward(hwy::HWY_NAMESPACE::CappedTag<F, 1>(), make_aos(in_buffer), make_soa(temp_soa), M); break;
                    case 3: simd_pre_process_backward(hwy::HWY_NAMESPACE::CappedTag<F, 2>(), make_aos(in_buffer), make_soa(temp_soa), M); break;
                    case 4: simd_pre_process_backward(hwy::HWY_NAMESPACE::CappedTag<F, 4>(), make_aos(in_buffer), make_soa(temp_soa), M); break;
                    case 5: simd_pre_process_backward(hwy::HWY_NAMESPACE::CappedTag<F, 8>(), make_aos(in_buffer), make_soa(temp_soa), M); break;
                }
            }
            
            auto out_c = std::span<C>(reinterpret_cast<C*>(out_buffer.data()), M);
            execute<false>(make_soa(temp_soa), make_aos(out_c), M);
        }

        /**
         * SoA to real backward
         * @param in_buffer
         * @param out_buffer
         */
        void backward(std::array<std::span<const F>, 2> in_buffer, std::span<F> out_buffer) {
            const size_t M = static_cast<size_t>(1) << order_;
            F* temp = workspace_.get();
            std::array<std::span<F>, 2> temp_soa = {
                std::span<F>(temp, M),
                std::span<F>(temp + stride_, M)
            };
            
            if (order_ + 1 >= 6) {
                simd_pre_process_backward(hwy::HWY_NAMESPACE::ScalableTag<F>(), make_soa(in_buffer), make_soa(temp_soa), M);
            } else {
                switch (order_ + 1) {
                    case 1: simd_pre_process_backward(hwy::HWY_NAMESPACE::CappedTag<F, 1>(), make_soa(in_buffer), make_soa(temp_soa), M); break;
                    case 2: simd_pre_process_backward(hwy::HWY_NAMESPACE::CappedTag<F, 1>(), make_soa(in_buffer), make_soa(temp_soa), M); break;
                    case 3: simd_pre_process_backward(hwy::HWY_NAMESPACE::CappedTag<F, 2>(), make_soa(in_buffer), make_soa(temp_soa), M); break;
                    case 4: simd_pre_process_backward(hwy::HWY_NAMESPACE::CappedTag<F, 4>(), make_soa(in_buffer), make_soa(temp_soa), M); break;
                    case 5: simd_pre_process_backward(hwy::HWY_NAMESPACE::CappedTag<F, 8>(), make_soa(in_buffer), make_soa(temp_soa), M); break;
                }
            }
            
            auto out_c = std::span<C>(reinterpret_cast<C*>(out_buffer.data()), M);
            execute<false>(make_soa(temp_soa), make_aos(out_c), M);
        }

    private:
        template <class D>
        void do_generate_rfft_twiddles(D d) {
            const size_t lanes = hwy::HWY_NAMESPACE::Lanes(d);
            const size_t M = static_cast<size_t>(1) << order_;
            const size_t N = M * 2;
            size_t num_elements = M / 2;
            if (num_elements == 0) return;
            size_t num_blocks = (num_elements + lanes - 1) / lanes;
            size_t twiddle_size = num_blocks * lanes * 2;
            rfft_twiddles_ = hwy::AllocateAligned<F>(twiddle_size);
            
            for (size_t b = 0; b < num_blocks; ++b) {
                for (size_t l = 0; l < lanes; ++l) {
                    size_t idx = b * lanes + l;
                    size_t k = idx + 1;
                    F wc = 0.0;
                    F ws = 0.5; // default for padded zeros
                    if (k <= num_elements) {
                        wc = static_cast<F>(0.5 * std::cos(2.0 * M_PI * k / N));
                        ws = static_cast<F>(0.5 * std::sin(2.0 * M_PI * k / N) + 0.5);
                    }
                    rfft_twiddles_[b * lanes * 2 + l] = wc;
                    rfft_twiddles_[b * lanes * 2 + lanes + l] = ws;
                }
            }
        }

        void generate_rfft_twiddles() {
            if (order_ + 1 < 2) return;
            if (order_ + 1 >= 6) {
                do_generate_rfft_twiddles(hwy::HWY_NAMESPACE::ScalableTag<F>());
            } else {
                switch (order_ + 1) {
                    case 2: do_generate_rfft_twiddles(hwy::HWY_NAMESPACE::CappedTag<F, 1>()); break;
                    case 3: do_generate_rfft_twiddles(hwy::HWY_NAMESPACE::CappedTag<F, 2>()); break;
                    case 4: do_generate_rfft_twiddles(hwy::HWY_NAMESPACE::CappedTag<F, 4>()); break;
                    case 5: do_generate_rfft_twiddles(hwy::HWY_NAMESPACE::CappedTag<F, 8>()); break;
                }
            }
        }

        template <class D, typename OutPtr>
        void simd_post_process_forward(D d, common::SoAPtr<F> temp_soa, OutPtr out_ptr, size_t M) {
            const size_t lanes = hwy::HWY_NAMESPACE::Lanes(d);

            F r0 = temp_soa.real[0];
            F i0 = temp_soa.imag[0];
            common::store_scalar<true>(out_ptr, r0 + i0, static_cast<F>(0.0));
            common::store_scalar<true>(out_ptr.shift(OutPtr::get_complex_offset(M)), r0 - i0, static_cast<F>(0.0));

            const size_t num_elements = M / 2;
            const size_t num_blocks = (num_elements + lanes - 1) / lanes;
            
            for (size_t b = 0; b < num_blocks; ++b) {
                const size_t k = b * lanes + 1;
                
                const auto r1 = hn::LoadU(d, temp_soa.real + k);
                const auto i1 = hn::LoadU(d, temp_soa.imag + k);
                
                const auto r2_raw = hn::LoadU(d, temp_soa.real + M - k - lanes + 1);
                const auto i2_raw = hn::LoadU(d, temp_soa.imag + M - k - lanes + 1);
                
                const auto r2 = hn::Reverse(d, r2_raw);
                const auto i2 = hn::Reverse(d, i2_raw);
                
                const auto wc = hn::Load(d, &rfft_twiddles_[b * lanes * 2]);
                const auto ws = hn::Load(d, &rfft_twiddles_[b * lanes * 2 + lanes]);
                
                const auto si = hn::Add(i1, i2);
                const auto dr = hn::Sub(r1, r2);
                
                const auto xr_tmp = hn::MulAdd(si, wc, r1);
                const auto xr_k   = hn::NegMulAdd(dr, ws, xr_tmp);
                const auto xi_tmp = hn::NegMulAdd(si, ws, i1);
                const auto xi_k   = hn::NegMulAdd(dr, wc, xi_tmp);
                
                const auto xr2_tmp = hn::NegMulAdd(si, wc, r2);
                const auto xr_Mk   = hn::MulAdd(dr, ws, xr2_tmp);
                const auto xi2_tmp = hn::NegMulAdd(si, ws, i2);
                const auto xi_Mk   = hn::NegMulAdd(dr, wc, xi2_tmp);
                
                common::store_complex<true>(d, out_ptr.shift(OutPtr::get_complex_offset(k)), xr_k, xi_k);
                
                const auto xr_Mk_rev = hn::Reverse(d, xr_Mk);
                const auto xi_Mk_rev = hn::Reverse(d, xi_Mk);
                
                common::store_complex<true>(d, out_ptr.shift(OutPtr::get_complex_offset(M - k - lanes + 1)), xr_Mk_rev, xi_Mk_rev);
            }
        }

        template <class D, typename InPtr>
        void simd_pre_process_backward(D d, InPtr in_ptr, common::SoAPtr<F> temp_soa, size_t M) {
            const size_t lanes = hwy::HWY_NAMESPACE::Lanes(d);

            F r0, i0, rM, iM;
            common::load_scalar<true>(in_ptr, r0, i0);
            common::load_scalar<true>(in_ptr.shift(InPtr::get_complex_offset(M)), rM, iM);
            temp_soa.real[0] = (r0 + rM) * static_cast<F>(0.5);
            temp_soa.imag[0] = (r0 - rM) * static_cast<F>(0.5);

            const size_t num_elements = M / 2;
            const size_t num_blocks = (num_elements + lanes - 1) / lanes;
            
            for (size_t b = 0; b < num_blocks; ++b) {
                const size_t k = b * lanes + 1;
                
                hn::Vec<decltype(d)> r1, i1;
                common::load_complex<true>(d, in_ptr.shift(InPtr::get_complex_offset(k)), r1, i1);
                
                hn::Vec<decltype(d)> r2_raw, i2_raw;
                common::load_complex<true>(d, in_ptr.shift(InPtr::get_complex_offset(M - k - lanes + 1)), r2_raw, i2_raw);
                
                const auto r2 = hn::Reverse(d, r2_raw);
                const auto i2 = hn::Reverse(d, i2_raw);
                
                const auto wc = hn::Load(d, &rfft_twiddles_[b * lanes * 2]);
                const auto ws = hn::Load(d, &rfft_twiddles_[b * lanes * 2 + lanes]);
                
                const auto si = hn::Add(i1, i2);
                const auto dr = hn::Sub(r1, r2);
                
                const auto zr_tmp = hn::NegMulAdd(si, wc, r1);
                const auto zr_k   = hn::NegMulAdd(dr, ws, zr_tmp);
                const auto zi_tmp = hn::NegMulAdd(si, ws, i1);
                const auto zi_k   = hn::MulAdd(dr, wc, zi_tmp);
                
                const auto zr2_tmp = hn::MulAdd(si, wc, r2);
                const auto zr_Mk   = hn::MulAdd(dr, ws, zr2_tmp);
                const auto zi2_tmp = hn::NegMulAdd(si, ws, i2);
                const auto zi_Mk   = hn::MulAdd(dr, wc, zi2_tmp);
                
                hn::StoreU(zr_k, d, temp_soa.real + k);
                hn::StoreU(zi_k, d, temp_soa.imag + k);
                
                const auto zr_Mk_rev = hn::Reverse(d, zr_Mk);
                const auto zi_Mk_rev = hn::Reverse(d, zi_Mk);
                
                hn::StoreU(zr_Mk_rev, d, temp_soa.real + M - k - lanes + 1);
                hn::StoreU(zi_Mk_rev, d, temp_soa.imag + M - k - lanes + 1);
            }
        }

        static common::AoSPtr<F> make_aos(std::span<C> span) {
            return common::AoSPtr<F>{reinterpret_cast<F*>(span.data())};
        }

        static common::AoSPtr<F> make_aos(std::span<const C> span) {
            return common::AoSPtr<F>{reinterpret_cast<F*>(const_cast<C*>(span.data()))};
        }

        static common::SoAPtr<F> make_soa(std::array<std::span<F>, 2> span) {
            return common::SoAPtr<F>{span[0].data(), span[1].data()};
        }

        static common::SoAPtr<F> make_soa(std::array<std::span<const F>, 2> span) {
            return common::SoAPtr<F>{const_cast<F*>(span[0].data()), const_cast<F*>(span[1].data())};
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
                F* __restrict in_aosoa = workspace_.get();
                F* __restrict out_aosoa = workspace_.get() + 2 * stride_;
                const F* __restrict w0 = twiddles_aosoa_.get();
                common::radix4_first_pass_fused_aosoa<is_forward>(in_ptr, out_aosoa, 64);
                common::radix4_width4_aosoa(out_aosoa, in_aosoa, 64, w0);
                const F* __restrict w1 = w0 + twiddles_shift_[1];
                common::radix4_last_pass_fused_aosoa<is_forward>(in_aosoa, out_ptr, 64, 16, w1);
                return;
            }
            case 7: {
                F* __restrict in_aosoa = workspace_.get();
                F* __restrict out_aosoa = workspace_.get() + 2 * stride_;
                const F* __restrict w0 = twiddles_aosoa_.get();
                common::radix8_first_pass_fused_aosoa<is_forward>(in_ptr, out_aosoa, 128);
                common::radix4_aosoa(out_aosoa, in_aosoa, 128, 8, w0);
                const F* __restrict w1 = w0 + twiddles_shift_[1];
                common::radix4_last_pass_fused_aosoa<is_forward>(in_aosoa, out_ptr, 128, 32, w1);
                return;
            }
            case 8: {
                F* __restrict in_aosoa = workspace_.get();
                F* __restrict out_aosoa = workspace_.get() + 2 * stride_;
                const F* __restrict w0 = twiddles_aosoa_.get();
                common::radix4_first_pass_fused_aosoa<is_forward>(in_ptr, out_aosoa, 256);
                common::radix4_width4_aosoa(out_aosoa, in_aosoa, 256, w0);
                const F* __restrict w1 = w0 + twiddles_shift_[1];
                common::radix4_aosoa(in_aosoa, out_aosoa, 256, 16, w1);
                const F* __restrict w2 = w1 + twiddles_shift_[2];
                common::radix4_last_pass_fused_aosoa<is_forward>(out_aosoa, out_ptr, 256, 64, w2);
                return;
            }
            case 9: {
                F* __restrict in_aosoa = workspace_.get();
                F* __restrict out_aosoa = workspace_.get() + 2 * stride_;
                const F* __restrict w0 = twiddles_aosoa_.get();
                common::radix8_first_pass_fused_aosoa<is_forward>(in_ptr, out_aosoa, 512);
                common::radix4_aosoa(out_aosoa, in_aosoa, 512, 8, w0);
                const F* __restrict w1 = w0 + twiddles_shift_[1];
                common::radix4_aosoa(in_aosoa, out_aosoa, 512, 32, w1);
                const F* __restrict w2 = w1 + twiddles_shift_[2];
                common::radix4_last_pass_fused_aosoa<is_forward>(out_aosoa, out_ptr, 512, 128, w2);
                return;
            }
            case 10: {
                F* __restrict in_aosoa = workspace_.get();
                F* __restrict out_aosoa = workspace_.get() + 2 * stride_;
                const F* __restrict w0 = twiddles_aosoa_.get();
                common::radix4_first_pass_fused_aosoa<is_forward>(in_ptr, out_aosoa, 1024);
                common::radix4_width4_aosoa(out_aosoa, in_aosoa, 1024, w0);
                const F* __restrict w1 = w0 + twiddles_shift_[1];
                common::radix4_aosoa(in_aosoa, out_aosoa, 1024, 16, w1);
                const F* __restrict w2 = w1 + twiddles_shift_[2];
                common::radix4_aosoa(out_aosoa, in_aosoa, 1024, 64, w2);
                const F* __restrict w3 = w2 + twiddles_shift_[3];
                common::radix4_last_pass_fused_aosoa<is_forward>(in_aosoa, out_ptr, 1024, 256, w3);
                return;
            }
            default:
                break;
            }

            F* __restrict in_aosoa = workspace_.get();
            F* __restrict out_aosoa = workspace_.get() + 2 * stride_;
            const F* __restrict w_ptr = twiddles_aosoa_.get();

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
