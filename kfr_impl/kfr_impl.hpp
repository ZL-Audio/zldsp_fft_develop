#pragma once

#include <kfr/dft.hpp>
#include <span>
#include <cassert>

namespace zlbenchmark {
    template <typename F>
    class KFRFFT final {
        using C = std::complex<F>;

    public:
        explicit KFRFFT(const size_t order) :
            fft_plan_(1 << order) {
            temp_buffer_.resize(fft_plan_.temp_size);
        }

        void forward(C* in_buffer, C* out_buffer) {
            fft_plan_.execute(out_buffer, in_buffer, temp_buffer_.data());
        }

    private:
        kfr::dft_plan<F> fft_plan_;
        kfr::univector<kfr::u8> temp_buffer_;
    };

    template <typename F>
    class KFRRFFT final {
        using C = std::complex<F>;

    public:
        explicit KFRRFFT(const size_t order) :
            fft_plan_(1 << order) {
            temp_buffer_.resize(fft_plan_.temp_size);
        }

        void forward(const F* in_buffer, C* out_buffer) {
            fft_plan_.execute(reinterpret_cast<kfr::complex<F>*>(out_buffer), in_buffer, temp_buffer_.data());
        }

    private:
        kfr::dft_plan_real<F> fft_plan_;
        kfr::univector<kfr::u8> temp_buffer_;
    };
}
