#pragma once

#include <array>
#include <complex>
#include <cstddef>
#include <memory>
#include <type_traits>

namespace zldsp_fft_example {
    template <typename F>
    using Complex = std::complex<F>;

    template <typename F>
    using SoA = std::array<F*, 2>;

    template <typename F>
    class CFFT {
    public:
        virtual ~CFFT() = default;

        virtual void forward(Complex<F>* input, Complex<F>* output) noexcept = 0;
        virtual void forward(Complex<F>* input, SoA<F> output) noexcept = 0;
        virtual void forward(SoA<F> input, Complex<F>* output) noexcept = 0;
        virtual void forward(SoA<F> input, SoA<F> output) noexcept = 0;
        virtual void backward(Complex<F>* input, Complex<F>* output) noexcept = 0;
        virtual void backward(Complex<F>* input, SoA<F> output) noexcept = 0;
        virtual void backward(SoA<F> input, Complex<F>* output) noexcept = 0;
        virtual void backward(SoA<F> input, SoA<F> output) noexcept = 0;
    };

    template <typename F>
    class RFFT {
    public:
        virtual ~RFFT() = default;

        virtual void forward(F* input, Complex<F>* output) noexcept = 0;
        virtual void forward(F* input, SoA<F> output) noexcept = 0;
        virtual void backward(Complex<F>* input, F* output) noexcept = 0;
        virtual void backward(SoA<F> input, F* output) noexcept = 0;
        virtual void forward_sqr_mag(F* input, F* output) noexcept = 0;
    };

    namespace detail {
        std::unique_ptr<CFFT<float>> prepare_cfft_float(std::size_t order);
        std::unique_ptr<CFFT<double>> prepare_cfft_double(std::size_t order);
        std::unique_ptr<RFFT<float>> prepare_rfft_float(std::size_t order);
        std::unique_ptr<RFFT<double>> prepare_rfft_double(std::size_t order);
    }

    template <typename F>
    std::unique_ptr<CFFT<F>> prepare_cfft(const std::size_t order) {
        if constexpr (std::is_same_v<F, float>) {
            return detail::prepare_cfft_float(order);
        }
        else if constexpr (std::is_same_v<F, double>) {
            return detail::prepare_cfft_double(order);
        }
        else {
            static_assert(std::is_same_v<F, float> || std::is_same_v<F, double>);
        }
    }

    template <typename F>
    std::unique_ptr<RFFT<F>> prepare_rfft(const std::size_t order) {
        if constexpr (std::is_same_v<F, float>) {
            return detail::prepare_rfft_float(order);
        }
        else if constexpr (std::is_same_v<F, double>) {
            return detail::prepare_rfft_double(order);
        }
        else {
            static_assert(std::is_same_v<F, float> || std::is_same_v<F, double>);
        }
    }
}
