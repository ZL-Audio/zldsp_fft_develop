#pragma once

#include <complex>
#include <cstddef>
#include <limits>
#include <new>
#include <stdexcept>

#include <armpl.h>
#include <fftw3.h>

namespace zlbenchmark {
    namespace detail {
        template <typename F>
        struct ArmPLFFTW;

        template <>
        struct ArmPLFFTW<float> {
            using real_type = float;
            using complex_type = fftwf_complex;
            using plan_type = fftwf_plan;

            static complex_type* alloc_complex(const std::size_t size) { return fftwf_alloc_complex(size); }
            static void free_complex(complex_type* ptr) { fftwf_free(ptr); }
            static real_type* alloc_real(const std::size_t size) { return fftwf_alloc_real(size); }
            static void free_real(real_type* ptr) { fftwf_free(ptr); }

            static plan_type plan_cfft(const int size, complex_type* in, complex_type* out, const unsigned flags) {
                return fftwf_plan_dft_1d(size, in, out, FFTW_FORWARD, flags);
            }

            static plan_type plan_rfft(const int size, real_type* in, complex_type* out, const unsigned flags) {
                return fftwf_plan_dft_r2c_1d(size, in, out, flags);
            }

            static void execute_cfft(plan_type plan, complex_type* in, complex_type* out) {
                fftwf_execute_dft(plan, in, out);
            }

            static void execute_rfft(plan_type plan, real_type* in, complex_type* out) {
                fftwf_execute_dft_r2c(plan, in, out);
            }

            static void destroy_plan(plan_type plan) { fftwf_destroy_plan(plan); }
        };

        template <>
        struct ArmPLFFTW<double> {
            using real_type = double;
            using complex_type = fftw_complex;
            using plan_type = fftw_plan;

            static complex_type* alloc_complex(const std::size_t size) { return fftw_alloc_complex(size); }
            static void free_complex(complex_type* ptr) { fftw_free(ptr); }
            static real_type* alloc_real(const std::size_t size) { return fftw_alloc_real(size); }
            static void free_real(real_type* ptr) { fftw_free(ptr); }

            static plan_type plan_cfft(const int size, complex_type* in, complex_type* out, const unsigned flags) {
                return fftw_plan_dft_1d(size, in, out, FFTW_FORWARD, flags);
            }

            static plan_type plan_rfft(const int size, real_type* in, complex_type* out, const unsigned flags) {
                return fftw_plan_dft_r2c_1d(size, in, out, flags);
            }

            static void execute_cfft(plan_type plan, complex_type* in, complex_type* out) {
                fftw_execute_dft(plan, in, out);
            }

            static void execute_rfft(plan_type plan, real_type* in, complex_type* out) {
                fftw_execute_dft_r2c(plan, in, out);
            }

            static void destroy_plan(plan_type plan) { fftw_destroy_plan(plan); }
        };

        inline int armpl_fft_size(const std::size_t order) {
            if (order >= std::numeric_limits<std::size_t>::digits) {
                throw std::invalid_argument("ArmPL FFT order is too large");
            }

            const auto size = std::size_t{1} << order;
            if (size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::invalid_argument("ArmPL FFT size exceeds the FFTW interface limit");
            }
            return static_cast<int>(size);
        }
    }

    template <typename F, unsigned Flags = FFTW_MEASURE>
    class ArmPLFFT final {
        using C = std::complex<F>;
        using FFTW = detail::ArmPLFFTW<F>;

    public:
        explicit ArmPLFFT(const std::size_t order) : size_(detail::armpl_fft_size(order)) {
            auto* dummy_in = FFTW::alloc_complex(static_cast<std::size_t>(size_));
            auto* dummy_out = FFTW::alloc_complex(static_cast<std::size_t>(size_));
            if (!dummy_in || !dummy_out) {
                FFTW::free_complex(dummy_in);
                FFTW::free_complex(dummy_out);
                throw std::bad_alloc();
            }

            plan_ = FFTW::plan_cfft(size_, dummy_in, dummy_out, Flags);
            FFTW::free_complex(dummy_in);
            FFTW::free_complex(dummy_out);

            if (!plan_) {
                throw std::runtime_error("ArmPL CFFT plan creation failed");
            }
        }

        ~ArmPLFFT() {
            if (plan_) {
                FFTW::destroy_plan(plan_);
            }
        }

        void forward(C* in_buffer, C* out_buffer) {
            FFTW::execute_cfft(plan_, reinterpret_cast<typename FFTW::complex_type*>(in_buffer),
                               reinterpret_cast<typename FFTW::complex_type*>(out_buffer));
        }

    private:
        int size_;
        typename FFTW::plan_type plan_{nullptr};
    };

    template <typename F, unsigned Flags = FFTW_MEASURE>
    class ArmPLRFFT final {
        using C = std::complex<F>;
        using FFTW = detail::ArmPLFFTW<F>;

    public:
        explicit ArmPLRFFT(const std::size_t order) : size_(detail::armpl_fft_size(order)) {
            auto* dummy_in = FFTW::alloc_real(static_cast<std::size_t>(size_));
            auto* dummy_out = FFTW::alloc_complex(static_cast<std::size_t>(size_ / 2 + 1));
            if (!dummy_in || !dummy_out) {
                FFTW::free_real(dummy_in);
                FFTW::free_complex(dummy_out);
                throw std::bad_alloc();
            }

            plan_ = FFTW::plan_rfft(size_, dummy_in, dummy_out, Flags);
            FFTW::free_real(dummy_in);
            FFTW::free_complex(dummy_out);

            if (!plan_) {
                throw std::runtime_error("ArmPL RFFT plan creation failed");
            }
        }

        ~ArmPLRFFT() {
            if (plan_) {
                FFTW::destroy_plan(plan_);
            }
        }

        void forward(const F* in_buffer, C* out_buffer) {
            FFTW::execute_rfft(plan_, const_cast<F*>(in_buffer),
                               reinterpret_cast<typename FFTW::complex_type*>(out_buffer));
        }

    private:
        int size_;
        typename FFTW::plan_type plan_{nullptr};
    };

    template <typename F, unsigned Flags = FFTW_MEASURE>
    using ARMPLFFT = ArmPLFFT<F, Flags>;

    template <typename F, unsigned Flags = FFTW_MEASURE>
    using ARMPLRFFT = ArmPLRFFT<F, Flags>;
}
