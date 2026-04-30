#pragma once

#include <cassert>
#include <complex>
#include <fftw3.h>
#include <span>
#include <vector>

namespace zlbenchmark {

    template <typename F, unsigned Flags = FFTW_MEASURE>
    class FFTW3FFT;

    template <unsigned Flags>
    class FFTW3FFT<double, Flags> final {
        using C = std::complex<double>;

    public:
        explicit FFTW3FFT(const size_t order) : size_(1 << order) {
            auto* dummy_in = fftw_alloc_complex(size_);
            auto* dummy_out = fftw_alloc_complex(size_);

            plan_ = fftw_plan_dft_1d(size_, dummy_in, dummy_out, FFTW_FORWARD, Flags);

            fftw_free(dummy_in);
            fftw_free(dummy_out);
        }

        ~FFTW3FFT() { fftw_destroy_plan(plan_); }

        void forward(C* in_buffer, C* out_buffer) {
            auto* in_ptr = reinterpret_cast<fftw_complex*>(in_buffer);
            auto* out_ptr = reinterpret_cast<fftw_complex*>(out_buffer);

            fftw_execute_dft(plan_, in_ptr, out_ptr);
        }

    private:
        size_t size_;
        fftw_plan plan_;
    };

    template <unsigned Flags>
    class FFTW3FFT<float, Flags> final {
        using C = std::complex<float>;

    public:
        explicit FFTW3FFT(const size_t order) : size_(1 << order) {
            auto* dummy_in = fftwf_alloc_complex(size_);
            auto* dummy_out = fftwf_alloc_complex(size_);

            plan_ = fftwf_plan_dft_1d(size_, dummy_in, dummy_out, FFTW_FORWARD, Flags);

            fftwf_free(dummy_in);
            fftwf_free(dummy_out);
        }

        ~FFTW3FFT() { fftwf_destroy_plan(plan_); }

        void forward(C* in_buffer, C* out_buffer) {
            auto* in_ptr = reinterpret_cast<fftwf_complex*>(in_buffer);
            auto* out_ptr = reinterpret_cast<fftwf_complex*>(out_buffer);

            fftwf_execute_dft(plan_, in_ptr, out_ptr);
        }

    private:
        size_t size_;
        fftwf_plan plan_;
    };

    template <typename F, unsigned Flags = FFTW_MEASURE>
    class FFTW3RFFT;

    template <unsigned Flags>
    class FFTW3RFFT<double, Flags> final {
        using C = std::complex<double>;

    public:
        explicit FFTW3RFFT(const size_t order) : size_(1 << order) {
            auto* dummy_in = fftw_alloc_real(size_);
            auto* dummy_out = fftw_alloc_complex(size_ / 2 + 1);

            plan_ = fftw_plan_dft_r2c_1d(size_, dummy_in, dummy_out, Flags);

            fftw_free(dummy_in);
            fftw_free(dummy_out);
        }

        ~FFTW3RFFT() { fftw_destroy_plan(plan_); }

        void forward(const double* in_buffer, C* out_buffer) {
            auto* in_ptr = const_cast<double*>(in_buffer);
            auto* out_ptr = reinterpret_cast<fftw_complex*>(out_buffer);

            fftw_execute_dft_r2c(plan_, in_ptr, out_ptr);
        }

    private:
        size_t size_;
        fftw_plan plan_;
    };

    template <unsigned Flags>
    class FFTW3RFFT<float, Flags> final {
        using C = std::complex<float>;

    public:
        explicit FFTW3RFFT(const size_t order) : size_(1 << order) {
            auto* dummy_in = fftwf_alloc_real(size_);
            auto* dummy_out = fftwf_alloc_complex(size_ / 2 + 1);

            plan_ = fftwf_plan_dft_r2c_1d(size_, dummy_in, dummy_out, Flags);

            fftwf_free(dummy_in);
            fftwf_free(dummy_out);
        }

        ~FFTW3RFFT() { fftwf_destroy_plan(plan_); }

        void forward(const float* in_buffer, C* out_buffer) {
            auto* in_ptr = const_cast<float*>(in_buffer);
            auto* out_ptr = reinterpret_cast<fftwf_complex*>(out_buffer);

            fftwf_execute_dft_r2c(plan_, in_ptr, out_ptr);
        }

    private:
        size_t size_;
        fftwf_plan plan_;
    };
} // namespace zlbenchmark
