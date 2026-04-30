#pragma once

#include <span>
#include <vector>
#include <complex>
#include <cassert>

#include "pffft/include/pffft/pffft.h"
#include "pffft/include/pffft/pffft_double.h"

namespace zlbenchmark {
    template <typename F>
    class PffftFFT;

    template <>
    class PffftFFT<double> final {
        using C = std::complex<double>;

    public:
        explicit PffftFFT(const size_t order) :
            size_(1 << order) {
            setup_ = pffftd_new_setup(static_cast<int>(size_), PFFFT_COMPLEX);

            work_ = static_cast<double*>(pffft_aligned_malloc(size_ * 2 * sizeof(double)));
        }

        ~PffftFFT() {
            if (work_)
                pffft_aligned_free(work_);
            if (setup_)
                pffftd_destroy_setup(setup_);
        }

        void forward(const C* in_buffer, C* out_buffer) {
            if (!setup_) {
                return;
            }
            auto* in_ptr = reinterpret_cast<const double*>(in_buffer);
            auto* out_ptr = reinterpret_cast<double*>(out_buffer);

            pffftd_transform_ordered(setup_, in_ptr, out_ptr, work_, PFFFT_FORWARD);
        }

    private:
        size_t size_;
        PFFFTD_Setup* setup_ = nullptr;
        double* work_ = nullptr;
    };

    template <>
    class PffftFFT<float> final {
        using C = std::complex<float>;

    public:
        explicit PffftFFT(const size_t order) :
            size_(1 << order) {
            setup_ = pffft_new_setup(static_cast<int>(size_), PFFFT_COMPLEX);

            work_ = static_cast<float*>(pffft_aligned_malloc(size_ * 2 * sizeof(float)));
        }

        ~PffftFFT() {
            if (work_)
                pffft_aligned_free(work_);
            if (setup_)
                pffft_destroy_setup(setup_);
        }

        void forward(const C* in_buffer, C* out_buffer) {
            if (!setup_) {
                return;
            }
            auto* in_ptr = reinterpret_cast<const float*>(in_buffer);
            auto* out_ptr = reinterpret_cast<float*>(out_buffer);

            pffft_transform_ordered(setup_, in_ptr, out_ptr, work_, PFFFT_FORWARD);
        }

    private:
        size_t size_;
        PFFFT_Setup* setup_ = nullptr;
        float* work_ = nullptr;
    };

    template <typename F>
    class PffftRFFT;

    template <>
    class PffftRFFT<double> final {
        using C = std::complex<double>;

    public:
        explicit PffftRFFT(const size_t order) :
            size_(1 << order) {
            setup_ = pffftd_new_setup(static_cast<int>(size_), PFFFT_REAL);

            work_ = static_cast<double*>(pffft_aligned_malloc(size_ * 2 * sizeof(double)));
            out_real_ = static_cast<double*>(pffft_aligned_malloc(size_ * sizeof(double)));
        }

        ~PffftRFFT() {
            if (out_real_)
                pffft_aligned_free(out_real_);
            if (work_)
                pffft_aligned_free(work_);
            if (setup_)
                pffftd_destroy_setup(setup_);
        }

        void forward(const double* in_buffer, C* out_buffer) {
            if (!setup_) {
                return;
            }
            pffftd_transform_ordered(setup_, in_buffer, out_real_, work_, PFFFT_FORWARD);
            
#ifndef THROUGHPUT_RFFT_TEST
            out_buffer[0] = C(out_real_[0], 0.0);
            out_buffer[size_ / 2] = C(out_real_[1], 0.0);
            for (size_t k = 1; k < size_ / 2; ++k) {
                out_buffer[k] = C(out_real_[2 * k], out_real_[2 * k + 1]);
            }
#endif
        }

    private:
        size_t size_;
        PFFFTD_Setup* setup_ = nullptr;
        double* work_ = nullptr;
        double* out_real_ = nullptr;
    };

    template <>
    class PffftRFFT<float> final {
        using C = std::complex<float>;

    public:
        explicit PffftRFFT(const size_t order) :
            size_(1 << order) {
            setup_ = pffft_new_setup(static_cast<int>(size_), PFFFT_REAL);

            work_ = static_cast<float*>(pffft_aligned_malloc(size_ * 2 * sizeof(float)));
            out_real_ = static_cast<float*>(pffft_aligned_malloc(size_ * sizeof(float)));
        }

        ~PffftRFFT() {
            if (out_real_)
                pffft_aligned_free(out_real_);
            if (work_)
                pffft_aligned_free(work_);
            if (setup_)
                pffft_destroy_setup(setup_);
        }

        void forward(const float* in_buffer, C* out_buffer) {
            if (!setup_) {
                return;
            }

            pffft_transform_ordered(setup_, in_buffer, out_real_, work_, PFFFT_FORWARD);
            
#ifndef THROUGHPUT_RFFT_TEST
            out_buffer[0] = C(out_real_[0], 0.0f);
            out_buffer[size_ / 2] = C(out_real_[1], 0.0f);
            for (size_t k = 1; k < size_ / 2; ++k) {
                out_buffer[k] = C(out_real_[2 * k], out_real_[2 * k + 1]);
            }
#endif
        }

    private:
        size_t size_;
        PFFFT_Setup* setup_ = nullptr;
        float* work_ = nullptr;
        float* out_real_ = nullptr;
    };

}
