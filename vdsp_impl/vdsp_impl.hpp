#pragma once

#include <Accelerate/Accelerate.h>
#include <stdexcept>
#include "../benchmark/fft_support.hpp"

namespace zlbenchmark {
    template <typename F>
    class VDSPFFT;

    template <>
    class VDSPFFT<float> final {
    public:
        explicit VDSPFFT(const size_t order) :
            order_(order), n_(fft_size(order)), fft_setup_(vDSP_create_fftsetup(order, FFT_RADIX2)) {
            if (!fft_setup_) {
                throw std::runtime_error("vDSP_create_fftsetup failed");
            }
        }

        VDSPFFT(const VDSPFFT&) = delete;

        VDSPFFT& operator=(const VDSPFFT&) = delete;

        ~VDSPFFT() {
            if (fft_setup_) {
                vDSP_destroy_fftsetup(fft_setup_);
            }
        }

        // 2*N scalars, split into real/imaginary halves.
        void forward(float* in_buffer, float* out_buffer) {
            transform(in_buffer, out_buffer, FFT_FORWARD);
        }

        void backward(float* in_buffer, float* out_buffer) {
            transform(in_buffer, out_buffer, FFT_INVERSE);
        }

    private:
        void transform(float* in_buffer, float* out_buffer, FFTDirection direction) {
            DSPSplitComplex in{in_buffer, in_buffer + n_}, out{out_buffer, out_buffer + n_};
            vDSP_fft_zop(fft_setup_, &in, 1, &out, 1, order_, direction);
        }

    private:
        size_t order_;
        size_t n_;
        FFTSetup fft_setup_;
    };

    template <>
    class VDSPFFT<double> final {
    public:
        explicit VDSPFFT(const size_t order) :
            order_(order), n_(fft_size(order)), fft_setup_(vDSP_create_fftsetupD(order, FFT_RADIX2)) {
            if (!fft_setup_) {
                throw std::runtime_error("vDSP_create_fftsetupD failed");
            }
        }

        VDSPFFT(const VDSPFFT&) = delete;

        VDSPFFT& operator=(const VDSPFFT&) = delete;

        ~VDSPFFT() {
            if (fft_setup_) {
                vDSP_destroy_fftsetupD(fft_setup_);
            }
        }

        void forward(double* in_buffer, double* out_buffer) {
            transform(in_buffer, out_buffer, FFT_FORWARD);
        }

        void backward(double* in_buffer, double* out_buffer) {
            transform(in_buffer, out_buffer, FFT_INVERSE);
        }

    private:
        void transform(double* in_buffer, double* out_buffer, FFTDirection direction) {
            DSPDoubleSplitComplex in{in_buffer, in_buffer + n_}, out{out_buffer, out_buffer + n_};
            vDSP_fft_zopD(fft_setup_, &in, 1, &out, 1, order_, direction);
        }

    private:
        size_t order_;
        size_t n_;
        FFTSetupD fft_setup_;
    };

    template <typename F>
    class VDSPRFFT;

    template <>
    class VDSPRFFT<float> final {
    public:
        explicit VDSPRFFT(const size_t order) :
            order_(order), n_(fft_size(order)), fft_setup_(vDSP_create_fftsetup(order, FFT_RADIX2)) {
            if (!fft_setup_) {
                throw std::runtime_error("vDSP_create_fftsetup failed");
            }
        }

        VDSPRFFT(const VDSPRFFT&) = delete;

        VDSPRFFT& operator=(const VDSPRFFT&) = delete;

        ~VDSPRFFT() {
            if (fft_setup_) {
                vDSP_destroy_fftsetup(fft_setup_);
            }
        }

        // N scalars: even/odd time samples in separate halves; native split
        // spectrum with DC/Nyquist packed at the start. Keep vDSP's scaling.
        void forward(float* in_buffer, float* out_buffer) {
            transform(in_buffer, out_buffer, FFT_FORWARD);
        }

        void backward(float* in_buffer, float* out_buffer) {
            transform(in_buffer, out_buffer, FFT_INVERSE);
        }

    private:
        void transform(float* in_buffer, float* out_buffer, FFTDirection direction) {
            DSPSplitComplex in{in_buffer, in_buffer + n_ / 2}, out{out_buffer, out_buffer + n_ / 2};
            vDSP_fft_zrop(fft_setup_, &in, 1, &out, 1, order_, direction);
        }

    private:
        size_t order_;
        size_t n_;
        FFTSetup fft_setup_;
    };

    template <>
    class VDSPRFFT<double> final {
    public:
        explicit VDSPRFFT(const size_t order) :
            order_(order), n_(fft_size(order)), fft_setup_(vDSP_create_fftsetupD(order, FFT_RADIX2)) {
            if (!fft_setup_) {
                throw std::runtime_error("vDSP_create_fftsetupD failed");
            }
        }

        VDSPRFFT(const VDSPRFFT&) = delete;

        VDSPRFFT& operator=(const VDSPRFFT&) = delete;

        ~VDSPRFFT() {
            if (fft_setup_) {
                vDSP_destroy_fftsetupD(fft_setup_);
            }
        }

        void forward(double* in_buffer, double* out_buffer) {
            transform(in_buffer, out_buffer, FFT_FORWARD);
        }

        void backward(double* in_buffer, double* out_buffer) {
            transform(in_buffer, out_buffer, FFT_INVERSE);
        }

    private:
        void transform(double* in_buffer, double* out_buffer, FFTDirection direction) {
            DSPDoubleSplitComplex in{in_buffer, in_buffer + n_ / 2}, out{out_buffer, out_buffer + n_ / 2};
            vDSP_fft_zropD(fft_setup_, &in, 1, &out, 1, order_, direction);
        }

    private:
        size_t order_;
        size_t n_;
        FFTSetupD fft_setup_;
    };
}
