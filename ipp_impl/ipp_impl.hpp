#pragma once

#include <complex>
#include <mutex>
#include <span>
#include <stdexcept>
#include <vector>

#include <ippcore.h>
#include <ipps.h>

namespace zlbenchmark {
    enum class IPPSimdLevel {
        Auto,
        SSE42,
        AVX2,
        AVX512
    };

    inline std::mutex& get_ipp_mutex() {
        static std::mutex mtx;
        return mtx;
    }

    class ScopedIPPDispatcher {
    public:
        explicit ScopedIPPDispatcher(IPPSimdLevel target_level) {
            std::lock_guard<std::mutex> lock(get_ipp_mutex());

            ippInit();
            ippGetCpuFeatures(&original_features_, nullptr);

            if (target_level == IPPSimdLevel::Auto) return;

            Ipp64u cpuFeatures = original_features_;

            if (target_level <= IPPSimdLevel::AVX2) {
                cpuFeatures &= ~(ippCPUID_AVX512F | ippCPUID_AVX512CD |
                                 ippCPUID_AVX512VL | ippCPUID_AVX512BW | ippCPUID_AVX512DQ);
            }
            if (target_level <= IPPSimdLevel::SSE42) {
                cpuFeatures &= ~(ippCPUID_AVX | ippCPUID_AVX2);
            }

            if (ippSetCpuFeatures(cpuFeatures) != ippStsNoErr) {
                throw std::runtime_error("Failed to explicitly set IPP CPU features.");
            }
        }

        ~ScopedIPPDispatcher() {
            std::lock_guard<std::mutex> lock(get_ipp_mutex());
            ippSetCpuFeatures(original_features_);
        }

    private:
        Ipp64u original_features_{0};
    };

    template <typename F, IPPSimdLevel kSimd = IPPSimdLevel::Auto>
    class IPPFFT;

    template <IPPSimdLevel kSimd>
    class IPPFFT<float, kSimd> final {
        using C = std::complex<float>;

    public:
        explicit IPPFFT(const int order) : order_(order), n_(1 << order) {
            int specSize = 0, initSize = 0, bufSize = 0;

            IppStatus status = ippsFFTGetSize_C_32fc(order_, IPP_FFT_DIV_INV_BY_N, ippAlgHintNone, &specSize, &initSize, &bufSize);
            if (status != ippStsNoErr) {
                throw std::runtime_error("ippsFFTGetSize_C_32fc failed");
            }

            spec_mem_ = ippsMalloc_8u(specSize);
            work_buf_ = bufSize > 0 ? ippsMalloc_8u(bufSize) : nullptr;
            Ipp8u* init_buf = initSize > 0 ? ippsMalloc_8u(initSize) : nullptr;

            status = ippsFFTInit_C_32fc(&spec_, order_, IPP_FFT_DIV_INV_BY_N, ippAlgHintNone, spec_mem_, init_buf);

            if (init_buf) {
                ippsFree(init_buf);
            }

            if (status != ippStsNoErr) {
                if (spec_mem_) ippsFree(spec_mem_);
                if (work_buf_) ippsFree(work_buf_);
                throw std::runtime_error("ippsFFTInit_C_32fc failed");
            }
        }

        ~IPPFFT() {
            if (spec_mem_) {
                ippsFree(spec_mem_);
            }
            if (work_buf_) {
                ippsFree(work_buf_);
            }
        }

        void forward(C* in_buffer, C* out_buffer) {
            ippsFFTFwd_CToC_32fc(
                reinterpret_cast<const Ipp32fc*>(in_buffer),
                reinterpret_cast<Ipp32fc*>(out_buffer),
                spec_,
                work_buf_
            );
        }

    private:
        ScopedIPPDispatcher state_guard_{kSimd};

        int order_;
        int n_;
        IppsFFTSpec_C_32fc* spec_{nullptr};
        Ipp8u* spec_mem_{nullptr};
        Ipp8u* work_buf_{nullptr};
    };

    template <IPPSimdLevel kSimd>
    class IPPFFT<double, kSimd> final {
        using C = std::complex<double>;

    public:
        explicit IPPFFT(const int order) : order_(order), n_(1 << order) {
            int specSize = 0, initSize = 0, bufSize = 0;

            IppStatus status = ippsFFTGetSize_C_64fc(order_, IPP_FFT_DIV_INV_BY_N, ippAlgHintNone, &specSize, &initSize, &bufSize);
            if (status != ippStsNoErr) {
                throw std::runtime_error("ippsFFTGetSize_C_64fc failed");
            }

            spec_mem_ = ippsMalloc_8u(specSize);
            work_buf_ = bufSize > 0 ? ippsMalloc_8u(bufSize) : nullptr;
            Ipp8u* init_buf = initSize > 0 ? ippsMalloc_8u(initSize) : nullptr;

            status = ippsFFTInit_C_64fc(&spec_, order_, IPP_FFT_DIV_INV_BY_N, ippAlgHintNone, spec_mem_, init_buf);

            if (init_buf) {
                ippsFree(init_buf);
            }

            if (status != ippStsNoErr) {
                if (spec_mem_) ippsFree(spec_mem_);
                if (work_buf_) ippsFree(work_buf_);
                throw std::runtime_error("ippsFFTInit_C_64fc failed");
            }
        }

        ~IPPFFT() {
            if (spec_mem_) {
                ippsFree(spec_mem_);
            }
            if (work_buf_) {
                ippsFree(work_buf_);
            }
        }

        void forward(C* in_buffer, C* out_buffer) {
            ippsFFTFwd_CToC_64fc(
                reinterpret_cast<const Ipp64fc*>(in_buffer),
                reinterpret_cast<Ipp64fc*>(out_buffer),
                spec_,
                work_buf_
            );
        }

    private:
        ScopedIPPDispatcher state_guard_{kSimd};

        int order_;
        int n_;
        IppsFFTSpec_C_64fc* spec_{nullptr};
        Ipp8u* spec_mem_{nullptr};
        Ipp8u* work_buf_{nullptr};
    };
}
