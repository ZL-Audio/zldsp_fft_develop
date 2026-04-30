#pragma once

#include <chrono>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <new>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "../zlfft/zldsp_fft_cfft.hpp"

#ifdef USE_DOUBLE
using F = double;
#else
using F = float;
#endif
using C = std::complex<F>;

#if defined(ENABLE_PFFFT)
#include "../pffft_impl/pffft_impl.hpp"
using FFTClass = zlbenchmark::PffftFFT<F>;
#elif defined(ENABLE_VDSP)
#include "../vdsp_impl/vdsp_impl.hpp"
using FFTClass = zlbenchmark::VDSPFFT<F>;
#elif defined(ENABLE_VDSP_STRIDE_2)
#include "../vdsp_impl/vdsp_impl.hpp"
using FFTClass = zlbenchmark::VDSPFFT<F>;
#elif defined(ENABLE_IPP)
#include "../ipp_impl/ipp_impl.hpp"
#if defined(IPP_USE_AVX2)
using FFTClass = zlbenchmark::IPPFFT<F, zlbenchmark::IPPSimdLevel::AVX2>;
#else
using FFTClass = zlbenchmark::IPPFFT<F, zlbenchmark::IPPSimdLevel::SSE42>;
#endif
#elif defined(ENABLE_KFR)
#include "../kfr_impl/kfr_impl.hpp"
using FFTClass = zlbenchmark::KFRFFT<F>;
#elif defined(ENABLE_FFTW3)
#include "../fftw3_impl/fftw3_impl.hpp"
using FFTClass = zlbenchmark::FFTW3FFT<F>;
#elif defined(ENABLE_FFTW3_ESTIMATE)
#include "../fftw3_impl/fftw3_impl.hpp"
using FFTClass = zlbenchmark::FFTW3FFT<F, FFTW_ESTIMATE>;
#elif defined(ENABLE_ZLDSP)
#include "../zlfft/zldsp_fft_cfft.hpp"
using FFTClass = zldsp::fft::CFFT<F>;
#endif

inline void generate_random_data(std::span<C> data) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<F> dist(static_cast<F>(-1.0), static_cast<F>(1.0));
    for (auto& x : data) {
        x = C(dist(gen), dist(gen));
    }
}

template <typename T, std::size_t Align = 64>
struct AlignedAllocator {
    using value_type = T;
    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Align>;
    };

    AlignedAllocator() = default;

    template <class U>
    constexpr AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_alloc();

        void* ptr = ::operator new(n * sizeof(T), std::align_val_t(Align));
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept { ::operator delete(p, std::align_val_t(Align)); }

    bool operator==(const AlignedAllocator&) const { return true; }
    bool operator!=(const AlignedAllocator&) const { return false; }
};
