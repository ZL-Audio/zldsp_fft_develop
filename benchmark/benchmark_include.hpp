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

#include "../source/naive_stockham_radix2.hpp"
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
#elif defined(ENABLE_NAIVE_COOLEY_RADIX2)
#include "../source/naive_cooley_radix2.hpp"
using FFTClass = zlfft::NaiveCooleyRadix2<F>;
#elif defined(ENABLE_STOCKHAM_RADIX2_KERNEL24)
#include "../source/stockham_radix2_kernel24.hpp"
using FFTClass = zlfft::StockhamRadix2Kernel24<F>;
#elif defined(ENABLE_NAIVE_STOCKHAM_RADIX4)
#include "../source/naive_stockham_radix4.hpp"
using FFTClass = zlfft::NaiveStockhamRadix4<F>;
#elif defined(ENABLE_STOCKHAM_RADIX2_KERNEL248)
#include "../source/stockham_radix2_kernel248.hpp"
using FFTClass = zlfft::StockhamRadix2Kernel248<F>;
#elif defined(ENABLE_SIMD_STOCKHAM_RADIX2)
#include "../source/simd_stockham_radix2.hpp"
using FFTClass = zlfft::SIMDStockhamRadix2<F>;
#elif defined(ENABLE_SIMD_STOCKHAM_RADIX2_KERNEL1)
#include "../source/simd_stockham_radix2_kernel1.hpp"
using FFTClass = zlfft::SIMDStockhamRadix2Kernel1<F>;
#elif defined(ENABLE_SIMD_STOCKHAM_RADIX2_KERNEL2)
#include "../source/simd_stockham_radix2_kernel2.hpp"
using FFTClass = zlfft::SIMDStockhamRadix2Kernel2<F>;
#elif defined(ENABLE_SIMD_STOCKHAM_RADIX2_KERNEL24)
#include "../source/simd_stockham_radix2_kernel24.hpp"
using FFTClass = zlfft::SIMDStockhamRadix2Kernel24<F>;
#elif defined(ENABLE_SIMD_STOCKHAM_RADIX2_KERNEL248)
#include "../source/simd_stockham_radix2_kernel248.hpp"
using FFTClass = zlfft::SIMDStockhamRadix2Kernel248<F>;
#elif defined(ENABLE_SIMD_STOCKHAM_RADIX4)
#include "../source/simd_stockham_radix4.hpp"
using FFTClass = zlfft::SIMDStockhamRadix4<F>;
#elif defined(ENABLE_SIMD_STOCKHAM_RADIX4_SOA)
#include "../source/simd_stockham_radix4_soa.hpp"
using FFTClass = zlfft::SIMDStockhamRadix4SOA<F>;
#elif defined(ENABLE_SIMD_STOCKHAM_RADIX8)
#include "../source/simd_stockham_radix8.hpp"
using FFTClass = zlfft::SIMDStockhamRadix8<F>;
#elif defined(ENABLE_SIMD_STOCKHAM_RADIX4_OPT1)
#include "../source/simd_stockham_radix4_opt1.hpp"
using FFTClass = zlfft::SIMDStockhamRadix4OPT1<F>;
#elif defined(ENABLE_SIMD_STOCKHAM_RADIX4_SOA_KERNEL4)
#include "../source/simd_stockham_radix4_soa_kernel4.hpp"
using FFTClass = zlfft::SIMDStockhamRadix4SOAKernel4<F>;
#elif defined(ENABLE_SIMD_STOCKHAM_RADIX4_SOA_KERNEL4_OPT1)
#include "../source/simd_stockham_radix4_soa_kernel4_opt1.hpp"
using FFTClass = zlfft::SIMDStockhamRadix4SOAKernel4OPT1<F>;
#elif defined(ENABLE_SIMD_LOW_ORDER)
#include "../source/simd_low_order.hpp"
using FFTClass = zlfft::SIMDLowOrder<F>;
#elif defined(ENABLE_SIMD_LOW_ORDER_OPT1)
#include "../zlfft_impl/simd_low_order_opt1.hpp"
using FFTClass = zlfft::SIMDLowOrderOPT1<F>;
#elif defined(ENABLE_SIMD_LOW_ORDER_OPT2)
#include "../zlfft_impl/simd_low_order_opt2.hpp"
using FFTClass = zlfft::SIMDLowOrderOPT2<F>;
#elif defined(ENABLE_SIMD_LOW_ORDER_AOSOA1)
#include "../zlfft_impl/simd_low_order_aosoa1.hpp"
using FFTClass = zlfft::SIMDLowOrderAOSOA1<F>;
#elif defined(ENABLE_SIMD_LOW_ORDER_AOSOA2)
#include "../zlfft_impl/simd_low_order_aosoa2.hpp"
using FFTClass = zlfft::SIMDLowOrderAOSOA2<F>;
#elif defined(ENABLE_SIMD_LOW_ORDER_AOSOA3)
#include "../zlfft_impl/simd_low_order_aosoa3.hpp"
using FFTClass = zlfft::SIMDLowOrderAOSOA3<F>;
#elif defined(ENABLE_SIMD_LOW_ORDER_AOSOA4)
#include "../zlfft_impl/simd_low_order_aosoa4.hpp"
using FFTClass = zlfft::SIMDLowOrderAOSOA4<F>;
#elif defined(ENABLE_HIGH_ORDER_AOSOA1)
#include "../zlfft_impl/simd_high_order_aosoa1.hpp"
using FFTClass = zlfft::SIMDHighOrderAOSOA1<F>;
#elif defined(ENABLE_HIGH_ORDER_AOSOA2)
#include "../zlfft_impl/simd_high_order_aosoa2.hpp"
using FFTClass = zlfft::SIMDHighOrderAOSOA2<F>;
#elif defined(ENABLE_SIMD_LOW_ORDER_AOSOA5)
#include "../zlfft_impl/simd_low_order_aosoa5.hpp"
using FFTClass = zlfft::SIMDLowOrderAOSOA5<F>;
#elif defined(ENABLE_SIMD_LOW_ORDER_AOSOA6)
#include "../zlfft_impl/simd_low_order_aosoa6.hpp"
using FFTClass = zlfft::SIMDLowOrderAOSOA6<F>;
#elif defined(ENABLE_SIMD_LOW_ORDER_AOSOA7)
#include "../zlfft_impl/simd_low_order_aosoa7.hpp"
using FFTClass = zlfft::SIMDLowOrderAOSOA7<F>;
#elif defined(ENABLE_SIMD_LOW_ORDER_AOSOA8)
#include "../zlfft_impl/simd_low_order_aosoa8.hpp"
using FFTClass = zlfft::SIMDLowOrderAOSOA8<F>;
#elif defined(ENABLE_SIMD_LOW_ORDER_AOSOA9)
#include "../zlfft_impl/simd_low_order_aosoa9.hpp"
using FFTClass = zlfft::SIMDLowOrderAOSOA9<F>;
#elif defined(ENABLE_HYBRID_AOSOA1)
#include "../zlfft_impl/hybrid_aosoa1.hpp"
using FFTClass = zlfft::HybridAoSoA1<F>;
#elif defined(ENABLE_HYBRID_AOSOA2)
#include "../zlfft_impl/hybrid_aosoa2.hpp"
using FFTClass = zlfft::HybridAoSoA2<F>;
#elif defined(ENABLE_ZLDSP)
#include "../zlfft/zldsp_fft_cfft.hpp"
using FFTClass = zldsp::fft::CFFT<F>;
#else
using FFTClass = zlfft::NaiveStockhamRadix2<F>;
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
