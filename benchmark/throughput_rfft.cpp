#include <benchmark/benchmark.h>
#include <iostream>
#include <string>
#include <complex>
#include <vector>
#include <random>
#include <span>
#include <limits>
#include <new>

#ifdef USE_DOUBLE
using F = double;
#else
using F = float;
#endif
using C = std::complex<F>;

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

#if defined(ENABLE_PFFFT)
#include "../pffft_impl/pffft_impl.hpp"
using FFTClass = zlbenchmark::PffftRFFT<F>;
#elif defined(ENABLE_KFR)
#include "../kfr_impl/kfr_impl.hpp"
using FFTClass = zlbenchmark::KFRRFFT<F>;
#elif defined(ENABLE_VDSP)
#include "../vdsp_impl/vdsp_impl.hpp"
using FFTClass = zlbenchmark::VDSPRFFT<F>;
#elif defined(ENABLE_IPP)
#include "../ipp_impl/ipp_impl.hpp"
#if defined(IPP_USE_AVX2)
using FFTClass = zlbenchmark::IPPRFFT<F, zlbenchmark::IPPSimdLevel::AVX2>;
#else
using FFTClass = zlbenchmark::IPPRFFT<F, zlbenchmark::IPPSimdLevel::SSE42>;
#endif
#elif defined(ENABLE_FFTW3)
#include "../fftw3_impl/fftw3_impl.hpp"
using FFTClass = zlbenchmark::FFTW3RFFT<F>;
#elif defined(ENABLE_FFTW3_ESTIMATE)
#include "../fftw3_impl/fftw3_impl.hpp"
using FFTClass = zlbenchmark::FFTW3RFFT<F, FFTW_ESTIMATE>;
#elif defined(ENABLE_ZLDSP)
#include "../zlfft/zldsp_fft_rfft.hpp"
using FFTClass = zldsp::fft::RFFT<F>;
#endif

inline void generate_random_data_f(std::span<F> data) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<F> dist(static_cast<F>(-1.0), static_cast<F>(1.0));
    for (auto& x : data) {
        x = dist(gen);
    }
}

static void BM_Rfft_Throughput(benchmark::State& state) {
    const int order = state.range(0);
    const size_t n = static_cast<size_t>(1) << order;
    const size_t out_n = n / 2 + 1;

    std::vector<F, AlignedAllocator<F>> in(n);
    generate_random_data_f(in);
    std::vector<C, AlignedAllocator<C>> out(out_n);

    // Some implementations (e.g. PFFFT) may not support orders < 5.
    // If we reach here, we allocate and run but their logic handles it gracefully or crashes.
    // PffftRFFT handles missing setup_ safely after previous patches.
    FFTClass fft(order);

    for (int i = 0; i < 3; ++i) {
        fft.forward(in.data(), out.data());
    }
    benchmark::ClobberMemory();

    for (auto _ : state) {
        fft.forward(in.data(), out.data());
        benchmark::DoNotOptimize(out.data()[0]);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * n);
}

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <n0> <n1> [google-benchmark-flags]" << std::endl;
        return 1;
    }

    int n0 = std::stoi(argv[1]);
    int n1 = std::stoi(argv[2]);

    benchmark::RegisterBenchmark("BM_Rfft_Throughput", BM_Rfft_Throughput)
        ->DenseRange(n0, n1, 1)
        ->Unit(benchmark::kMicrosecond);

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
