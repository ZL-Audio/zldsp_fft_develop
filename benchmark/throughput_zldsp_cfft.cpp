#include <benchmark/benchmark.h>
#include <array>
#include <complex>
#include <iostream>
#include <limits>
#include <new>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "../zldsp_fft/zldsp_fft_cfft.hpp"

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

inline void generate_random_data_c(std::span<C> data) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<F> dist(static_cast<F>(-1.0), static_cast<F>(1.0));
    for (auto& x : data) {
        x = C(dist(gen), dist(gen));
    }
}

inline void generate_random_data_f(std::span<F> data) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<F> dist(static_cast<F>(-1.0), static_cast<F>(1.0));
    for (auto& x : data) {
        x = dist(gen);
    }
}

static void BM_ZLDSP_CFFT_AoS_Forward(benchmark::State& state) {
    const int order = state.range(0);
    const size_t n = static_cast<size_t>(1) << order;

    std::vector<C, AlignedAllocator<C>> in(n);
    generate_random_data_c(in);
    std::vector<C, AlignedAllocator<C>> out(n);

    zldsp::fft::CFFT<F> fft(order);

    for (int i = 0; i < 3; ++i) {
        fft.forward(in.data(), out.data());
        fft.forward(out.data(), in.data());
    }
    benchmark::ClobberMemory();

    for (auto _ : state) {
        fft.forward(in.data(), out.data());
        benchmark::DoNotOptimize(out.data()[0]);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
        fft.forward(out.data(), in.data());
        benchmark::DoNotOptimize(in.data()[0]);
        benchmark::DoNotOptimize(in.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * 2 * n);
}

static void BM_ZLDSP_CFFT_SoA_Forward(benchmark::State& state) {
    const int order = state.range(0);
    const size_t n = static_cast<size_t>(1) << order;

    std::vector<F, AlignedAllocator<F>> in_r(n), in_i(n);
    generate_random_data_f(in_r);
    generate_random_data_f(in_i);
    std::vector<F, AlignedAllocator<F>> out_r(n), out_i(n);

    std::array<F*, 2> in_soa{in_r.data(), in_i.data()};
    std::array<F*, 2> out_soa{out_r.data(), out_i.data()};

    zldsp::fft::CFFT<F> fft(order);

    for (int i = 0; i < 3; ++i) {
        fft.forward(in_soa, out_soa);
        fft.forward(out_soa, in_soa);
    }
    benchmark::ClobberMemory();

    for (auto _ : state) {
        fft.forward(in_soa, out_soa);
        benchmark::DoNotOptimize(out_r.data()[0]);
        benchmark::DoNotOptimize(out_i.data()[0]);
        benchmark::ClobberMemory();
        fft.forward(out_soa, in_soa);
        benchmark::DoNotOptimize(in_r.data()[0]);
        benchmark::DoNotOptimize(in_i.data()[0]);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * 2 * n);
}

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <n0> <n1> [google-benchmark-flags]" << std::endl;
        return 1;
    }

    int n0 = std::stoi(argv[1]);
    int n1 = std::stoi(argv[2]);

    benchmark::RegisterBenchmark("AoS to AoS CFFT forward", BM_ZLDSP_CFFT_AoS_Forward)
        ->DenseRange(n0, n1, 1)
        ->Unit(benchmark::kMicrosecond);

    benchmark::RegisterBenchmark("SoA to SoA CFFT forward", BM_ZLDSP_CFFT_SoA_Forward)
        ->DenseRange(n0, n1, 1)
        ->Unit(benchmark::kMicrosecond);

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
