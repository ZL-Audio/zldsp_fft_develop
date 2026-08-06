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

#include "../zldsp_fft/zldsp_fft_rfft.hpp"

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

inline void generate_random_data_f(std::span<F> data) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<F> dist(static_cast<F>(-1.0), static_cast<F>(1.0));
    for (auto& x : data) {
        x = dist(gen);
    }
}

static void BM_ZLDSP_RFFT_Real_To_AoS_Forward(benchmark::State& state) {
    const int order = state.range(0);
    const size_t n = static_cast<size_t>(1) << order;
    const size_t out_n = n / 2 + 1;

    std::vector<F, AlignedAllocator<F>> in(n);
    generate_random_data_f(in);
    std::vector<C, AlignedAllocator<C>> out(out_n);

    zldsp::fft::RFFT<F> rfft(order);

    for (int i = 0; i < 3; ++i) {
        rfft.forward(in.data(), out.data());
    }
    benchmark::ClobberMemory();

    for (auto _ : state) {
        rfft.forward(in.data(), out.data());
        benchmark::DoNotOptimize(out.data()[0]);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * n);
}

static void BM_ZLDSP_RFFT_Real_To_SoA_Backward(benchmark::State& state) {
    const int order = state.range(0);
    const size_t n = static_cast<size_t>(1) << order;
    const size_t in_n = n / 2 + 1;

    std::vector<F, AlignedAllocator<F>> in_r(in_n), in_i(in_n);
    generate_random_data_f(in_r);
    generate_random_data_f(in_i);
    in_i[0] = static_cast<F>(0.0);
    in_i[in_n - 1] = static_cast<F>(0.0);

    std::array<F*, 2> in_soa{in_r.data(), in_i.data()};
    std::vector<F, AlignedAllocator<F>> out(n);

    zldsp::fft::RFFT<F> rfft(order);

    for (int i = 0; i < 3; ++i) {
        rfft.backward(in_soa, out.data());
    }
    benchmark::ClobberMemory();

    for (auto _ : state) {
        rfft.backward(in_soa, out.data());
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

    benchmark::RegisterBenchmark("real to AoS RFFT forward", BM_ZLDSP_RFFT_Real_To_AoS_Forward)
        ->DenseRange(n0, n1, 1)
        ->Unit(benchmark::kMicrosecond);

    benchmark::RegisterBenchmark("real to SoA RFFT backward", BM_ZLDSP_RFFT_Real_To_SoA_Backward)
        ->DenseRange(n0, n1, 1)
        ->Unit(benchmark::kMicrosecond);

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
