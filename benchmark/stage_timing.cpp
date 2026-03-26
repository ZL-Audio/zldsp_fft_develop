#include <benchmark/benchmark.h>
#include <iostream>
#include <string>
#include "benchmark_include.hpp"

static int g_order = 0;

static void BM_Fft_StageTiming(benchmark::State& state) {
    const size_t num_stages = static_cast<size_t>(state.range(0));
    const size_t n = static_cast<size_t>(1) << g_order;

    std::vector<C, AlignedAllocator<std::complex<float>>> in(n);
    generate_random_data(in);
    std::vector<C, AlignedAllocator<std::complex<float>>> out(n);

    FFTClass fft(g_order);

    for (int i = 0; i < 3; ++i) {
        fft.forward_stages(in, out, num_stages);
        fft.forward_stages(out, in, num_stages);
    }
    benchmark::ClobberMemory();

    for (auto _ : state) {
        fft.forward_stages(in, out, num_stages);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
        fft.forward_stages(out, in, num_stages);
        benchmark::DoNotOptimize(in.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * n);
}

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <order> [google-benchmark-flags]" << std::endl;
        return 1;
    }

    g_order = std::stoi(argv[1]);

    FFTClass fft(g_order);
    const size_t total_stages = fft.num_stages();

    std::cerr << "Order: " << g_order << ", Total stages: " << total_stages << std::endl;

    benchmark::RegisterBenchmark("BM_Fft_StageTiming", BM_Fft_StageTiming)
        ->DenseRange(1, static_cast<int>(total_stages), 1)
        ->Unit(benchmark::kMicrosecond);

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
