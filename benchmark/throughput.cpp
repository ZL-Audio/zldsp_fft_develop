#include <benchmark/benchmark.h>
#include <iostream>
#include <string>
#include "benchmark_include.hpp"

static void BM_Fft_Throughput(benchmark::State& state) {
    const int order = state.range(0);
    const size_t n = static_cast<size_t>(1) << order;

    std::vector<C, AlignedAllocator<std::complex<F>>> in(n);
    generate_random_data(in);
    std::vector<C, AlignedAllocator<std::complex<F>>> out(n);

    FFTClass fft(order);

    for (int i = 0; i < 3; ++i) {
        fft.forward(in, out);
        fft.forward(out, in);
    }
    benchmark::ClobberMemory();

    for (auto _ : state) {
        fft.forward(in, out);
        benchmark::DoNotOptimize(out.data()[0]);
        benchmark::DoNotOptimize(out.data());
        benchmark::ClobberMemory();
        fft.forward(out, in);
        benchmark::DoNotOptimize(in.data()[0]);
        benchmark::DoNotOptimize(in.data());
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

    benchmark::RegisterBenchmark("BM_Fft_Throughput", BM_Fft_Throughput)
        ->DenseRange(n0, n1, 1)
        ->Unit(benchmark::kMicrosecond);

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
