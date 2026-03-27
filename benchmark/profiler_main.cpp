#include "benchmark_include.hpp"

template <class T>
__attribute__((always_inline)) inline void DoNotOptimize(const T& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

int main(const int, char** argv) {
    const int order = std::stoi(argv[1]);
    const int iterations = std::stoi(argv[2]);
    const size_t n = static_cast<size_t>(1) << order;

    std::vector<C, AlignedAllocator<std::complex<float>>> in(n);
    generate_random_data(in);
    std::vector<C, AlignedAllocator<std::complex<float>>> out(n);

    FFTClass fft(order);

    // Warmup
    fft.forward(in, out);
    DoNotOptimize(out.data());
    fft.forward(out, in);
    DoNotOptimize(in.data());

    // Timed loop
    for (int i = 0; i < iterations; ++i) {
        fft.forward(in, out);
        DoNotOptimize(out.data());
        fft.forward(out, in);
        DoNotOptimize(in.data());
    }

    std::cout << "Done: order=" << order
              << " iterations=" << iterations
              << " n=" << n << std::endl;

    return 0;
}
