#include "benchmark_include.hpp"

template <class T>
__attribute__((always_inline)) inline void DoNotOptimize(const T& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

using ComplexVec = std::vector<C, AlignedAllocator<std::complex<float>>>;

extern "C" __attribute__((noinline)) void target_fft_execution(
    void* fft_obj, void* in_vec, void* out_vec, int iterations) 
{
    auto& fft = *static_cast<FFTClass*>(fft_obj);
    auto& in = *static_cast<ComplexVec*>(in_vec);
    auto& out = *static_cast<ComplexVec*>(out_vec);

    for (int i = 0; i < iterations; ++i) {
        fft.forward(in, out);
        DoNotOptimize(out.data());
        fft.forward(out, in);
        DoNotOptimize(in.data());
    }
}

int main(const int, char** argv) {
    const int order = std::stoi(argv[1]);
    const int iterations = std::stoi(argv[2]);
    const size_t n = static_cast<size_t>(1) << order;

    ComplexVec in(n);
    generate_random_data(in);
    ComplexVec out(n);

    FFTClass fft(order);

    fft.forward(in, out);
    DoNotOptimize(out.data());
    fft.forward(out, in);
    DoNotOptimize(in.data());

    target_fft_execution(&fft, &in, &out, iterations);

    std::cout << "Done: order=" << order
              << " iterations=" << iterations
              << " n=" << n << std::endl;

    return 0;
}