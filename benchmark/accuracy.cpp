#include "benchmark_include.hpp"

F calculate_mse(const std::span<C> ref, const std::span<C> test) {
    double mse = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const auto r_diff = ref[i].real() - test[i].real();
        const auto i_diff = ref[i].imag() - test[i].imag();
        mse += static_cast<double>(r_diff * r_diff + i_diff * i_diff);
    }
    return static_cast<F>(mse / static_cast<double>(ref.size()));
}

int main(const int, char** argv) {
    const int order = std::stoi(argv[1]);
    const size_t n = static_cast<size_t>(1) << order;

    std::vector<C, AlignedAllocator<std::complex<F>>> in(n);
    std::vector<C, AlignedAllocator<std::complex<F>>> in_copy;
    generate_random_data(in);
    std::vector<C, AlignedAllocator<std::complex<F>>> out_ref(n);
    std::vector<C, AlignedAllocator<std::complex<F>>> out_test(n);

    zlfft::NaiveStockhamRadix2<F> ref_fft(order);
    in_copy = in;
    ref_fft.forward(in_copy, out_ref);

    FFTClass test_fft(order);
    in_copy = in;
    test_fft.forward(in_copy, out_test);

    const double mse = calculate_mse(out_ref, out_test);
    std::cout << std::scientific << mse << std::endl;
    return 0;
}
