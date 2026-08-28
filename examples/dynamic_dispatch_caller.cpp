#include <array>
#include <complex>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include "dynamic_dispatch_wrapper.hpp"

template <typename F>
void run(const std::size_t order) {
    const std::size_t kSize = std::size_t{1} << order;
    using Complex = std::complex<F>;
    using SoA = std::array<F*, 2>;

    std::vector<Complex> complex_input(kSize, Complex{F{1}, F{0}});
    std::vector<Complex> complex_output(kSize);
    std::vector<F> complex_input_real(kSize, F{1});
    std::vector<F> complex_input_imag(kSize, F{0});
    std::vector<F> complex_output_real(kSize);
    std::vector<F> complex_output_imag(kSize);
    SoA complex_input_soa{complex_input_real.data(), complex_input_imag.data()};
    SoA complex_output_soa{complex_output_real.data(), complex_output_imag.data()};

    auto cfft = zldsp_fft_example::prepare_cfft<F>(order);
    // CFFT forward AoS to AoS
    cfft->forward(complex_input.data(), complex_output.data());
    // CFFT forward AoS to SoA
    cfft->forward(complex_input.data(), complex_output_soa);
    // CFFT forward SoA to AoS
    cfft->forward(complex_input_soa, complex_output.data());
    // CFFT forward SoA to SoA
    cfft->forward(complex_input_soa, complex_output_soa);

    std::vector<Complex> complex_restored(kSize);
    std::vector<F> complex_restored_real(kSize);
    std::vector<F> complex_restored_imag(kSize);
    SoA complex_restored_soa{complex_restored_real.data(), complex_restored_imag.data()};
    // CFFT backward AoS to AoS
    cfft->backward(complex_output.data(), complex_restored.data());
    // CFFT backward AoS to SoA
    cfft->backward(complex_output.data(), complex_restored_soa);
    // CFFT backward SoA to AoS
    cfft->backward(complex_output_soa, complex_restored.data());
    // CFFT backward SoA to SoA
    cfft->backward(complex_output_soa, complex_restored_soa);

    std::vector<F> real_input(kSize, F{1});
    std::vector<Complex> real_output(kSize / 2 + 1);
    std::vector<F> real_output_real(kSize / 2 + 1);
    std::vector<F> real_output_imag(kSize / 2 + 1);
    SoA real_output_soa{real_output_real.data(), real_output_imag.data()};

    auto rfft = zldsp_fft_example::prepare_rfft<F>(order);
    // RFFT forward AoS
    rfft->forward(real_input.data(), real_output.data());
    // RFFT forward SoA
    rfft->forward(real_input.data(), real_output_soa);

    std::vector<F> real_restored(kSize);
    // RFFT backward AoS
    rfft->backward(real_output.data(), real_restored.data());
    // RFFT backward SoA
    rfft->backward(real_output_soa, real_restored.data());
}

int main(const int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <start-order> <end-order>" << std::endl;
        return 2;
    }

    try {
        const std::size_t start_order = std::stoul(argv[1]);
        const std::size_t end_order = std::stoul(argv[2]);
        for (std::size_t order = start_order; order <= end_order; ++order) {
            run<float>(order);
            run<double>(order);
        }
    }
    catch (...) {
        std::cerr << "Invalid order range: " << argv[1] << " " << argv[2] << std::endl;
        return 2;
    }
    return 0;
}
