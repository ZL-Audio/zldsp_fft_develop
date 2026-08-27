#include <array>
#include <complex>
#include <cstddef>
#include <vector>

#include "zldsp_fft_cfft.hpp"
#include "zldsp_fft_rfft.hpp"

template <typename F>
void run() {
    constexpr std::size_t kOrder = 10;
    constexpr std::size_t kSize = std::size_t{1} << kOrder;
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

    zldsp::fft::CFFT<F> cfft(kOrder);
    // CFFT forward AoS to AoS
    cfft.forward(complex_input.data(), complex_output.data());
    // CFFT forward AoS to SoA
    cfft.forward(complex_input.data(), complex_output_soa);
    // CFFT forward SoA to AoS
    cfft.forward(complex_input_soa, complex_output.data());
    // CFFT forward SoA to SoA
    cfft.forward(complex_input_soa, complex_output_soa);

    std::vector<F> real_input(kSize, F{1});
    std::vector<Complex> real_output(kSize / 2 + 1);
    std::vector<F> real_output_real(kSize / 2 + 1);
    std::vector<F> real_output_imag(kSize / 2 + 1);
    SoA real_output_soa{real_output_real.data(), real_output_imag.data()};

    zldsp::fft::RFFT<F> rfft(kOrder);
    // RFFT forward AoS
    rfft.forward(real_input.data(), real_output.data());
    // RFFT forward SoA
    rfft.forward(real_input.data(), real_output_soa);
}

int main() {
    run<float>();
    run<double>();
    return 0;
}
