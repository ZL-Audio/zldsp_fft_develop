#include <complex>
#include <cstddef>
#include <vector>

#include "zldsp_fft_cfft.hpp"
#include "zldsp_fft_rfft.hpp"

template <typename F>
void run() {
    constexpr std::size_t kOrder = 4;
    constexpr std::size_t kSize = std::size_t{1} << kOrder;
    using Complex = std::complex<F>;

    std::vector<Complex> complex_input(kSize, Complex{F{1}, F{0}});
    std::vector<Complex> complex_output(kSize);
    zldsp::fft::CFFT<F> cfft(kOrder);
    cfft.forward(complex_input.data(), complex_output.data());

    std::vector<F> real_input(kSize, F{1});
    std::vector<Complex> real_output(kSize / 2 + 1);
    zldsp::fft::RFFT<F> rfft(kOrder);
    rfft.forward(real_input.data(), real_output.data());
}

int main() {
    run<float>();
    run<double>();
    return 0;
}
