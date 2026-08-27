#include <complex>
#include <cstddef>
#include <vector>

#include "dynamic_dispatch_wrapper.hpp"

template <typename F>
void run() {
    constexpr std::size_t kOrder = 4;
    constexpr std::size_t kSize = std::size_t{1} << kOrder;
    using Complex = std::complex<F>;

    std::vector<Complex> complex_input(kSize, Complex{F{1}, F{0}});
    std::vector<Complex> complex_output(kSize);
    auto cfft = zldsp_fft_example::prepare_cfft<F>(kOrder);
    cfft->forward(complex_input.data(), complex_output.data());

    std::vector<F> real_input(kSize, F{1});
    std::vector<Complex> real_output(kSize / 2 + 1);
    auto rfft = zldsp_fft_example::prepare_rfft<F>(kOrder);
    rfft->forward(real_input.data(), real_output.data());
}

int main() {
    run<float>();
    run<double>();
    return 0;
}
