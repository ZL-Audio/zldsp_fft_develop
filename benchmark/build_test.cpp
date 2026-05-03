#include <array>
#include <complex>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../zlfft/zldsp_fft_cfft.hpp"
#include "../zlfft/zldsp_fft_rfft.hpp"

#ifdef USE_DOUBLE
using F = double;
#else
using F = float;
#endif
using C = std::complex<F>;

int main(int argc, char** argv) {
    size_t order = 10;
    if (argc > 1) {
        try {
            order = std::stoul(argv[1]);
        }
        catch (...) {
            std::cerr << "Invalid order: " << argv[1] << ". Using default 10." << std::endl;
        }
    }
    const size_t n = 1ULL << order;

    std::vector<F> real_in(n, 1.0f);
    std::vector<F> real_out(n, 0.0f);
    std::vector<C> cplx_in(n, C(1.0f, 1.0f));
    std::vector<C> cplx_out(n, C(0.0f, 0.0f));

    std::vector<F> soa_real(n, 0.0f);
    std::vector<F> soa_imag(n, 0.0f);
    std::array<F*, 2> soa_out = {soa_real.data(), soa_imag.data()};

    std::vector<F> soa_in_real(n, 1.0f);
    std::vector<F> soa_in_imag(n, 1.0f);
    std::array<F*, 2> soa_in = {soa_in_real.data(), soa_in_imag.data()};

    // CFFT
    zldsp::fft::CFFT<F> cfft(order);

    // AoS -> AoS
    cfft.forward(cplx_in.data(), cplx_out.data());
    std::cout << "CFFT AoS Fwd: " << cplx_out[0] << std::endl;
    cfft.backward(cplx_in.data(), cplx_out.data());
    std::cout << "CFFT AoS Bwd: " << cplx_out[0] << std::endl;

    // AoS -> SoA
    cfft.forward(cplx_in.data(), soa_out);
    std::cout << "CFFT AoS->SoA Fwd: " << soa_out[0][0] << std::endl;
    cfft.backward(cplx_in.data(), soa_out);
    std::cout << "CFFT AoS->SoA Bwd: " << soa_out[0][0] << std::endl;

    // SoA -> AoS
    cfft.forward(soa_in, cplx_out.data());
    std::cout << "CFFT SoA->AoS Fwd: " << cplx_out[0] << std::endl;
    cfft.backward(soa_in, cplx_out.data());
    std::cout << "CFFT SoA->AoS Bwd: " << cplx_out[0] << std::endl;

    // SoA -> SoA
    cfft.forward(soa_in, soa_out);
    std::cout << "CFFT SoA->SoA Fwd: " << soa_out[0][0] << std::endl;
    cfft.backward(soa_in, soa_out);
    std::cout << "CFFT SoA->SoA Bwd: " << soa_out[0][0] << std::endl;

    // RFFT
    zldsp::fft::RFFT<F> rfft(order);

    // Real -> AoS
    rfft.forward(real_in.data(), cplx_out.data());
    std::cout << "RFFT Real->AoS Fwd: " << cplx_out[0] << std::endl;

    // AoS -> Real
    rfft.backward(cplx_in.data(), real_out.data());
    std::cout << "RFFT AoS->Real Bwd: " << real_out[0] << std::endl;

    // Real -> SoA
    rfft.forward(real_in.data(), soa_out);
    std::cout << "RFFT Real->SoA Fwd: " << soa_out[0][0] << std::endl;

    // SoA -> Real
    rfft.backward(soa_in, real_out.data());
    std::cout << "RFFT SoA->Real Bwd: " << real_out[0] << std::endl;

    // SqMag
    rfft.forward_sqr_mag(real_in.data(), real_out.data());
    std::cout << "RFFT SqMag: " << real_out[0] << std::endl;

    std::cout << "Build test successful!" << std::endl;
    return 0;
}
