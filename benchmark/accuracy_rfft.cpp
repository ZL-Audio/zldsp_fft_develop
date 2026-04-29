#include <cmath>
#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <limits>
#include <new>
#include <span>
#include <complex>

#ifdef USE_DOUBLE
using F = double;
#else
using F = float;
#endif
using C = std::complex<F>;

template <typename T, std::size_t Align = 64>
struct AlignedAllocator {
    using value_type = T;
    template <class U>
    struct rebind {
        using other = AlignedAllocator<U, Align>;
    };

    AlignedAllocator() = default;

    template <class U>
    constexpr AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept {}

    T* allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_alloc();

        void* ptr = ::operator new(n * sizeof(T), std::align_val_t(Align));
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept { ::operator delete(p, std::align_val_t(Align)); }

    bool operator==(const AlignedAllocator&) const { return true; }
    bool operator!=(const AlignedAllocator&) const { return false; }
};

#if defined(ENABLE_PFFFT)
#include "../pffft_impl/pffft_impl.hpp"
using FFTClass = zlbenchmark::PffftRFFT<F>;
#elif defined(ENABLE_KFR)
#include "../kfr_impl/kfr_impl.hpp"
using FFTClass = zlbenchmark::KFRRFFT<F>;
#elif defined(ENABLE_FFTW3)
#include "../fftw3_impl/fftw3_impl.hpp"
using FFTClass = zlbenchmark::FFTW3RFFT<F>;
#elif defined(ENABLE_ZLDSP)
#include "../zlfft/zldsp_fft_rfft.hpp"
using FFTClass = zldsp::fft::RFFT<F>;
#else
#include "../fftw3_impl/fftw3_impl.hpp"
using FFTClass = zlbenchmark::FFTW3RFFT<F>;
#endif

void generate_random_data_f(std::span<F> data) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<F> dist(static_cast<F>(-1.0), static_cast<F>(1.0));
    for (auto& x : data) {
        x = dist(gen);
    }
}

void generate_random_data_c_rfft(std::span<C> data) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<F> dist(static_cast<F>(-1.0), static_cast<F>(1.0));
    for (size_t i = 0; i < data.size(); ++i) {
        if (i == 0 || i == data.size() - 1) {
            data[i] = C(dist(gen), 0.0);
        } else {
            data[i] = C(dist(gen), dist(gen));
        }
    }
}

double calculate_mse_aos(const std::span<C> ref, const std::span<C> test) {
    double mse = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const auto r_diff = ref[i].real() - test[i].real();
        const auto i_diff = ref[i].imag() - test[i].imag();
        mse += static_cast<double>(r_diff * r_diff + i_diff * i_diff);
    }
    return mse / static_cast<double>(ref.size());
}

double calculate_mse_soa(const std::span<C> ref, const std::array<std::span<F>, 2> test) {
    double mse = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const auto r_diff = ref[i].real() - test[0][i];
        const auto i_diff = ref[i].imag() - test[1][i];
        mse += static_cast<double>(r_diff * r_diff + i_diff * i_diff);
    }
    return mse / static_cast<double>(ref.size());
}

double calculate_mse_soa_soa(const std::array<std::span<F>, 2> ref, const std::array<std::span<F>, 2> test) {
    double mse = 0.0;
    for (size_t i = 0; i < ref[0].size(); ++i) {
        const auto r_diff = ref[0][i] - test[0][i];
        const auto i_diff = ref[1][i] - test[1][i];
        mse += static_cast<double>(r_diff * r_diff + i_diff * i_diff);
    }
    return mse / static_cast<double>(ref[0].size());
}

double calculate_mse_real(const std::span<F> ref, const std::span<F> test) {
    double mse = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const auto diff = ref[i] - test[i];
        mse += static_cast<double>(diff * diff);
    }
    return mse / static_cast<double>(ref.size());
}

int main(const int argc, char** argv) {
    if (argc < 2)
        return 1;
    const int order = std::stoi(argv[1]);
    const size_t size = static_cast<size_t>(1) << order;
    const size_t out_size = size / 2 + 1;

    FFTClass rfft(order);

    std::vector<F, AlignedAllocator<F>> in_f(size);
    generate_random_data_f(in_f);

    std::vector<C, AlignedAllocator<C>> out_aos(out_size);
    std::vector<F, AlignedAllocator<F>> out_soa_r(out_size);
    std::vector<F, AlignedAllocator<F>> out_soa_i(out_size);
    std::array<std::span<F>, 2> out_soa = {std::span<F>(out_soa_r), std::span<F>(out_soa_i)};

    std::vector<F, AlignedAllocator<F>> in_f_copy = in_f;
    rfft.forward(in_f_copy, out_aos);

#if defined(ENABLE_ZLDSP)
    in_f_copy = in_f;
    rfft.forward(in_f_copy, out_soa);
    double max_mse_forward = calculate_mse_soa(out_aos, out_soa);
#else
    double max_mse_forward = 0.0;
#endif

    std::vector<C, AlignedAllocator<C>> in_c(out_size);
    generate_random_data_c_rfft(in_c);

    std::vector<F, AlignedAllocator<F>> in_soa_r(out_size);
    std::vector<F, AlignedAllocator<F>> in_soa_i(out_size);
    for (size_t i = 0; i < out_size; ++i) {
        in_soa_r[i] = in_c[i].real();
        in_soa_i[i] = in_c[i].imag();
    }
    std::array<std::span<F>, 2> in_c_soa = {std::span<F>(in_soa_r), std::span<F>(in_soa_i)};

    std::vector<F, AlignedAllocator<F>> back_aos(size);
    std::vector<F, AlignedAllocator<F>> back_soa(size);

    std::vector<C, AlignedAllocator<C>> in_c_copy = in_c;
    rfft.backward(in_c_copy, back_aos);

#if defined(ENABLE_ZLDSP)
    std::vector<F, AlignedAllocator<F>> in_soa_r_copy = in_soa_r;
    std::vector<F, AlignedAllocator<F>> in_soa_i_copy = in_soa_i;
    std::array<std::span<const F>, 2> in_c_soa_copy = {std::span<const F>(in_soa_r_copy), std::span<const F>(in_soa_i_copy)};
    rfft.backward(in_c_soa_copy, back_soa);
    double max_mse_backward = calculate_mse_real(back_aos, back_soa);
#else
    double max_mse_backward = 0.0;
#endif

    std::vector<F, AlignedAllocator<F>> fwd_then_back(size);
    std::vector<C, AlignedAllocator<C>> out_aos_copy = out_aos;
    rfft.backward(out_aos_copy, fwd_then_back);

    const F n_inv = static_cast<F>(2.0 / static_cast<double>(size));
    std::vector<F, AlignedAllocator<F>> normalized_out(size);
    for (size_t i = 0; i < size; ++i) {
        normalized_out[i] = fwd_then_back[i] * n_inv;
    }
    double mse_identity = calculate_mse_real(in_f, normalized_out);

    std::cout << std::scientific << max_mse_forward << " " << max_mse_backward << " " << mse_identity << std::endl;
    return 0;
}
