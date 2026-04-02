#include <algorithm>
#include <array>
#include <complex>
#include <iostream>
#include <new>
#include <random>
#include <span>
#include <vector>

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

using AlignedVec = std::vector<C, AlignedAllocator<C>>;

#include "../fftw3_impl/fftw3_impl.hpp"
#include "../kfr_impl/kfr_impl.hpp"
#include "../pffft_impl/pffft_impl.hpp"
#include "../zlfft/zldsp_fft_cfft.hpp"

static constexpr size_t NUM_ALGOS = 4;
static const char* ALGO_NAMES[NUM_ALGOS] = {"kfr", "fftw3", "zldsp", "pffft"};

void generate_random_data(std::span<C> data, std::mt19937& gen) {
    std::uniform_real_distribution<F> dist(static_cast<F>(-1.0), static_cast<F>(1.0));
    for (auto& x : data) {
        x = C(dist(gen), dist(gen));
    }
}

void compute_median(const std::array<AlignedVec, NUM_ALGOS>& results, AlignedVec& median_out) {
    const size_t n = results[0].size();
    median_out.resize(n);

    std::array<F, NUM_ALGOS> reals{};
    std::array<F, NUM_ALGOS> imags{};

    for (size_t i = 0; i < n; ++i) {
        for (size_t a = 0; a < NUM_ALGOS; ++a) {
            reals[a] = results[a][i].real();
            imags[a] = results[a][i].imag();
        }
        std::sort(reals.begin(), reals.end());
        std::sort(imags.begin(), imags.end());

        // Median of 4 values = average of middle two
        F med_r = (reals[1] + reals[2]) / static_cast<F>(2.0);
        F med_i = (imags[1] + imags[2]) / static_cast<F>(2.0);
        median_out[i] = C(med_r, med_i);
    }
}

double calculate_mse(const std::span<C> ref, const std::span<C> test) {
    double mse = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const auto r_diff = ref[i].real() - test[i].real();
        const auto i_diff = ref[i].imag() - test[i].imag();
        mse += static_cast<double>(r_diff * r_diff + i_diff * i_diff);
    }
    return mse / static_cast<double>(ref.size());
}

int main(const int, char** argv) {
    const int order = std::stoi(argv[1]);
    const int reps = std::stoi(argv[2]);
    const unsigned seed = std::stoul(argv[3]);

    const size_t n = static_cast<size_t>(1) << order;

    zlbenchmark::KFRFFT<F> kfr_fft(order);
    zlbenchmark::FFTW3FFT<F> fftw3_fft(order);
    zldsp::fft::CFFT<F> zldsp_fft(order);
    zlbenchmark::PffftFFT<F> pffft_fft(order);

    AlignedVec in(n);
    AlignedVec in_copy(n);
    std::array<AlignedVec, NUM_ALGOS> out;
    for (auto& o : out)
        o.resize(n);

    AlignedVec median_ref(n);

    std::array<double, NUM_ALGOS> mse_accum{};

    std::mt19937 gen(seed);

    for (int rep = 0; rep < reps; ++rep) {
        generate_random_data(in, gen);

        in_copy = in;
        kfr_fft.forward(in_copy, out[0]);

        in_copy = in;
        fftw3_fft.forward(in_copy, out[1]);

        in_copy = in;
        zldsp_fft.forward(in_copy, out[2]);

        in_copy = in;
        pffft_fft.forward(in_copy, out[3]);

        compute_median(out, median_ref);

        for (size_t a = 0; a < NUM_ALGOS; ++a) {
            mse_accum[a] += calculate_mse(median_ref, out[a]);
        }
    }

    std::cout << std::scientific;
    for (size_t a = 0; a < NUM_ALGOS; ++a) {
        if (a > 0)
            std::cout << " ";
        std::cout << (mse_accum[a] / static_cast<double>(reps));
    }
    std::cout << std::endl;

    return 0;
}
