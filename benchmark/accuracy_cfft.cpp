#include <cmath>
#include <iostream>
#include <vector>
#include "benchmark_include.hpp"

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

int main(const int argc, char** argv) {
    if (argc < 2)
        return 1;
    const int order = std::stoi(argv[1]);
    const size_t size = static_cast<size_t>(1) << order;

    FFTClass cfft(order);

    std::vector<C, AlignedAllocator<C>> in_aos(size);
    generate_random_data(in_aos);

    std::vector<F, AlignedAllocator<F>> in_soa_r(size);
    std::vector<F, AlignedAllocator<F>> in_soa_i(size);
    for (size_t i = 0; i < size; ++i) {
        in_soa_r[i] = in_aos[i].real();
        in_soa_i[i] = in_aos[i].imag();
    }

    std::vector<C, AlignedAllocator<C>> out_aos_aos(size);

    std::vector<F, AlignedAllocator<F>> out_aos_soa_r(size);
    std::vector<F, AlignedAllocator<F>> out_aos_soa_i(size);
    std::array<std::span<F>, 2> out_aos_soa = {std::span<F>(out_aos_soa_r), std::span<F>(out_aos_soa_i)};

    std::vector<C, AlignedAllocator<C>> out_soa_aos(size);

    std::vector<F, AlignedAllocator<F>> out_soa_soa_r(size);
    std::vector<F, AlignedAllocator<F>> out_soa_soa_i(size);
    std::array<std::span<F>, 2> out_soa_soa = {std::span<F>(out_soa_soa_r), std::span<F>(out_soa_soa_i)};

    std::vector<C, AlignedAllocator<C>> in_aos_copy = in_aos;
    cfft.forward(in_aos_copy, out_aos_aos);

    in_aos_copy = in_aos;
    cfft.forward(in_aos_copy, out_aos_soa);

    std::vector<F, AlignedAllocator<F>> in_soa_r_copy = in_soa_r;
    std::vector<F, AlignedAllocator<F>> in_soa_i_copy = in_soa_i;
    std::array<std::span<F>, 2> in_soa_copy = {std::span<F>(in_soa_r_copy), std::span<F>(in_soa_i_copy)};
    cfft.forward(in_soa_copy, out_soa_aos);

    in_soa_r_copy = in_soa_r;
    in_soa_i_copy = in_soa_i;
    in_soa_copy = {std::span<F>(in_soa_r_copy), std::span<F>(in_soa_i_copy)};
    cfft.forward(in_soa_copy, out_soa_soa);

    double max_mse_forward = 0.0;
    max_mse_forward = std::max(max_mse_forward, calculate_mse_soa(out_aos_aos, out_aos_soa));
    max_mse_forward = std::max(max_mse_forward, calculate_mse_aos(out_aos_aos, out_soa_aos));
    max_mse_forward = std::max(max_mse_forward, calculate_mse_soa(out_aos_aos, out_soa_soa));

    std::vector<C, AlignedAllocator<C>> back_aos_aos(size);
    std::vector<F, AlignedAllocator<F>> back_aos_soa_r(size);
    std::vector<F, AlignedAllocator<F>> back_aos_soa_i(size);
    std::array<std::span<F>, 2> back_aos_soa = {std::span<F>(back_aos_soa_r), std::span<F>(back_aos_soa_i)};
    std::vector<C, AlignedAllocator<C>> back_soa_aos(size);
    std::vector<F, AlignedAllocator<F>> back_soa_soa_r(size);
    std::vector<F, AlignedAllocator<F>> back_soa_soa_i(size);
    std::array<std::span<F>, 2> back_soa_soa = {std::span<F>(back_soa_soa_r), std::span<F>(back_soa_soa_i)};

    in_aos_copy = in_aos;
    cfft.backward(in_aos_copy, back_aos_aos);

    in_aos_copy = in_aos;
    cfft.backward(in_aos_copy, back_aos_soa);

    in_soa_r_copy = in_soa_r;
    in_soa_i_copy = in_soa_i;
    in_soa_copy = {std::span<F>(in_soa_r_copy), std::span<F>(in_soa_i_copy)};
    cfft.backward(in_soa_copy, back_soa_aos);

    in_soa_r_copy = in_soa_r;
    in_soa_i_copy = in_soa_i;
    in_soa_copy = {std::span<F>(in_soa_r_copy), std::span<F>(in_soa_i_copy)};
    cfft.backward(in_soa_copy, back_soa_soa);

    double max_mse_backward = 0.0;
    max_mse_backward = std::max(max_mse_backward, calculate_mse_soa(back_aos_aos, back_aos_soa));
    max_mse_backward = std::max(max_mse_backward, calculate_mse_aos(back_aos_aos, back_soa_aos));
    max_mse_backward = std::max(max_mse_backward, calculate_mse_soa(back_aos_aos, back_soa_soa));

    std::vector<C, AlignedAllocator<C>> fwd_then_back(size);
    std::vector<C, AlignedAllocator<C>> out_aos_aos_copy = out_aos_aos;
    cfft.backward(out_aos_aos_copy, fwd_then_back);

    const F n_inv = static_cast<F>(1.0 / static_cast<double>(size));
    std::vector<C, AlignedAllocator<C>> normalized_out(size);
    for (size_t i = 0; i < size; ++i) {
        normalized_out[i] = C(fwd_then_back[i].real() * n_inv, fwd_then_back[i].imag() * n_inv);
    }
    double mse_identity = calculate_mse_aos(in_aos, normalized_out);

    std::cout << std::scientific << max_mse_forward << " " << max_mse_backward << " " << mse_identity << std::endl;
    return 0;
}
