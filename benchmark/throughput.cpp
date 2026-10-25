#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <benchmark/benchmark.h>
#include "benchmark_cases.hpp"

namespace zlbenchmark {
    inline bool throughput_failed = false;

    // Throughput-only wrappers use native scalar buffers, without imposing
    // canonical complex ordering, packing or normalization on their FFT calls.
    template <bool Backward, bool Real, typename FFT = std::conditional_t<Real, RFFTClass, CFFTClass>>
    struct NativeCase {
        static constexpr bool convolution = false;
        static constexpr const char* direction = Backward ? "backward" : "forward";
        // Preserve the shared result names while keeping buffers in native format.
        static constexpr const char* api = Real ? (Backward ? "aos_real" : "real_aos") : "aos_aos";
        FFT fft;
        AlignedVector<F> in, out, original;

        explicit NativeCase(int order) :
            fft(order), in(size_t(fft_size(order)) * (Real ? 1 : 2)),
            out(in.size(), std::numeric_limits<F>::quiet_NaN()), original(in.size()) {
            std::mt19937 rng(42);
            std::uniform_real_distribution<F> random(F(-1), F(1));
            for (auto& value : original)
                value = random(rng);
            if constexpr (Backward) {
                // Prepare a valid native spectrum once, outside measurement.
                fft.forward(original.data(), in.data());
                original = in;
            }
            reset();
        }

        void reset() {
            std::copy(original.begin(), original.end(), in.begin());
        }

        void execute() {
            if constexpr (Backward)
                fft.backward(in.data(), out.data());
            else
                fft.forward(in.data(), out.data());
        }

        bool input_unchanged() const {
            return in == original;
        }

        auto output_data() {
            return out.data();
        }

        bool output_finite() const {
            return all_finite(out);
        }
    };

    double parse_min_time(const std::string& text) {
        size_t end = 0;
        const double value = std::stod(text, &end);
        if (end != text.size() || !std::isfinite(value) || value <= 0) {
            throw std::invalid_argument("minimum time must be finite and positive");
        }
        return value;
    }

    double min_time_for_order(int order, int n0, int n1, double min_time0, double min_time1) {
        // A single-order run uses the start time; keep both endpoints exact.
        if (order == n0 || min_time0 == min_time1)
            return min_time0;
        if (order == n1)
            return min_time1;
        const double fraction = static_cast<double>(order - n0) / (n1 - n0);
        return std::exp(std::lerp(std::log(min_time0), std::log(min_time1), fraction));
    }

    struct PreparedCaseBase {
        virtual ~PreparedCaseBase() = default;
    };

    template <typename Case>
    struct PreparedCase : PreparedCaseBase {
        int order;
        Case test;
        bool destructive;

        explicit PreparedCase(int requested_order) :
            order(requested_order), test(order) {
            test.execute();
            destructive = !test.input_unchanged();
            if (!test.output_finite()) {
                throw std::runtime_error("non-finite output during validation");
            }
            if constexpr (Case::convolution) {
                if (!test.kernel_unchanged()) {
                    throw std::runtime_error("convolution modified its kernel");
                }
            }
        }
    };

    // Repetitions run consecutively. Keep their plan and buffer addresses fixed,
    // but release the previous case before allocating the next large transform.
    inline std::unique_ptr<PreparedCaseBase> active_throughput_case;

    template <typename Case>
    PreparedCase<Case>& prepare_throughput_case(int order) {
        auto* prepared = dynamic_cast<PreparedCase<Case>*>(active_throughput_case.get());
        if (!prepared || prepared->order != order) {
            active_throughput_case.reset();
            active_throughput_case = std::make_unique<PreparedCase<Case>>(order);
        }
        return static_cast<PreparedCase<Case>&>(*active_throughput_case);
    }

    template <typename Case>
    void measure(benchmark::State& state) {
        try {
            auto& prepared = prepare_throughput_case<Case>(static_cast<int>(state.range(0)));
            auto& test = prepared.test;
            const bool destructive = prepared.destructive;
            // Reset and validation scans precede the final execution warm-up.
            test.reset();
            for (int i = 0; i < 3; ++i) {
                if (destructive && i != 0)
                    test.reset();
                test.execute();
            }
            auto output = test.output_data();
            benchmark::DoNotOptimize(output);
            for (auto _ : state) {
                // Never feed an unscaled transform's output into the next call.
                // Destructive APIs get fresh input outside the measured region.
                if (destructive) {
                    state.PauseTiming();
                    test.reset();
                    state.ResumeTiming();
                }
                test.execute();
                benchmark::ClobberMemory();
            }
            if (!test.output_finite()) {
                throughput_failed = true;
                state.SkipWithError("non-finite output after measurement");
            }
            if constexpr (Case::convolution) {
                if (!test.kernel_unchanged()) {
                    throughput_failed = true;
                    state.SkipWithError("convolution modified its kernel");
                }
                state.counters["convolutions_per_iteration"] = 1;
            }
            state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(fft_size(state.range(0))));
            state.counters["transforms_per_iteration"] = Case::convolution ? 2 : 1;
            state.counters["input_reset_required"] = destructive ? 1 : 0;
        }
        catch (const UnsupportedSize& error) {
            state.SkipWithError((std::string("unsupported size: ") + error.what()).c_str());
        }
        catch (const std::exception& error) {
            throughput_failed = true;
            state.SkipWithError(error.what());
        }
    }
}

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            throw std::invalid_argument("usage: executable <n0> <n1> [--min-time0=0.5] [--min-time1=5] "
                "[google-benchmark-flags]");
        }
        const int n0 = zlbenchmark::parse_integer(argv[1], 1, 30);
        const int n1 = zlbenchmark::parse_integer(argv[2], n0, 30);
        double min_time0 = 0.5, min_time1 = 5.0;
        // Remove our arguments before letting Google Benchmark parse its flags.
        int benchmark_argc = 1;
        for (int i = 3; i < argc; ++i) {
            const std::string argument(argv[i]);
            const auto equals = argument.find('=');
            const auto name = argument.substr(0, equals);
            if (name == "--min-time0" || name == "--min-time1") {
                std::string value;
                if (equals != std::string::npos) {
                    value = argument.substr(equals + 1);
                } else {
                    if (++i == argc)
                        throw std::invalid_argument(name + " requires a value");
                    value = argv[i];
                }
                (name == "--min-time0" ? min_time0 : min_time1) = zlbenchmark::parse_min_time(value);
            } else {
                argv[benchmark_argc++] = argv[i];
            }
        }
        argc = benchmark_argc;
        argv[argc] = nullptr;
        benchmark::Initialize(&argc, argv);
        if (benchmark::ReportUnrecognizedArguments(argc, argv))
            return 1;
        for (int order = n0; order <= n1; ++order) {
            const auto register_case = [&]<typename Case>() {
                const std::string name = std::string(Case::direction) + "/" + Case::api;
                benchmark::RegisterBenchmark(name.c_str(), zlbenchmark::measure<Case>)
                    ->Arg(order)->MinTime(zlbenchmark::min_time_for_order(order, n0, n1, min_time0, min_time1))
                    ->Unit(benchmark::kMicrosecond);
            };
#if !defined(ZLB_ALL_APIS) && (defined(ENABLE_PFFFT) || defined(ENABLE_VDSP))
#ifdef ZLB_REAL_FFT
            constexpr bool real = true;
#else
            constexpr bool real = false;
#endif
            register_case.template operator()<zlbenchmark::NativeCase<false, real>>();
            register_case.template operator()<zlbenchmark::NativeCase<true, real>>();
#else
            zlbenchmark::for_each_case(register_case);
            zlbenchmark::for_each_convolve_baseline(register_case);
#endif
        }
        benchmark::RunSpecifiedBenchmarks();
        zlbenchmark::active_throughput_case.reset();
        benchmark::Shutdown();
        return zlbenchmark::throughput_failed ? 1 : 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
