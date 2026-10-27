#include "dynamic_dispatch_wrapper.hpp"

#include <stdexcept>

#include <hwy/targets.h>

namespace zldsp_fft_example {
#define ZLDSP_FFT_BACKEND(isa) \
    namespace N_##isa { \
        CFFT<float>* prepare_cfft_float_impl(std::size_t order); \
        CFFT<double>* prepare_cfft_double_impl(std::size_t order); \
        RFFT<float>* prepare_rfft_float_impl(std::size_t order); \
        RFFT<double>* prepare_rfft_double_impl(std::size_t order); \
    }
    ZLDSP_FFT_SEPARATE_TARGETS
#undef ZLDSP_FFT_BACKEND

    namespace {
        struct Backend {
            int64_t target;
            CFFT<float>* (*cfft_float)(std::size_t);
            CFFT<double>* (*cfft_double)(std::size_t);
            RFFT<float>* (*rfft_float)(std::size_t);
            RFFT<double>* (*rfft_double)(std::size_t);
        };

        constexpr Backend backends[] = {
#define ZLDSP_FFT_BACKEND(isa) \
            {HWY_##isa, N_##isa::prepare_cfft_float_impl, N_##isa::prepare_cfft_double_impl, \
                        N_##isa::prepare_rfft_float_impl, N_##isa::prepare_rfft_double_impl},
            ZLDSP_FFT_SEPARATE_TARGETS
#undef ZLDSP_FFT_BACKEND
        };

        const Backend& select_backend() {
            const auto supported = hwy::SupportedTargets();
            const Backend* best = nullptr;
            for (const auto& backend : backends) {
                if ((supported & backend.target) && (!best || backend.target < best->target)) {
                    best = &backend;
                }
            }
            if (!best) {
                throw std::runtime_error("No compiled FFT backend is supported on this CPU");
            }
            return *best;
        }
    }

    namespace detail {
        std::unique_ptr<CFFT<float>> prepare_cfft_float(const std::size_t order) {
            return std::unique_ptr<CFFT<float>>(select_backend().cfft_float(order));
        }

        std::unique_ptr<CFFT<double>> prepare_cfft_double(const std::size_t order) {
            return std::unique_ptr<CFFT<double>>(select_backend().cfft_double(order));
        }

        std::unique_ptr<RFFT<float>> prepare_rfft_float(const std::size_t order) {
            return std::unique_ptr<RFFT<float>>(select_backend().rfft_float(order));
        }

        std::unique_ptr<RFFT<double>> prepare_rfft_double(const std::size_t order) {
            return std::unique_ptr<RFFT<double>>(select_backend().rfft_double(order));
        }
    }
}
