#pragma once

#include <memory>

#include "pffft/include/pffft/pffft.h"
#include "pffft/include/pffft/pffft_double.h"
#include "../benchmark/fft_support.hpp"

namespace zlbenchmark {
    namespace detail {
        template <typename F>
        struct PffftApi;

        template <>
        struct PffftApi<float> {
            using Setup = PFFFT_Setup;

            static bool valid(int n, pffft_transform_t kind) {
                return pffft_is_valid_size(n, kind);
            }

            static Setup* create(int n, pffft_transform_t kind) {
                return pffft_new_setup(n, kind);
            }

            static void destroy(Setup* p) {
                pffft_destroy_setup(p);
            }

            static void execute(Setup* p, const float* in, float* out, float* work,
                                pffft_direction_t direction) {
                pffft_transform(p, in, out, work, direction);
            }
        };

        template <>
        struct PffftApi<double> {
            using Setup = PFFFTD_Setup;

            static bool valid(int n, pffft_transform_t kind) {
                return pffftd_is_valid_size(n, kind);
            }

            static Setup* create(int n, pffft_transform_t kind) {
                return pffftd_new_setup(n, kind);
            }

            static void destroy(Setup* p) {
                pffftd_destroy_setup(p);
            }

            static void execute(Setup* p, const double* in, double* out, double* work,
                                pffft_direction_t direction) {
                pffftd_transform(p, in, out, work, direction);
            }
        };

        template <typename F, pffft_transform_t Kind>
        class PffftPlan {
            using API = PffftApi<F>;
            std::unique_ptr<typename API::Setup, decltype(&API::destroy)> setup_{nullptr, API::destroy};
            AlignedVector<F> work_;

        public:
            explicit PffftPlan(size_t order) {
                const int n = fft_size(order);
                if (!API::valid(n, Kind))
                    throw UnsupportedSize("PFFFT does not support this transform size");
                setup_.reset(API::create(n, Kind));
                if (!setup_)
                    throw std::runtime_error("PFFFT setup failed");
                work_.resize(Kind == PFFFT_REAL ? n : 2 * size_t(n));
            }

            void execute(const F* in, F* out, pffft_direction_t direction) {
                API::execute(setup_.get(), in, out, work_.data(), direction);
            }
        };
    }

    template <typename F>
    class PffftFFT final {
        detail::PffftPlan<F, PFFFT_COMPLEX> plan_;

    public:
        explicit PffftFFT(size_t order) :
            plan_(order) {
        }

        // 2*N scalars: interleaved time data and PFFFT's native unordered spectrum.
        void forward(const F* in, F* out) {
            plan_.execute(in, out, PFFFT_FORWARD);
        }

        void backward(const F* in, F* out) {
            plan_.execute(in, out, PFFFT_BACKWARD);
        }
    };

    template <typename F>
    class PffftRFFT final {
        detail::PffftPlan<F, PFFFT_REAL> plan_;

    public:
        explicit PffftRFFT(size_t order) :
            plan_(order) {
        }

        // N scalars in both domains; retain native packing, order and scaling.
        void forward(const F* in, F* out) {
            plan_.execute(in, out, PFFFT_FORWARD);
        }

        void backward(const F* in, F* out) {
            plan_.execute(in, out, PFFFT_BACKWARD);
        }
    };
}
