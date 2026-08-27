#include "dynamic_dispatch_wrapper.hpp"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "dynamic_dispatch_wrapper.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "zldsp_fft_cfft-inl.hpp"
#include "zldsp_fft_rfft-inl.hpp"

HWY_BEFORE_NAMESPACE();

namespace zldsp_fft_example::HWY_NAMESPACE {
    template <typename F>
    class CFFTImpl final : public CFFT<F> {
    public:
        explicit CFFTImpl(const std::size_t order) :
            cfft_(order) {
        }

        void forward(Complex<F>* input, Complex<F>* output) noexcept override {
            cfft_.forward(input, output);
        }

    private:
        zldsp::fft::HWY_NAMESPACE::CFFT<F> cfft_;
    };

    template <typename F>
    class RFFTImpl final : public RFFT<F> {
    public:
        explicit RFFTImpl(const std::size_t order) :
            rfft_(order) {
        }

        void forward(F* input, Complex<F>* output) noexcept override {
            rfft_.forward(input, output);
        }

    private:
        zldsp::fft::HWY_NAMESPACE::RFFT<F> rfft_;
    };

    CFFT<float>* prepare_cfft_float_impl(const std::size_t order) {
        return new CFFTImpl<float>(order);
    }

    CFFT<double>* prepare_cfft_double_impl(const std::size_t order) {
        return new CFFTImpl<double>(order);
    }

    RFFT<float>* prepare_rfft_float_impl(const std::size_t order) {
        return new RFFTImpl<float>(order);
    }

    RFFT<double>* prepare_rfft_double_impl(const std::size_t order) {
        return new RFFTImpl<double>(order);
    }
}

HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace zldsp_fft_example {
    HWY_EXPORT(prepare_cfft_float_impl);
    HWY_EXPORT(prepare_cfft_double_impl);
    HWY_EXPORT(prepare_rfft_float_impl);
    HWY_EXPORT(prepare_rfft_double_impl);

    namespace detail {
        std::unique_ptr<CFFT<float>> prepare_cfft_float(const std::size_t order) {
            return std::unique_ptr<CFFT<float>>(
                HWY_DYNAMIC_DISPATCH(prepare_cfft_float_impl)(order));
        }

        std::unique_ptr<CFFT<double>> prepare_cfft_double(const std::size_t order) {
            return std::unique_ptr<CFFT<double>>(
                HWY_DYNAMIC_DISPATCH(prepare_cfft_double_impl)(order));
        }

        std::unique_ptr<RFFT<float>> prepare_rfft_float(const std::size_t order) {
            return std::unique_ptr<RFFT<float>>(
                HWY_DYNAMIC_DISPATCH(prepare_rfft_float_impl)(order));
        }

        std::unique_ptr<RFFT<double>> prepare_rfft_double(const std::size_t order) {
            return std::unique_ptr<RFFT<double>>(
                HWY_DYNAMIC_DISPATCH(prepare_rfft_double_impl)(order));
        }
    }
}

#endif
