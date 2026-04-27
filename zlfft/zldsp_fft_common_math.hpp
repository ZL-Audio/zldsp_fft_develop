#pragma once

#include <cmath>
#include <numbers>

namespace zldsp::fft::common::math {
#if defined(__APPLE__)

    inline double cospi(const double x) {
        return __cospi(x);
    }

    inline double sinpi(const double x) {
        return __sinpi(x);
    }

#else

    inline double cospi(const double x) {
        double rem = std::fmod(std::abs(x), 1.0);
        if (rem == 0.5) {
            return 0.0;
        }
        return std::cos(x * std::numbers::pi_v<double>);
    }

    inline double sinpi(const double x) {
        double rem = std::fmod(std::abs(x), 1.0);
        if (rem == 0.0) {
            return std::copysign(0.0, x);
        }
        return std::sin(x * std::numbers::pi_v<double>);
    }

#endif
}
