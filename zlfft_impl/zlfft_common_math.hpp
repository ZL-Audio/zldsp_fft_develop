#pragma once

#include <cmath>
#include <numbers>

namespace zlfft::math {

#if defined(__APPLE__)

    inline double cospi_d(const double x) { return ::__cospi(x); }
    inline double sinpi_d(const double x) { return ::__sinpi(x); }
    inline float cospi_f(const float x) { return ::__cospif(x); }
    inline float sinpi_f(const float x) { return ::__sinpif(x); }

#else
    constexpr double PI_D = std::numbers::pi_v<double>;
    constexpr float PI_F = std::numbers::pi_v<float>;

    inline double cospi_d(const double x) {
        double rem = std::fmod(std::abs(x), 1.0);
        if (rem == 0.5)
            return 0.0;
        return std::cos(x * PI_D);
    }

    inline double sinpi_d(const double x) {
        double rem = std::fmod(std::abs(x), 1.0);
        if (rem == 0.0)
            return 0.0;
        return std::sin(x * PI_D);
    }

    inline float cospi_f(const float x) {
        float rem = std::fmod(std::abs(x), 1.0f);
        if (rem == 0.5f)
            return 0.0f;
        return std::cos(x * PI_F);
    }

    inline float sinpi_f(const float x) {
        float rem = std::fmod(std::abs(x), 1.0f);
        if (rem == 0.0f)
            return 0.0f;
        return std::sin(x * PI_F);
    }
#endif

    template <typename T>
    inline T cospi(T x);

    template <>
    inline float cospi<float>(const float x) { return cospi_f(x); }

    template <>
    inline double cospi<double>(const double x) { return cospi_d(x); }

    template <typename T>
    inline T sinpi(T x);

    template <>
    inline float sinpi<float>(const float x) { return sinpi_f(x); }

    template <>
    inline double sinpi<double>(const double x) { return sinpi_d(x); }
}
