#if defined(ZLDSP_FFT_COMMON_TWIDDLE_HPP_) == defined(HWY_TARGET_TOGGLE)
#ifdef ZLDSP_FFT_COMMON_TWIDDLE_HPP_
#undef ZLDSP_FFT_COMMON_TWIDDLE_HPP_
#else
#define ZLDSP_FFT_COMMON_TWIDDLE_HPP_
#endif

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <hwy/highway.h>

#include "zldsp_fft_common_twiddle_constants.hpp"

// preserve intermediate rounding; explicit FMAs remain enabled
// Clang: use fast-honor-pragmas for fast contraction; see tools/twiddle/README.md
#if defined(__clang__)
#pragma float_control(precise, on, push)
#pragma clang fp contract(off)
#elif defined(_MSC_VER)
#pragma float_control(precise, on, push)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("no-fast-math", "fp-contract=off")
#endif

HWY_BEFORE_NAMESPACE();

namespace zldsp::fft::HWY_NAMESPACE::common::twiddle {
    namespace hn = hwy::HWY_NAMESPACE;
    using detail::twiddle::Anchor;
    using detail::twiddle::DoubleDouble;

    template <typename F>
    struct Pair { F c, s; };

    // TwoSum/TwoProduct for normal values and zeros in round-to-nearest
    // see Ogita, Rump & Oishi (2005), doi:10.1137/030601818
    [[nodiscard]] inline DoubleDouble two_sum(const double a, const double b) noexcept {
        const double hi = a + b;
        const double v = hi - a;
        return {hi, (a - (hi - v)) + (b - v)};
    }

    // FastTwoSum; requires |hi| >= |lo|
    [[nodiscard]] inline DoubleDouble normalize(const double hi, const double lo) noexcept {
        const double sum = hi + lo;
        return {sum, lo - (sum - hi)};
    }

    [[nodiscard]] inline DoubleDouble add(const DoubleDouble a, const DoubleDouble b) noexcept {
        const auto sum = two_sum(a.hi, b.hi);
        return normalize(sum.hi, sum.lo + (a.lo + b.lo));
    }

    [[nodiscard]] inline DoubleDouble negate(const DoubleDouble a) noexcept {
        return {-a.hi, -a.lo};
    }

    [[nodiscard]] inline DoubleDouble product(const double a, const double b) noexcept {
        const double hi = a * b;
#if HWY_NATIVE_FMA
        // use the dispatched target's native FMA with a scalar lane
        const hn::CappedTag<double, 1> d;
        const double lo = hn::GetLane(hn::MulAdd(hn::Set(d, a), hn::Set(d, b), hn::Set(d, -hi)));
#else
        // Dekker split with 2^27 + 1; avoids software FMA on SSE2/SSE4
        constexpr double splitter = 0x1.0000002p+27;
        const double ac = splitter * a, bc = splitter * b;
        const double ah = ac - (ac - a), bh = bc - (bc - b);
        const double al = a - ah, bl = b - bh;
        const double lo = ((ah * bh - hi) + ah * bl + al * bh) + al * bl;
#endif
        return {hi, lo};
    }

    [[nodiscard]] inline DoubleDouble multiply(const DoubleDouble a, const DoubleDouble b) noexcept {
        const auto p = product(a.hi, b.hi);
#if HWY_NATIVE_FMA
        const hn::CappedTag<double, 1> d;
        auto tail = hn::Set(d, a.lo * b.lo);
        tail = hn::MulAdd(hn::Set(d, a.hi), hn::Set(d, b.lo), tail);
        tail = hn::MulAdd(hn::Set(d, a.lo), hn::Set(d, b.hi), tail);
        return normalize(p.hi, hn::GetLane(tail) + p.lo);
#else
        const double tail = ((a.lo * b.lo + a.hi * b.lo) + a.lo * b.hi) + p.lo;
        return normalize(p.hi, tail);
#endif
    }

    template <typename F>
    [[nodiscard]] inline F rounded(const DoubleDouble a) noexcept {
        static_assert(std::is_same_v<F, float> || std::is_same_v<F, double>);
        if constexpr (std::is_same_v<F, double>) {
            // normalized hi is already rounded to binary64
            return a.hi;
        } else {
            const float hi = static_cast<float>(a.hi);
            // use the residual to resolve binary32 midpoint ties
            // nonzero twiddles are normal floats
            constexpr std::uint64_t mask = (std::uint64_t{1} << 29) - 1;
            constexpr std::uint64_t midpoint = std::uint64_t{1} << 28;
            if ((std::bit_cast<std::uint64_t>(a.hi) & mask) == midpoint &&
                ((a.lo > 0.0 && static_cast<double>(hi) < a.hi) ||
                 (a.lo < 0.0 && static_cast<double>(hi) > a.hi))) {
                return std::nextafter(hi, a.lo > 0.0 ? std::numeric_limits<float>::infinity()
                                                   : -std::numeric_limits<float>::infinity());
            }
            return hi;
        }
    }

    [[nodiscard]] inline Pair<double> rfft_weights(const Anchor a) noexcept {
        // retain the residual through 1 + sin(theta), with 0 <= sin(theta) <= 1
        // final scaling by 0.5 is exact
        const auto sum = normalize(1.0, a.s.hi);
        const auto shifted = normalize(sum.hi, sum.lo + a.s.lo);
        return {0.5 * rounded<double>(a.c), 0.5 * rounded<double>(shifted)};
    }

    [[nodiscard]] inline Pair<float> rfft_weights(const Pair<double> a) noexcept {
        // form the final weights in double before narrowing
        return {static_cast<float>(0.5 * a.c), static_cast<float>(0.5 * (1.0 + a.s))};
    }

    class SplitTable {
    public:
        // n is a power of two >= 4; last bounds the requested first octant
        // float_lookup skips small-grid setup for octant<float>
        explicit SplitTable(const size_t n, const size_t last, const bool float_lookup = false)
            : order_(static_cast<unsigned>(std::countr_zero(n))),
              shift_(order_ > 3 ? (order_ - 3) / 2 : 0),
              mask_((size_t{1} << shift_) - 1), fine_offset_((last >> shift_) + 1),
              turns_scale_(std::ldexp(1.0, -static_cast<int>(order_))) {
            assert(n >= 4 && std::has_single_bit(n) && last <= n / 8);
            // float uses the embedded grid directly through order 10
            if (float_lookup && order_ <= 10) {
                return;
            }
            // store coarse and fine anchors in one allocation
            anchors_.reserve(fine_offset_ + std::min(mask_, last) + 1);
            for (size_t q = 0; q < fine_offset_; ++q) {
                anchors_.push_back(anchor(q << shift_));
            }
            for (size_t r = 0; r <= std::min(mask_, last); ++r) {
                anchors_.push_back(anchor(r));
            }
        }

        template <typename F>
        [[nodiscard]] inline auto octant(const size_t index) const noexcept {
            static_assert(std::is_same_v<F, float> || std::is_same_v<F, double>);
            if constexpr (std::is_same_v<F, float>) {
                if (order_ <= 10) {
                    const auto& a = detail::twiddle::kAnchors[index << (10 - order_)];
                    return Pair<double>{a.c.hi, a.s.hi};
                }
                // combine anchor high words for float storage
                const auto& a = anchors_[index >> shift_];
                const auto& b = anchors_[fine_offset_ + (index & mask_)];
                return Pair<double>{a.c.hi * b.c.hi - a.s.hi * b.s.hi,
                                    a.s.hi * b.c.hi + a.c.hi * b.s.hi};
            } else {
                const auto& a = anchors_[index >> shift_];
                const auto& b = anchors_[fine_offset_ + (index & mask_)];
                // skip identity rotations
                if ((index & mask_) == 0) {
                    return a;
                }
                if ((index >> shift_) == 0) {
                    return b;
                }
                return Anchor{add(multiply(a.c, b.c), negate(multiply(a.s, b.s))),
                              add(multiply(a.s, b.c), multiply(a.c, b.s))};
            }
        }

    private:
        [[nodiscard]] inline Anchor anchor(const size_t index) const noexcept {
            const unsigned coarse_shift = order_ > 10 ? order_ - 10 : 0;
            const unsigned expand_shift = order_ < 10 ? 10 - order_ : 0;
            const auto a = detail::twiddle::kAnchors[(index >> coarse_shift) << expand_shift];
            const size_t residual = index & ((size_t{1} << coarse_shift) - 1);
            if (residual == 0) {
                return a;
            }
            const double u = static_cast<double>(residual) * turns_scale_;
            const auto t = multiply({u, 0.0}, detail::twiddle::kTau);
            const auto z = multiply(t, t);
            auto ps = detail::twiddle::kSinResidual[4];
            auto pc = detail::twiddle::kCosResidual[4];
            for (int j = 3; j >= 0; --j) {
                ps = add(multiply(z, ps), detail::twiddle::kSinResidual[j]);
                pc = add(multiply(z, pc), detail::twiddle::kCosResidual[j]);
            }
            const auto s = add(t, multiply(multiply(t, z), ps));
            const auto c = add({1.0, 0.0}, multiply(z, pc));
            return {add(multiply(a.c, c), negate(multiply(a.s, s))),
                    add(multiply(a.s, c), multiply(a.c, s))};
        }

        unsigned order_, shift_;
        size_t mask_, fine_offset_;
        double turns_scale_;
        std::vector<Anchor> anchors_;
    };

    template <typename F>
    class Stage {
    public:
        // share octant roots across harmonics; max_index includes the multiplier
        explicit Stage(const size_t n, const size_t max_index) : n_(n) {
            const size_t last = std::min(n / 8, max_index);
            const SplitTable table(n, last, std::is_same_v<F, float>);
            roots_.reserve(last + 1);
            for (size_t k = 0; k <= last; ++k) {
                const auto a = table.octant<F>(k);
                if constexpr (std::is_same_v<F, float>) {
                    roots_.push_back({static_cast<float>(a.c), static_cast<float>(a.s)});
                } else {
                    roots_.push_back({rounded<F>(a.c), rounded<F>(a.s)});
                }
            }
        }

        [[nodiscard]] inline Pair<F> forward(const size_t index) const noexcept {
            const size_t quarter = n_ / 4, half_n = n_ / 2;
            size_t k = index & (quarter - 1);
            const bool reflect = k > quarter / 2;
            if (reflect) {
                k = quarter - k;
            }
            auto p = roots_[k];
            if (reflect != static_cast<bool>(index & quarter)) {
                std::swap(p.c, p.s);
            }
            if ((index + quarter) & half_n) {
                p.c = -p.c;
            }
            if (!(index & half_n)) {
                p.s = -p.s;
            }
            return p;
        }

    private:
        size_t n_;
        std::vector<Pair<F>> roots_;
    };
}

HWY_AFTER_NAMESPACE();

#if defined(__clang__) || defined(_MSC_VER)
#pragma float_control(pop)
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif
