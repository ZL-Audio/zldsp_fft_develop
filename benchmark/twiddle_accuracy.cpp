// independent MPFR coefficient audit
#include "zldsp_fft_cfft.hpp"
#include "zldsp_fft_rfft.hpp"

#include <mpfr.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace fft = zldsp::fft::HWY_STATIC_NAMESPACE::common;
namespace tw = fft::twiddle;
namespace hn = hwy::HWY_STATIC_NAMESPACE;

namespace {
    struct Reference { double d; float f; };

    class Oracle {
    public:
        Oracle() { mpfr_inits2(192, x_, lower_, upper_, static_cast<mpfr_ptr>(nullptr)); }
        ~Oracle() { mpfr_clears(x_, lower_, upper_, static_cast<mpfr_ptr>(nullptr)); }

        Reference get(const size_t k, const size_t n, const bool sine,
                      const bool shifted = false, const bool forward = false) {
            for (mpfr_prec_t precision = 192; precision <= 3072; precision *= 2) {
                mpfr_set_prec(lower_, precision);
                mpfr_set_prec(upper_, precision);
                // indices below 2^30 are exact
                mpfr_set_d(x_, static_cast<double>(k), MPFR_RNDN);
                mpfr_mul_2si(x_, x_, 1 - static_cast<long>(std::countr_zero(n)), MPFR_RNDN);
                if (forward) {
                    mpfr_neg(x_, x_, MPFR_RNDN);
                }
                if (sine) {
                    mpfr_sinpi(lower_, x_, MPFR_RNDD);
                    mpfr_sinpi(upper_, x_, MPFR_RNDU);
                } else {
                    mpfr_cospi(lower_, x_, MPFR_RNDD);
                    mpfr_cospi(upper_, x_, MPFR_RNDU);
                }
                if (shifted) {
                    mpfr_add_ui(lower_, lower_, 1, MPFR_RNDD);
                    mpfr_add_ui(upper_, upper_, 1, MPFR_RNDU);
                    mpfr_div_2ui(lower_, lower_, 1, MPFR_RNDD);
                    mpfr_div_2ui(upper_, upper_, 1, MPFR_RNDU);
                }
                const Reference r{mpfr_get_d(lower_, MPFR_RNDN), mpfr_get_flt(lower_, MPFR_RNDN)};
                if (r.d == mpfr_get_d(upper_, MPFR_RNDN) && r.f == mpfr_get_flt(upper_, MPFR_RNDN)) {
                    return r;
                }
            }
            throw std::runtime_error("MPFR interval did not determine rounding");
        }

        void product(const double a, const double b, double& hi, double& lo) {
            mpfr_set_prec(lower_, 192);
            mpfr_set_d(x_, a, MPFR_RNDN);
            mpfr_mul_d(lower_, x_, b, MPFR_RNDN); // exact binary64 product at 192 bits
            hi = mpfr_get_d(lower_, MPFR_RNDN);
            mpfr_sub_d(lower_, lower_, hi, MPFR_RNDN);
            lo = mpfr_get_d(lower_, MPFR_RNDN);
        }

    private:
        mpfr_t x_, lower_, upper_;
    };

    struct Stats {
        std::uint64_t checked = 0, wrong = 0, zero_wrong = 0, max_steps = 0;
        std::uint64_t float_rounding_differences = 0;
        double max_abs = 0.0;

        template <typename F>
        void check(const F actual, const F expected, const char* label, const size_t n, const size_t k,
                   const bool allow_float_rounding = false) {
            ++checked;
            if (actual == expected) { // accept either zero sign
                return;
            }
            if (!std::isfinite(actual)) {
                throw std::runtime_error("Nonfinite coefficient");
            }
            using Bits = std::conditional_t<sizeof(F) == 4, std::uint32_t, std::uint64_t>;
            const auto ordered = [](const F value) -> std::uint64_t {
                constexpr Bits sign = Bits{1} << (8 * sizeof(F) - 1);
                const Bits bits = std::bit_cast<Bits>(value);
                return bits & sign ? sign - (bits & (sign - 1)) : sign + bits;
            };
            const auto a = ordered(actual), b = ordered(expected);
            const auto steps = a > b ? a - b : b - a;
            max_steps = std::max(max_steps, steps);
            max_abs = std::max(max_abs, std::abs(static_cast<double>(actual) - static_cast<double>(expected)));
            if constexpr (std::is_same_v<F, float>) {
                if (allow_float_rounding && expected != 0.0f && steps <= 1) {
                    ++float_rounding_differences;
                    return;
                }
            }
            ++wrong;
            if (expected == F{0}) {
                ++zero_wrong;
            }
            if (wrong <= 8) {
                std::cerr << label << ": N=" << n << " k=" << k << " bits=" << 8 * sizeof(F)
                          << " actual=" << std::hexfloat << actual << " expected=" << expected
                          << std::defaultfloat << '\n';
            }
        }

        template <typename F>
        void check(const F actual, const Reference r, const char* label, const size_t n, const size_t k,
                   const bool allow_float_rounding = false) {
            if constexpr (sizeof(F) == 4) {
                check(actual, r.f, label, n, k, allow_float_rounding);
            } else {
                check(actual, r.d, label, n, k, allow_float_rounding);
            }
        }

        template <typename F, typename Expected>
        void coefficient(const F actual, const Expected expected, const char* label,
                         const size_t n, const size_t k) {
            // keep embedded grids, axes, and diagonals exact
            const bool allow = std::is_same_v<F, float> && n > 1024 && (k & (n / 8 - 1)) != 0;
            check(actual, expected, label, n, k, allow);
        }
    };

    void primitives(Oracle& oracle, Stats& stats) {
        const auto diagonal = oracle.get(1, 8, false);
        stats.check(zldsp::fft::detail::twiddle::kInvSqrt2, diagonal, "diagonal", 8, 1);
        stats.check(static_cast<float>(zldsp::fft::detail::twiddle::kInvSqrt2), diagonal, "diagonal", 8, 1);
        for (const auto [a, b] : std::array<std::pair<double, double>, 4>{{
                 {0x1.0000000000001p+0, 0x1.fffffffffffffp-1}, {0x1.921fb54442d18p+2, 0x1.23456p-30},
                 {0x1.6a09e667f3bcdp-1, 0x1.6a09e667f3bcdp-1}, {0.0, 1.0}}}) {
            double hi, lo;
            oracle.product(a, b, hi, lo);
            const auto p = tw::product(a, b);
            stats.check(p.hi, hi, "product high", 0, 0);
            stats.check(p.lo, lo, "product residual", 0, 0);
        }
    }

    std::vector<size_t> sampled_indices(const unsigned order) {
        const size_t n = size_t{1} << order, last = n / 8;
        const size_t split = size_t{1} << (order > 3 ? (order - 3) / 2 : 0);
        std::vector<size_t> indices{0, last};
        const auto neighbors = [&](const size_t k) {
            if (k > 0 && k <= last) indices.push_back(k - 1);
            if (k <= last) indices.push_back(k);
            if (k < last) indices.push_back(k + 1);
        };
        std::mt19937_64 random(0x51f17 + order);
        for (unsigned j = 0; j < 2048; ++j) {
            indices.push_back(random() % (last + 1));
        }
        for (size_t j = 0; j <= 128; ++j) {
            neighbors(last * j / 128); // seed boundaries
            neighbors(std::min(last, j * split));
            neighbors(last - std::min(last, j * split));
        }
        // order-28 split regression
        if (order == 28) neighbors(n / 2 - 124806999);
        // float RFFT rounding midpoint
        if (order == 30) neighbors(93157219);
        std::sort(indices.begin(), indices.end());
        indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
        return indices;
    }

    void grid(const unsigned order, const bool exhaustive, Oracle& oracle, Stats& stats) {
        const size_t n = size_t{1} << order;
        const tw::SplitTable table(n, n / 8);
        const auto check = [&](const size_t k) {
            const auto a = table.octant<double>(k);
            const auto f = table.octant<float>(k);
            const auto c = oracle.get(k, n, false), s = oracle.get(k, n, true);
            const auto shifted_c = oracle.get(k, n, false, true), shifted_s = oracle.get(k, n, true, true);
            const auto cd = tw::rfft_weights(a);
            const auto cf = tw::rfft_weights(f);
            const auto md = tw::rfft_weights(tw::Anchor{a.s, a.c});
            const auto mf = tw::rfft_weights(tw::Pair<double>{f.s, f.c});
            stats.check(a.c.hi, c, "cos", n, k);
            stats.check(a.s.hi, s, "sin", n, k);
            stats.coefficient(static_cast<float>(f.c), c, "cos", n, k);
            stats.coefficient(static_cast<float>(f.s), s, "sin", n, k);
            stats.check(cd.c, c.d * 0.5, "cos/2", n, k);
            stats.coefficient(cf.c, c.f * 0.5f, "cos/2", n, k);
            stats.check(md.c, s.d * 0.5, "mirror cos/2", n, k);
            stats.coefficient(mf.c, s.f * 0.5f, "mirror cos/2", n, k);
            stats.check(cd.s, shifted_s, "(1+sin)/2", n, k);
            stats.coefficient(cf.s, shifted_s, "(1+sin)/2", n, k);
            stats.check(md.s, shifted_c, "mirror (1+sin)/2", n, k);
            stats.coefficient(mf.s, shifted_c, "mirror (1+sin)/2", n, k);
        };
        if (exhaustive) {
            for (size_t k = 0; k <= n / 8; ++k) check(k);
        } else {
            for (const auto k : sampled_indices(order)) check(k);
        }
        std::cout << "grid order " << order << (exhaustive ? " exhaustive" : " sampled") << '\n' << std::flush;
    }

    template <typename F>
    void folding(const unsigned order, Oracle& oracle, Stats& stats) {
        const size_t n = size_t{1} << order;
        const tw::Stage<F> roots(n, n / 8);
        for (size_t k = 0; k < n; ++k) {
            const auto p = roots.forward(k);
            stats.coefficient(p.c, oracle.get(k, n, false, false, true), "fold cosine", n, k);
            stats.coefficient(p.s, oracle.get(k, n, true, false, true), "fold sine", n, k);
        }
        // check single writes against folded roots
        const size_t width = n / 4;
        std::vector<unsigned char> written(3 * width);
        tw::generate_radix4<F>(n, [&](const size_t m, const size_t k, const tw::Pair<F> p) {
            if (m >= 3 || k >= width || written[m * width + k]++) {
                throw std::runtime_error("Invalid or duplicate dense twiddle write");
            }
            const auto expected = roots.forward(k * (m + 1));
            stats.check(p.c, expected.c, "dense cosine", n, k * (m + 1));
            stats.check(p.s, expected.s, "dense sine", n, k * (m + 1));
        });
        if (std::find(written.begin(), written.end(), 0) != written.end()) {
            throw std::runtime_error("Missing dense twiddle write");
        }
        // check partial-octant aliases
        const size_t last = std::min<size_t>(7, n / 8);
        const tw::Stage<F> prefix(n, last);
        for (size_t k = 0; k <= last; ++k) {
            for (const size_t quadrant : {size_t{0}, n / 4, n / 2, 3 * n / 4}) {
                for (const size_t j : {(quadrant + k) & (n - 1), (quadrant - k) & (n - 1)}) {
                    const auto p = prefix.forward(j);
                    stats.coefficient(p.c, oracle.get(j, n, false, false, true), "prefix cosine", n, j);
                    stats.coefficient(p.s, oracle.get(j, n, true, false, true), "prefix sine", n, j);
                }
            }
        }
    }

    template <typename F>
    void emitted(const unsigned order, Oracle& oracle, Stats& stats) {
        const size_t n = size_t{1} << order;
        constexpr size_t lanes = hn::MaxLanes(hn::ScalableTag<F>());
        fft::CFFTState<F> state;
        state.cfft_order = order;
        state.cfft_size = n;
        fft::init_cfft_state(order, state);
        const auto component = [&](const F value, const size_t k, const size_t denominator, const bool sine) {
            stats.coefficient(value, oracle.get(k, denominator, sine, false, true), "stored CFFT", denominator, k);
        };
        const auto dense = [&](const F* values, const size_t width, const size_t span,
                               const size_t denominator, const size_t pack) {
            const size_t count = 6 * std::max(span, pack);
            for (size_t off = 0; off < count; ++off) {
                const size_t block = off / (6 * pack), sub = off % (6 * pack);
                const size_t k = (block * pack + sub % pack) % width;
                const size_t harmonic = 1 + sub / (2 * pack);
                component(values[off], k * harmonic, denominator, (sub / pack) % 2 != 0);
            }
            return count;
        };
        size_t offset = 0;
        if (state.micro_order == 4 || state.micro_order == 5) {
            const size_t micro_size = size_t{1} << state.micro_order;
            const size_t width = micro_size / 4;
            for (size_t m = 1; m <= 3; ++m) {
                for (size_t k = 0; k < width; ++k, ++offset) {
                    component(state.micro_w[offset], k * m, micro_size, false);
                    component(state.micro_w[3 * width + offset], k * m, micro_size, true);
                }
            }
        } else {
            size_t width = state.micro_order % 2 == 0 ? 4 : 8;
            for (size_t i = 1; i < state.micro_stages.size(); ++i) {
                const size_t pack = state.micro_stages[i] == fft::StageType::kRadix4Width4
                                        ? std::max<size_t>(4, lanes) : lanes;
                const size_t count = dense(state.micro_w.get() + offset, width, width, 4 * width, pack);
                if (count != state.micro_w_strides[i]) throw std::runtime_error("Micro layout mismatch");
                offset += count;
                width *= 4;
            }
        }
        offset = 0;
        for (size_t i = 0; i < state.macro_stages.size(); ++i) {
            const size_t sub_n = n >> (2 * i), width = sub_n / 4;
            const auto stage = state.macro_stages[i];
            const size_t fine = stage.tile_width == 0 ? width : stage.tile_width;
            size_t count = dense(state.macro_w.get() + offset, width, fine, sub_n, lanes);
            if (stage.tile_width != 0) {
                const size_t coarse = width / fine;
                for (size_t q = 0; q < coarse; ++q) {
                    for (size_t m = 0; m < 3; ++m) {
                        component(state.macro_w[offset + count + 6 * q + 2 * m], q * fine * (m + 1), sub_n, false);
                        component(state.macro_w[offset + count + 6 * q + 2 * m + 1], q * fine * (m + 1), sub_n, true);
                    }
                }
                count += ((6 * coarse + lanes - 1) / lanes) * lanes;
            }
            if (count != stage.w_stride) throw std::runtime_error("Macro layout mismatch");
            offset += count;
        }
        hwy::AlignedFreeUniquePtr<F[]> weights;
        fft::generate_rfft_pre_post_w(order - 1, weights);
        if (order == 1) {
            if (weights) throw std::runtime_error("Unexpected order-1 RFFT weights");
            return;
        }
        size_t real_lanes = 0;
        fft::dispatch_rfft_tag<F>(order - 1, [&](auto d) { real_lanes = hn::Lanes(d); });
        const size_t count = ((n / 4 + real_lanes - 1) / real_lanes) * real_lanes;
        for (size_t i = 0; i < count; ++i) {
            const size_t k = i + 1, off = 2 * (i / real_lanes) * real_lanes + i % real_lanes;
            if (k <= n / 4) {
                const auto c = oracle.get(k, n, false), s = oracle.get(k, n, true, true);
                stats.coefficient(weights[off], Reference{c.d * 0.5, c.f * 0.5f}, "stored RFFT cosine", n, k);
                stats.coefficient(weights[off + real_lanes], s, "stored RFFT affine", n, k);
            } else {
                stats.check(weights[off], F{0}, "RFFT padding", n, k);
                stats.check(weights[off + real_lanes], F{0.5}, "RFFT padding", n, k);
            }
        }
    }
}

int main(const int argc, const char* const argv[]) {
    if (argc > 2 || (argc == 2 && std::string(argv[1]) != "--quick")) {
        std::cerr << "usage: zldsp_fft_twiddle_accuracy [--quick]\n";
        return 2;
    }
    try {
        const bool quick = argc == 2;
        Oracle oracle;
        Stats stats;
        primitives(oracle, stats);
        for (unsigned order = 2; order <= (quick ? 12u : 16u); ++order) {
            grid(order, true, oracle, stats);
        }
        for (unsigned order = quick ? 13u : 17u; order <= 30; ++order) {
            grid(order, !quick && (order == 20 || order == 22), oracle, stats);
        }
        for (unsigned order = 2; order <= 12; ++order) {
            folding<float>(order, oracle, stats);
            folding<double>(order, oracle, stats);
        }
        std::vector<unsigned> orders;
        for (unsigned order = 1; order <= (quick ? 12u : 16u); ++order) orders.push_back(order);
        orders.push_back(quick ? 16 : 20);
        for (const auto order : orders) {
            emitted<float>(order, oracle, stats);
            emitted<double>(order, oracle, stats);
            std::cout << "stored tables order " << order << '\n' << std::flush;
        }
        std::cout << "checked=" << stats.checked << " failures=" << stats.wrong
                  << " float_rounding_differences=" << stats.float_rounding_differences
                  << " exact_zero_misses=" << stats.zero_wrong << " max_steps=" << stats.max_steps
                  << " max_abs=" << stats.max_abs << '\n';
        return stats.wrong == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
