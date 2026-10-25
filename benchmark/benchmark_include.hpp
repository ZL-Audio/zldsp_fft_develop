#pragma once

#include <complex>
#include "fft_support.hpp"

#ifdef USE_DOUBLE
using F = double;
#else
using F = float;
#endif
using C = std::complex<F>;

#if defined(ZLB_ALL_APIS) || defined(ENABLE_ZLDSP)
#include "../zldsp_fft/zldsp_fft_cfft.hpp"
#include "../zldsp_fft/zldsp_fft_rfft.hpp"
using CFFTClass = zldsp::fft::CFFT<F>;
using RFFTClass = zldsp::fft::RFFT<F>;
// The public zldsp inverse RFFT is half the conventional unscaled inverse.
inline constexpr F rfft_backward_scale = 2;
#else
inline constexpr F rfft_backward_scale = 1;
#if defined(ENABLE_PFFFT)
#include "../pffft_impl/pffft_impl.hpp"
using CFFTClass = zlbenchmark::PffftFFT<F>;
using RFFTClass = zlbenchmark::PffftRFFT<F>;
#elif defined(ENABLE_VDSP)
#include "../vdsp_impl/vdsp_impl.hpp"
using CFFTClass = zlbenchmark::VDSPFFT<F>;
using RFFTClass = zlbenchmark::VDSPRFFT<F>;
#elif defined(ENABLE_IPP)
#include "../ipp_impl/ipp_impl.hpp"
#ifdef IPP_USE_AVX2
inline constexpr auto ipp_simd = zlbenchmark::IPPSimdLevel::AVX2;
#else
inline constexpr auto ipp_simd = zlbenchmark::IPPSimdLevel::SSE42;
#endif
using CFFTClass = zlbenchmark::IPPFFT<F, ipp_simd>;
using RFFTClass = zlbenchmark::IPPRFFT<F, ipp_simd>;
#elif defined(ENABLE_ARMPL) || defined(ENABLE_ARMPL_ESTIMATE)
#include "../armpl_impl/armpl_impl.hpp"
#ifdef ENABLE_ARMPL_ESTIMATE
inline constexpr unsigned fftw_flags = FFTW_ESTIMATE;
#else
inline constexpr unsigned fftw_flags = FFTW_MEASURE;
#endif
using CFFTClass = zlbenchmark::ArmPLFFT<F, fftw_flags>;
using RFFTClass = zlbenchmark::ArmPLRFFT<F, fftw_flags>;
#elif defined(ENABLE_KFR)
#include "../kfr_impl/kfr_impl.hpp"
using CFFTClass = zlbenchmark::KFRFFT<F>;
using RFFTClass = zlbenchmark::KFRRFFT<F>;
#elif defined(ENABLE_FFTW3) || defined(ENABLE_FFTW3_ESTIMATE)
#include "../fftw3_impl/fftw3_impl.hpp"
#ifdef ENABLE_FFTW3_ESTIMATE
inline constexpr unsigned fftw_flags = FFTW_ESTIMATE;
#else
inline constexpr unsigned fftw_flags = FFTW_MEASURE;
#endif
using CFFTClass = zlbenchmark::FFTW3FFT<F, fftw_flags>;
using RFFTClass = zlbenchmark::FFTW3RFFT<F, fftw_flags>;
#else
#error "Select one ENABLE_<library> benchmark backend"
#endif
#endif
