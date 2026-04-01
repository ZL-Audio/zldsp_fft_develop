cmake_minimum_required(VERSION 3.20)
project(zlfft)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ACCURACY_TEST "Accuracy Test" ON)
option(THROUGHPUT_TEST "Throughput Test" OFF)
option(PROFILER_TEST "Profiler Test (no Google Benchmark)" OFF)
option(USE_AVX2 "Use AVX2 for benchmarking" OFF)

if (ACCURACY_TEST)
    add_executable(zlfft_benchmark benchmark/accuracy.cpp)
endif ()
if (THROUGHPUT_TEST)
    add_executable(zlfft_benchmark benchmark/throughput.cpp)

    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing Google benchmark's tests" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Suppressing Google benchmark's install" FORCE)
    set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "Suppressing Google benchmark's dependencies downloading" FORCE)
    add_subdirectory(google/benchmark)
    target_link_libraries(zlfft_benchmark PRIVATE benchmark::benchmark)
endif ()

option(STAGE_TIMING_TEST "Stage Timing Test" OFF)
if (STAGE_TIMING_TEST)
    add_executable(zlfft_benchmark benchmark/stage_timing.cpp)

    if (NOT TARGET benchmark::benchmark)
        set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing Google benchmark's tests" FORCE)
        set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Suppressing Google benchmark's install" FORCE)
        set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "Suppressing Google benchmark's dependencies downloading" FORCE)
        add_subdirectory(google/benchmark)
    endif ()
    target_link_libraries(zlfft_benchmark PRIVATE benchmark::benchmark)
endif ()

# Collect all FFT targets
set(ZLFFT_TARGETS)
if (TARGET zlfft_benchmark)
    list(APPEND ZLFFT_TARGETS zlfft_benchmark)
endif ()
if (PROFILER_TEST)
    add_executable(zlfft_profiler benchmark/profiler_main.cpp)
    list(APPEND ZLFFT_TARGETS zlfft_profiler)
endif ()

foreach(_tgt ${ZLFFT_TARGETS})
    target_include_directories(${_tgt} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/benchmark ${CMAKE_CURRENT_SOURCE_DIR}/source ${CMAKE_CURRENT_SOURCE_DIR}/zlfft_impl)
endforeach()

if (USE_DOUBLE)
    foreach(_tgt ${ZLFFT_TARGETS})
        target_compile_definitions(${_tgt} PRIVATE USE_DOUBLE)
    endforeach()
endif ()

# Algorithm selection flags (applied to all targets)
set(ZLFFT_ALGO_OPTIONS
    ENABLE_NAIVE_COOLEY_RADIX2
    ENABLE_STOCKHAM_RADIX2_KERNEL24
    ENABLE_NAIVE_STOCKHAM_RADIX4
    ENABLE_STOCKHAM_RADIX2_KERNEL248
    ENABLE_SIMD_STOCKHAM_RADIX2
    ENABLE_SIMD_STOCKHAM_RADIX2_KERNEL1
    ENABLE_SIMD_STOCKHAM_RADIX2_KERNEL2
    ENABLE_SIMD_STOCKHAM_RADIX2_KERNEL24
    ENABLE_SIMD_STOCKHAM_RADIX2_KERNEL248
    ENABLE_SIMD_STOCKHAM_RADIX4
    ENABLE_SIMD_STOCKHAM_RADIX4_SOA
    ENABLE_SIMD_STOCKHAM_RADIX8
    ENABLE_SIMD_STOCKHAM_RADIX4_OPT1
    ENABLE_SIMD_STOCKHAM_RADIX4_SOA_KERNEL4
    ENABLE_SIMD_STOCKHAM_RADIX4_SOA_KERNEL4_OPT1
    ENABLE_SIMD_LOW_ORDER
    ENABLE_SIMD_LOW_ORDER_OPT1
    ENABLE_SIMD_LOW_ORDER_OPT2
    ENABLE_SIMD_LOW_ORDER_AOSOA1
    ENABLE_SIMD_LOW_ORDER_AOSOA2
    ENABLE_SIMD_LOW_ORDER_AOSOA3
    ENABLE_SIMD_LOW_ORDER_AOSOA4
    ENABLE_SIMD_LOW_ORDER_AOSOA5
    ENABLE_SIMD_LOW_ORDER_AOSOA6
    ENABLE_HIGH_ORDER_AOSOA1
    ENABLE_HIGH_ORDER_AOSOA2
    ENABLE_HYBRID_AOSOA1
    ENABLE_ZLDSP
)

foreach(_algo ${ZLFFT_ALGO_OPTIONS})
    option(${_algo} "Enable ${_algo}" OFF)
    if (${_algo})
        foreach(_tgt ${ZLFFT_TARGETS})
            target_compile_definitions(${_tgt} PRIVATE ${_algo})
        endforeach()
    endif ()
endforeach()

option(ENABLE_PFFFT "Enable pffft" OFF)
if (ENABLE_PFFFT)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        set(PFFFT_USE_SIMD_NEON ON CACHE BOOL "Enable neon in pffft" FORCE)
    else ()
        if (USE_AVX2)
            set(DISABLE_SIMD_AVX OFF CACHE BOOL "Enable AVX in pffft" FORCE)
        else ()
            set(DISABLE_SIMD_AVX ON CACHE BOOL "Disable AVX in pffft" FORCE)
        endif ()
    endif ()
    add_subdirectory(pffft_impl/pffft)
    if (TARGET PFFFT)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
            target_compile_options(PFFFT PRIVATE "-march=armv8-a+simd" "-O3")
        else ()
            if (USE_AVX2)
                target_compile_options(PFFFT PRIVATE "-march=skylake" "-O3")
            else ()
                target_compile_options(PFFFT PRIVATE "-msse2" "-O3")
            endif ()
        endif ()
    endif()
    foreach(_tgt ${ZLFFT_TARGETS})
        target_link_libraries(${_tgt} PRIVATE PFFFT)
        target_compile_definitions(${_tgt} PRIVATE ENABLE_PFFFT)
    endforeach()
endif ()

option(ENABLE_VDSP "Enable vDSP" OFF)
if (ENABLE_VDSP)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    foreach(_tgt ${ZLFFT_TARGETS})
        target_link_libraries(${_tgt} PRIVATE ${ACCELERATE_FRAMEWORK})
        target_compile_definitions(${_tgt} PRIVATE ENABLE_VDSP)
    endforeach()
endif ()

option(ENABLE_VDSP_STRIDE_2 "Enable vDSP stride 2" OFF)
if (ENABLE_VDSP_STRIDE_2)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    foreach(_tgt ${ZLFFT_TARGETS})
        target_link_libraries(${_tgt} PRIVATE ${ACCELERATE_FRAMEWORK})
        target_compile_definitions(${_tgt} PRIVATE ENABLE_VDSP_STRIDE_2)
    endforeach()
endif ()

option(ENABLE_IPP "Enable Intel IPP" OFF)
if (ENABLE_IPP)
    foreach(_tgt ${ZLFFT_TARGETS})
        if (USE_AVX2)
            target_compile_definitions(${_tgt} PRIVATE IPP_USE_AVX2)
        endif()
    endforeach()
    if(WIN32)
        set(IPP_ROOT "$ENV{USERPROFILE}/.nuget/packages/intelipp.static.win-x64/2022.3.0.387")
        set(IPP_INC "${IPP_ROOT}/build/native/include/ipp")
        set(IPP_LIB "${IPP_ROOT}/build/native/win-x64/")
        set(IPP_LIBS ippsmt ippcoremt ippimt ippcvmt ippvmmt)
    elseif(UNIX)
        find_path(IPP_INC
                NAMES ippcore.h
                HINTS
                $ENV{IPPROOT}/include/ipp
                $ENV{IPPROOT}/include
                PATHS
                "/opt/intel/oneapi/include/ipp"
                "/opt/intel/oneapi/include"
                "/opt/intel/oneapi/ipp/latest/include"
        )
        find_library(IPP_CORE_LIB
                NAMES ippcore
                HINTS
                $ENV{IPPROOT}/lib
                $ENV{IPPROOT}/lib/intel64
                PATHS
                "/opt/intel/oneapi/lib"
                "/opt/intel/oneapi/ipp/latest/lib/intel64"
        )
        if (IPP_INC AND IPP_CORE_LIB)
            get_filename_component(IPP_LIB "${IPP_CORE_LIB}" DIRECTORY)
            set(IPP_LIBS ipps ippcore ippi ippcv ippvm)
            set(IPP_ROOT "${IPP_INC}")
            message(STATUS "Dynamically located IPP Headers at: ${IPP_INC}")
            message(STATUS "Dynamically located IPP Libraries at: ${IPP_LIB}")
        else()
            message(FATAL_ERROR "CMake could not locate Intel IPP headers or libraries.")
        endif()
    endif()
    if (IS_DIRECTORY "${IPP_ROOT}")
        message(STATUS "INTEL IPP FOUND at ${IPP_ROOT}")
        foreach(_tgt ${ZLFFT_TARGETS})
            target_include_directories(${_tgt} PRIVATE "${IPP_INC}")
            target_link_directories(${_tgt} PRIVATE "${IPP_LIB}")
            target_link_libraries(${_tgt} PRIVATE ${IPP_LIBS})
            target_compile_definitions(${_tgt} PRIVATE ENABLE_IPP)
        endforeach()
    else ()
        message(FATAL_ERROR "INTEL IPP NOT LOADED: ${IPP_ROOT} was not found")
    endif ()
endif ()

option(ENABLE_KFR "Enable kfr" OFF)
if (ENABLE_KFR)
    set(KFR_USE_BOOST_MATH OFF CACHE BOOL "Disable Boost Math usage in kfr" FORCE)
    set(KFR_USE_BOOST OFF CACHE BOOL "Disable Boost usage in kfr" FORCE)
    set(KFR_ENABLE_MULTIARCH OFF CACHE BOOL "Disable kfr SIMD runtime dispatch" FORCE)
    if(NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64"))
        set(CMAKE_OSX_ARCHITECTURES "x86_64" CACHE STRING "Set to OSX x86_64" FORCE)
        if (USE_AVX2)
            set(KFR_ARCH "avx2" CACHE STRING "Set SIMD set to AVX2 in kfr" FORCE)
        else ()
            set(KFR_ARCH "sse2" CACHE STRING "Set SIMD set to SSE2 in kfr" FORCE)
        endif ()
    endif()
    add_subdirectory(kfr_impl/kfr)
    foreach(_tgt ${ZLFFT_TARGETS})
        target_link_libraries(${_tgt} PRIVATE kfr kfr_dft)
        target_compile_definitions(${_tgt} PRIVATE ENABLE_KFR)
    endforeach()
endif ()

option(ENABLE_FFTW3 "Enable fftw3" OFF)
option(ENABLE_FFTW3_ESTIMATE "Enable fftw3 estimate" OFF)
if (ENABLE_FFTW3 OR ENABLE_FFTW3_ESTIMATE)
    set(BUILD_TESTS OFF CACHE BOOL "Disable fftw3 tests" FORCE)
    set(ENABLE_FLOAT ON CACHE BOOL "Enable single precision support in fftw3" FORCE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        set(ENABLE_NEON ON CACHE BOOL "Enable NEON in fftw3" FORCE)
    else ()
        if (USE_AVX2)
            set(ENABLE_SSE2 OFF CACHE BOOL "Disable SSE2 in fftw3" FORCE)
            set(ENABLE_AVX2 ON CACHE BOOL "Enable AVX2 in fftw3" FORCE)
        else ()
            set(ENABLE_SSE2 ON CACHE BOOL "Enable SSE2 in fftw3" FORCE)
            set(ENABLE_AVX OFF CACHE BOOL "Disable AVX in fftw3" FORCE)
            set(ENABLE_AVX2 OFF CACHE BOOL "Disable AVX2 in fftw3" FORCE)
        endif ()
    endif ()
    if (NOT TARGET fftw3f)
        add_subdirectory(fftw3_impl/fftw3 EXCLUDE_FROM_ALL)
    endif ()
    if (TARGET fftw3f)
        get_target_property(_fftw_defs fftw3f COMPILE_DEFINITIONS)
        if (_fftw_defs)
            list(REMOVE_ITEM _fftw_defs "/bigobj")
            set_target_properties(fftw3f PROPERTIES COMPILE_DEFINITIONS "${_fftw_defs}")
        endif()

        if (MSVC)
            target_compile_options(fftw3f PRIVATE "/bigobj")
        endif()
    endif()
    foreach(_tgt ${ZLFFT_TARGETS})
        target_link_libraries(${_tgt} PRIVATE fftw3f)
        target_include_directories(${_tgt} PRIVATE fftw3_impl/fftw3/api)
        if (ENABLE_FFTW3)
            target_compile_definitions(${_tgt} PRIVATE ENABLE_FFTW3)
        endif ()
        if (ENABLE_FFTW3_ESTIMATE)
            target_compile_definitions(${_tgt} PRIVATE ENABLE_FFTW3_ESTIMATE)
        endif ()
    endforeach()
endif ()

set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "Disable Highway examples" FORCE)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "Disable Highway tests" FORCE)
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "Disable Highway contrib" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Force static libraries" FORCE)
set(HWY_ENABLE_SHARED OFF CACHE BOOL "Disable Highway shared library" FORCE)

add_subdirectory(google/highway)

foreach(_tgt ${ZLFFT_TARGETS})
    target_link_libraries(${_tgt} PRIVATE hwy)

    if (MSVC)
        target_compile_options(${_tgt} PRIVATE $<$<CONFIG:RELEASE>:/fp:precise>)
    else ()
        target_compile_options(${_tgt} PRIVATE $<$<CONFIG:RELEASE>:-O3 -ffp-contract=fast -fno-signed-zeros -freciprocal-math>)
        target_compile_options(${_tgt} PRIVATE $<$<CONFIG:RelWithDebInfo>:-O3 -ffp-contract=fast -fno-signed-zeros -freciprocal-math>)
    endif ()

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        target_compile_options(${_tgt} PRIVATE "-march=armv8-a+simd")
    else ()
        if (USE_AVX2)
            target_compile_options(${_tgt} PRIVATE "-march=skylake")
        else ()
            target_compile_options(${_tgt} PRIVATE "-msse2" "-mno-avx" "-mno-avx2")
        endif ()
    endif ()
    target_compile_definitions(${_tgt} PRIVATE HWY_COMPILE_ONLY_STATIC)
endforeach()

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)

if (ipo_supported AND TARGET zlfft_benchmark)
    message(STATUS "LTO/IPO is supported and enabled.")
    set_target_properties(zlfft_benchmark PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
else ()
    message(WARNING "LTO/IPO is not supported: ${error}")
endif ()
