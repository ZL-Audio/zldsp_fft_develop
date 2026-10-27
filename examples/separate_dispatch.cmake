include_guard(GLOBAL)

if(NOT ZLDSP_FFT_SEPARATE_TARGETS)
    message(FATAL_ERROR "Set ZLDSP_FFT_SEPARATE_TARGETS to a list of SSE2, SSE4, and/or AVX2.")
endif()

set(_fft_flags_SSE2 -march=x86-64)
set(_fft_flags_SSE4 -march=x86-64-v2 -maes -mpclmul)
set(_fft_flags_AVX2 -march=x86-64-v3 -maes -mpclmul)

add_library(zldsp_fft_separate_dispatch STATIC
    "${CMAKE_CURRENT_LIST_DIR}/separate_dispatch_wrapper.cpp")
target_include_directories(zldsp_fft_separate_dispatch PUBLIC "${CMAKE_CURRENT_LIST_DIR}")
set(_fft_components zldsp_fft_separate_dispatch)
set(_fft_backends "")

foreach(_fft_isa IN LISTS ZLDSP_FFT_SEPARATE_TARGETS)
    if(NOT DEFINED _fft_flags_${_fft_isa})
        message(FATAL_ERROR "Unknown FFT target: ${_fft_isa}")
    endif()
    set(_fft_flags ${_fft_flags_${_fft_isa}})
    if(MSVC)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            list(TRANSFORM _fft_flags PREPEND "/clang:")
        elseif(_fft_isa STREQUAL "SSE2")
            set(_fft_flags "")
        elseif(_fft_isa STREQUAL "AVX2")
            set(_fft_flags /arch:AVX2)
        else()
            message(FATAL_ERROR "${_fft_isa} requires Clang/clang-cl or GCC.")
        endif()
    endif()

    string(TOLOWER "zldsp_fft_${_fft_isa}" _fft_target)
    add_library(${_fft_target} OBJECT "${CMAKE_CURRENT_LIST_DIR}/dynamic_dispatch_wrapper.cpp")
    target_compile_definitions(${_fft_target} PRIVATE
        HWY_COMPILE_ONLY_STATIC "ZLDSP_FFT_BACKEND_ONLY=HWY_${_fft_isa}")
    target_compile_options(${_fft_target} PRIVATE ${_fft_flags})
    target_link_libraries(zldsp_fft_separate_dispatch PRIVATE ${_fft_target})
    list(APPEND _fft_components ${_fft_target})
    string(APPEND _fft_backends "ZLDSP_FFT_BACKEND(${_fft_isa})")
endforeach()
target_compile_definitions(zldsp_fft_separate_dispatch PRIVATE
    "ZLDSP_FFT_SEPARATE_TARGETS=${_fft_backends}")

foreach(_fft_target IN LISTS _fft_components)
    target_compile_features(${_fft_target} PUBLIC cxx_std_20)
    target_link_libraries(${_fft_target} PRIVATE hwy)
    target_include_directories(${_fft_target} PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../zldsp_fft")
    set_target_properties(${_fft_target} PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION FALSE POSITION_INDEPENDENT_CODE ON)
endforeach()

if(NOT MSVC)
    target_compile_options(zldsp_fft_separate_dispatch PRIVATE -march=x86-64)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(zldsp_fft_separate_dispatch PRIVATE /clang:-march=x86-64)
endif()
