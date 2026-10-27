include_guard(GLOBAL)

if(NOT ZLDSP_FFT_SEPARATE_TARGETS)
    message(FATAL_ERROR "Set ZLDSP_FFT_SEPARATE_TARGETS to a list of SSE2, SSE4, and/or AVX2.")
endif()

if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(_zldsp_fft_flags_SSE2 "")
    set(_zldsp_fft_flags_AVX2 /arch:AVX2)
else()
    set(_zldsp_fft_flags_SSE2 -march=x86-64)
    set(_zldsp_fft_flags_SSE4 -march=x86-64-v2 -maes -mpclmul)
    set(_zldsp_fft_flags_AVX2 -march=x86-64-v3 -maes -mpclmul)
endif()

add_library(zldsp_fft_separate_dispatch STATIC
    "${CMAKE_CURRENT_LIST_DIR}/separate_dispatch_wrapper.cpp")
target_include_directories(zldsp_fft_separate_dispatch PUBLIC "${CMAKE_CURRENT_LIST_DIR}")
set(_zldsp_fft_components zldsp_fft_separate_dispatch)
set(_zldsp_fft_backends "")

foreach(_zldsp_fft_isa IN LISTS ZLDSP_FFT_SEPARATE_TARGETS)
    if(NOT DEFINED _zldsp_fft_flags_${_zldsp_fft_isa})
        message(FATAL_ERROR "Unsupported FFT target for ${CMAKE_CXX_COMPILER_ID}: ${_zldsp_fft_isa}")
    endif()
    set(_zldsp_fft_flags ${_zldsp_fft_flags_${_zldsp_fft_isa}})
    if(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        list(TRANSFORM _zldsp_fft_flags PREPEND "/clang:")
    endif()

    string(TOLOWER "zldsp_fft_${_zldsp_fft_isa}" _zldsp_fft_target)
    add_library(${_zldsp_fft_target} OBJECT "${CMAKE_CURRENT_LIST_DIR}/dynamic_dispatch_wrapper.cpp")
    target_compile_definitions(${_zldsp_fft_target} PRIVATE
        HWY_COMPILE_ONLY_STATIC "ZLDSP_FFT_BACKEND_ONLY=HWY_${_zldsp_fft_isa}")
    target_compile_options(${_zldsp_fft_target} PRIVATE ${_zldsp_fft_flags})
    target_link_libraries(zldsp_fft_separate_dispatch PRIVATE ${_zldsp_fft_target})
    list(APPEND _zldsp_fft_components ${_zldsp_fft_target})
    string(APPEND _zldsp_fft_backends "ZLDSP_FFT_BACKEND(${_zldsp_fft_isa})")
endforeach()
target_compile_definitions(zldsp_fft_separate_dispatch PRIVATE
    "ZLDSP_FFT_SEPARATE_TARGETS=${_zldsp_fft_backends}")

foreach(_zldsp_fft_target IN LISTS _zldsp_fft_components)
    target_compile_features(${_zldsp_fft_target} PUBLIC cxx_std_20)
    target_link_libraries(${_zldsp_fft_target} PRIVATE hwy)
    target_include_directories(${_zldsp_fft_target} PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../zldsp_fft")
    set_target_properties(${_zldsp_fft_target} PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION FALSE POSITION_INDEPENDENT_CODE ON)
endforeach()
