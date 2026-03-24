import argparse
import subprocess
import os
import sys
import platform
import json

algos = ["naive_stockham_radix2", "naive_cooley_radix2", "naive_stockham_radix4",
         "stockham_radix2_kernel24", "stockham_radix2_kernel248",

         "simd_stockham_radix2",
         "simd_stockham_radix2_kernel1",
         "simd_stockham_radix2_kernel2",
         "simd_stockham_radix2_kernel24",
         "simd_stockham_radix2_kernel248",
         "simd_stockham_radix4",
         "simd_stockham_radix4_soa",
         "simd_stockham_radix8",
         "simd_stockham_radix4_opt1",
         "simd_stockham_radix4_soa_kernel4",
         "simd_stockham_radix4_soa_kernel4_opt1",

         "simd_low_order",
         "simd_low_order_opt1",
         "simd_low_order_opt2",
         "simd_low_order_aosoa1",
         "simd_low_order_aosoa2",

         "fftw3", "fftw3_estimate", "kfr", "vdsp", "vdsp_stride_2", "pffft", "ipp"]

def get_algo_list(full=False):
    if not full:
        return ["kfr",
                "simd_low_order_opt1", "simd_low_order_opt2",
                "simd_low_order_aosoa1", "simd_low_order_aosoa2"]
    elif platform.system() == "Darwin":
        return ["fftw3", "fftw3_estimate", "kfr", "vdsp", "pffft",
                "simd_low_order_opt1", "simd_low_order_opt2",
                "simd_low_order_aosoa1", "simd_low_order_aosoa2"]
    else:
        return ["fftw3", "fftw3_estimate", "kfr", "ipp", "pffft",
                "simd_low_order_opt1", "simd_low_order_opt2",
                "simd_low_order_aosoa1", "simd_low_order_aosoa2"]

def build_benchmark(algorithm, benchmark_type, use_avx2=False, to_print=False):
    build_dir = "build_fft"
    os.makedirs(build_dir, exist_ok=True)

    cmake_cmd = ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release", "-G", "Ninja"]
    if benchmark_type == "accuracy":
        cmake_cmd += ["-DACCURACY_TEST=ON", "-DTHROUGHPUT_TEST=OFF"]
    elif benchmark_type == "throughput":
        cmake_cmd += ["-DACCURACY_TEST=OFF", "-DTHROUGHPUT_TEST=ON"]

    if use_avx2:
        cmake_cmd += ["-DUSE_AVX2=ON"]
    else:
        cmake_cmd += ["-DUSE_AVX2=OFF"]

    if platform.system() == "Windows":
        cmake_cmd += ["-DCMAKE_C_COMPILER=clang-cl", "-DCMAKE_CXX_COMPILER=clang-cl"]
    if platform.system() == "Linux":
        cmake_cmd += ["-DCMAKE_C_COMPILER=clang", "-DCMAKE_CXX_COMPILER=clang++"]

    for algo in algos:
        if algo == algorithm:
            cmake_cmd.append(f"-DENABLE_{algo.upper()}=ON")
        else:
            cmake_cmd.append(f"-DENABLE_{algo.upper()}=OFF")

    build_cmd = ["cmake", "--build", ".", "--target", "zlfft_benchmark", "--config", "Release", "-j"]

    if platform.system() == "Windows":
        vcvars = '"C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Auxiliary\\Build\\vcvars64.bat"'

        cmake_cmd_str = " ".join(cmake_cmd)
        full_cmake_cmd = f'call {vcvars} && {cmake_cmd_str}'
        
        build_cmd_str = " ".join(build_cmd)
        full_build_cmd = f'call {vcvars} && {build_cmd_str}'

        if to_print:
            print(f"Configuring: {full_cmake_cmd}")
        subprocess.run(full_cmake_cmd, capture_output=True, cwd=build_dir, check=True, shell=True)
        if to_print:
            print(f"Building: {full_build_cmd}")
        subprocess.run(full_build_cmd, capture_output=True, cwd=build_dir, check=True, shell=True)
        
        return os.path.join(build_dir, "zlfft_benchmark.exe")

    else:
        if to_print:
            print(f"Configuring: {cmake_cmd}")
        subprocess.run(cmake_cmd, capture_output=True, cwd=build_dir, check=True)

        if to_print:
            print(f"Building: {build_cmd}")
        subprocess.run(build_cmd, capture_output=True, cwd=build_dir, check=True)

        return os.path.join(build_dir, "zlfft_benchmark")
