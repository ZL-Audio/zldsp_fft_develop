import argparse
import subprocess
import os
import sys
import platform
import json

algos = ["zldsp",
         "fftw3", "fftw3_estimate", "kfr", "vdsp", "vdsp_stride_2", "pffft", "ipp"]


def get_algo_list(full=False):
    if not full:
        return ["kfr", "vdsp", "ipp", "zldsp", "pffft"]
    else:
        return ["vdsp", "ipp", "fftw3", "fftw3_estimate", "kfr", "pffft"]

def replace_result_keys(results):
    r = {}
    for key, value in results.items():
        if key == "vdsp":
            r["vDSP"] = value
        elif key == "ipp":
            r["IPP"] = value
        elif key == "pffft":
            r["PFFFT"] = value
        elif key == "kfr":
            r["KFR"] = value
        elif key == "fftw3":
            r["FFTW3"] = value
        elif key == "fftw3_estimate":
            r["FFTW3 estimate"] = value
        else:
            r[key] = value
    return r

def build_benchmark(algorithm, benchmark_type, use_avx2=False, use_double=False, to_print=False):
    build_dir = "build_fft"
    os.makedirs(build_dir, exist_ok=True)

    cmake_cmd = ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release", "-G", "Ninja"]
    if benchmark_type == "accuracy":
        cmake_cmd += ["-DACCURACY_TEST=ON", "-DTHROUGHPUT_TEST=OFF", "-DSTAGE_TIMING_TEST=OFF", "-DPROFILER_TEST=OFF", "-DACCURACY_CFFT_TEST=OFF"]
    elif benchmark_type == "accuracy_cfft":
        cmake_cmd += ["-DACCURACY_TEST=OFF", "-DTHROUGHPUT_TEST=OFF", "-DSTAGE_TIMING_TEST=OFF", "-DPROFILER_TEST=OFF", "-DACCURACY_CFFT_TEST=ON", "-DACCURACY_RFFT_TEST=OFF"]
    elif benchmark_type == "accuracy_rfft":
        cmake_cmd += ["-DACCURACY_TEST=OFF", "-DTHROUGHPUT_TEST=OFF", "-DSTAGE_TIMING_TEST=OFF", "-DPROFILER_TEST=OFF", "-DACCURACY_CFFT_TEST=OFF", "-DACCURACY_RFFT_TEST=ON"]
    elif benchmark_type == "throughput":
        cmake_cmd += ["-DACCURACY_TEST=OFF", "-DTHROUGHPUT_TEST=ON", "-DTHROUGHPUT_RFFT_TEST=OFF", "-DSTAGE_TIMING_TEST=OFF", "-DPROFILER_TEST=OFF", "-DACCURACY_CFFT_TEST=OFF"]
    elif benchmark_type == "throughput_rfft":
        cmake_cmd += ["-DACCURACY_TEST=OFF", "-DTHROUGHPUT_TEST=OFF", "-DTHROUGHPUT_RFFT_TEST=ON", "-DSTAGE_TIMING_TEST=OFF", "-DPROFILER_TEST=OFF", "-DACCURACY_CFFT_TEST=OFF"]
    elif benchmark_type == "stage_timing":
        cmake_cmd += ["-DACCURACY_TEST=OFF", "-DTHROUGHPUT_TEST=OFF", "-DSTAGE_TIMING_TEST=ON", "-DPROFILER_TEST=OFF", "-DACCURACY_CFFT_TEST=OFF"]
    elif benchmark_type == "profiler":
        cmake_cmd += ["-DACCURACY_TEST=OFF", "-DTHROUGHPUT_TEST=OFF", "-DSTAGE_TIMING_TEST=OFF", "-DPROFILER_TEST=ON", "-DACCURACY_CFFT_TEST=OFF"]

    if use_avx2:
        cmake_cmd += ["-DUSE_AVX2=ON"]
    else:
        cmake_cmd += ["-DUSE_AVX2=OFF"]

    if use_double:
        cmake_cmd += ["-DUSE_DOUBLE=ON"]
    else:
        cmake_cmd += ["-DUSE_DOUBLE=OFF"]

    if platform.system() == "Windows":
        cmake_cmd += ["-DCMAKE_C_COMPILER=clang-cl", "-DCMAKE_CXX_COMPILER=clang-cl"]
    if platform.system() == "Linux":
        cmake_cmd += ["-DCMAKE_C_COMPILER=clang", "-DCMAKE_CXX_COMPILER=clang++"]

    for algo in algos:
        if algo == algorithm:
            cmake_cmd.append(f"-DENABLE_{algo.upper()}=ON")
        else:
            cmake_cmd.append(f"-DENABLE_{algo.upper()}=OFF")

    if benchmark_type == "profiler":
        target_name = "zlfft_profiler"
    elif benchmark_type == "accuracy_cfft":
        target_name = "zlfft_accuracy_cfft"
    elif benchmark_type == "accuracy_rfft":
        target_name = "zlfft_accuracy_rfft"
    else:
        target_name = "zlfft_benchmark"
    build_cmd = ["cmake", "--build", ".", "--target", target_name, "--config", "Release", "-j"]

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

        exe_name = f"{target_name}.exe"
        return os.path.join(build_dir, exe_name)

    else:
        if to_print:
            print(f"Configuring: {cmake_cmd}")
        subprocess.run(cmake_cmd, capture_output=True, cwd=build_dir, check=True)

        if to_print:
            print(f"Building: {build_cmd}")
        subprocess.run(build_cmd, capture_output=True, cwd=build_dir, check=True)

        return os.path.join(build_dir, target_name)
