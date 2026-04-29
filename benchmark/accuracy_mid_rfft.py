import argparse
import subprocess
import os
import sys
import platform
import json
import shutil
from build_config import replace_result_keys

ALGO_NAMES = ["kfr", "fftw3", "zldsp", "pffft"]
NUM_REPS = 10


def build_accuracy_mid(use_avx2=False, use_double=False):
    build_dir = "build_fft"
    os.makedirs(build_dir, exist_ok=True)

    cmake_cmd = ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release", "-G", "Ninja",
                  "-DACCURACY_TEST=OFF", "-DTHROUGHPUT_TEST=OFF",
                  "-DSTAGE_TIMING_TEST=OFF", "-DPROFILER_TEST=OFF",
                  "-DACCURACY_MID_RFFT_TEST=ON"]

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

    from build_config import algos
    for algo in algos:
        cmake_cmd.append(f"-DENABLE_{algo.upper()}=OFF")

    target_name = "zlfft_accuracy_mid_rfft"
    build_cmd = ["cmake", "--build", ".", "--target", target_name, "--config", "Release", "-j"]

    subprocess.run(cmake_cmd, capture_output=True, cwd=build_dir, check=True)

    subprocess.run(build_cmd, capture_output=True, cwd=build_dir, check=True)

    if platform.system() == "Windows":
        return os.path.join(build_dir, f"{target_name}.exe")
    return os.path.join(build_dir, target_name)


def main():
    parser = argparse.ArgumentParser(description="Accuracy Mid RFFT Benchmark (median-based MSE, 4 algorithms)")
    parser.add_argument("n0", type=int, help="Start FFT order (size 2^n)")
    parser.add_argument("n1", type=int, help="End FFT order (size 2^n)")
    parser.add_argument("--avx2", action="store_true", help="Enable AVX2 architecture")
    parser.add_argument("--double", action="store_true", help="Enable Double")

    args = parser.parse_args()

    try:
        exe_path = build_accuracy_mid(use_avx2=args.avx2, use_double=args.double)
    except subprocess.CalledProcessError as e:
        print(f"Build failed!")
        print(f"Stdout:\n{e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout}")
        print(f"Stderr:\n{e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr}")
        sys.exit(1)

    results = {name: [] for name in ALGO_NAMES}

    print(f"\nRunning accuracy_mid_rfft benchmark from order {args.n0} to {args.n1} ({NUM_REPS} reps each)...")
    print(f"{'Order':<10} " + " ".join(f"{name:<18}" for name in ALGO_NAMES))
    print("-" * (10 + 19 * len(ALGO_NAMES)))

    for order in range(args.n0, args.n1 + 1):
        cmd = [exe_path, str(order), str(NUM_REPS), str(42)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            values = result.stdout.strip().split()
            mses = [float(v) for v in values]

            row = f"{order:<10} "
            for i, name in enumerate(ALGO_NAMES):
                results[name].append(mses[i])
                row += f"{mses[i]:<18.8e} "
            print(row)

        except subprocess.CalledProcessError as e:
            print(f"Error at order {order}: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            for name in ALGO_NAMES:
                results[name].append(None)

    print()
    r = replace_result_keys(results)
    print(str(r).replace("],", "],\n"))
    shutil.rmtree("build_fft")

if __name__ == "__main__":
    main()
