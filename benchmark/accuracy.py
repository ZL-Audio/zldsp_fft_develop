import argparse
import subprocess
import os
import sys
import platform

from build_config import build_benchmark


def run_benchmark(exe_path, n0, n1, algorithm):
    print(f"Running accuracy benchmark for {algorithm} from order {n0} to {n1}...")
    print(f"{'Order':<10} {'MSE':<18}")
    print("-" * 28)

    mses = []
    for n in range(n0, n1 + 1):
        cmd = [exe_path, str(n)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        mse = float(result.stdout)
        mses.append(mse)
        print(f"{n:<10} {mse:<18.8e}")

    print()
    print(mses)


def main():
    parser = argparse.ArgumentParser(description="Accuracy Benchmark for FFT")
    parser.add_argument("n0", type=int, help="Start FFT order (size 2^n)")
    parser.add_argument("n1", type=int, help="End FFT order (size 2^n)")
    parser.add_argument("algorithm", type=str, help="Algorithm to test (e.g., kfr)")
    parser.add_argument("--avx2", action="store_true", help="Enable AVX2 architecture")
    parser.add_argument("--double", action="store_true", help="Enable Double")

    args = parser.parse_args()

    try:
        exe_path = build_benchmark(args.algorithm, "accuracy",
                                   use_avx2=args.avx2, use_double=args.double, to_print=False)
        run_benchmark(exe_path, args.n0, args.n1, args.algorithm)
    except subprocess.CalledProcessError as e:
        print(f"Error executing benchmark: {e}")
        print(f"Stdout:\n{e.stdout}")
        print(f"Stderr:\n{e.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    main()
