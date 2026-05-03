import argparse
import subprocess
import os
import sys

from build_config import build_benchmark


def run_benchmark(exe_path, n0, n1, algorithm, threshold):
    print(f"Running accuracy CFFT benchmark for {algorithm} from order {n0} to {n1}...")
    print(f"{'Order':<10} {'Fwd Max MSE':<18} {'Bwd Max MSE':<18} {'Id Max MSE':<18}")
    print("-" * 66)

    fwd_mses = []
    bwd_mses = []
    id_mses = []
    for n in range(n0, n1 + 1):
        cmd = [exe_path, str(n)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        parts = result.stdout.split()
        mse_fwd = float(parts[0])
        mse_bwd = float(parts[1])
        mse_id = float(parts[2])
        fwd_mses.append(mse_fwd)
        bwd_mses.append(mse_bwd)
        id_mses.append(mse_id)
        print(f"{n:<10} {mse_fwd:<18.8e} {mse_bwd:<18.8e} {mse_id:<18.8e}")

    print()
    print("Forward MSEs:", fwd_mses)
    print("Backward MSEs:", bwd_mses)
    print("Identity MSEs:", id_mses)

    for mse in fwd_mses:
        if mse > threshold:
            print(f"\nERROR: One or more MSE values exceeded the threshold of {threshold}")
            sys.exit(1)

    for mse in bwd_mses:
        if mse > threshold:
            print(f"\nERROR: One or more MSE values exceeded the threshold of {threshold}")
            sys.exit(1)

    for mse in bwd_mses:
        if mse > threshold:
            print(f"\nERROR: One or more MSE values exceeded the threshold of {threshold}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Accuracy CFFT Benchmark")
    parser.add_argument("n0", type=int, help="Start FFT order (size 2^n)")
    parser.add_argument("n1", type=int, help="End FFT order (size 2^n)")
    parser.add_argument("algorithm", type=str, help="Algorithm to test (e.g., zldsp)")
    parser.add_argument("--avx2", action="store_true", help="Enable AVX2 architecture")
    parser.add_argument("--double", action="store_true", help="Enable Double")
    parser.add_argument("--threshold", type=float, default=1e-5, help="Failure threshold for MSE")

    args = parser.parse_args()

    try:
        exe_path = build_benchmark(args.algorithm, "accuracy_cfft",
                                   use_avx2=args.avx2, use_double=args.double, to_print=False)
        run_benchmark(exe_path, args.n0, args.n1, args.algorithm, args.threshold)
    except subprocess.CalledProcessError as e:
        print(f"Error executing benchmark: {e}")
        print(f"Stdout:\n{e.stdout}")
        print(f"Stderr:\n{e.stderr}")
        sys.exit(1)

    print("Test Pass!")


if __name__ == "__main__":
    main()
