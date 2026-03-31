import argparse
import subprocess
import os
import sys
import platform
import json
import time

from build_config import build_benchmark


def run_benchmark(exe_path, n0, n1, algorithm):
    print(f"Running throughput benchmark for {algorithm} from order {n0} to {n1}...")
    cmd = [exe_path, str(n0), str(n1), "--benchmark_format=json"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Benchmark failed.\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        return

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        print("Failed to parse JSON output.")
        return

    print(f"{'Order':<10} {'Time (us)':<15} {'Throughput (MFLOPS)':<20}")
    print("-" * 45)

    cpu_times = []
    throughputs = []
    for bench in data.get("benchmarks", []):
        name = bench["name"]
        try:
            n = int(name.split('/')[-1])
        except ValueError:
            continue

        cpu_time_us = bench["cpu_time"] * 0.5
        cpu_times.append(cpu_time_us)

        ops = 5 * (2 ** n) * n
        throughput = ops / cpu_time_us
        throughputs.append(throughput)

        print(f"{n:<10} {cpu_time_us:<15.4f} {throughput:<20.4f}")

    print()
    print(cpu_times)
    print(throughputs)


def main():
    parser = argparse.ArgumentParser(description="Throughput Benchmark for FFT")
    parser.add_argument("n0", type=int, help="Start FFT order (size 2^n)")
    parser.add_argument("n1", type=int, help="End FFT order (size 2^n)")
    parser.add_argument("algorithm", type=str, help="Algorithm to test (e.g., naive_stockham_radix2, kfr)")
    parser.add_argument("--avx2", action="store_true", help="Enable AVX2 architecture")
    parser.add_argument("--double", action="store_true", help="Enable Double")

    args = parser.parse_args()

    try:
        exe_path = build_benchmark(args.algorithm, "throughput",
                                   use_avx2=args.avx2, use_double=args.double, to_print=False)
        time.sleep(10)
        run_benchmark(exe_path, args.n0, args.n1, args.algorithm)
    except subprocess.CalledProcessError as e:
        print(f"Error executing benchmark: {e}")
        print(f"Stdout:\n{e.stdout}")
        print(f"Stderr:\n{e.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    main()
