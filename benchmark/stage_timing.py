import argparse
import subprocess
import os
import sys
import platform
import json
import time

from build_config import build_benchmark


def run_benchmark(exe_path, order, algorithm):
    print(f"Running stage timing benchmark for {algorithm} at order {order}...")
    cmd = [exe_path, str(order), "--benchmark_format=json"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Benchmark failed.\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        return

    # Print any non-JSON output (e.g. "Order: X, Total stages: Y")
    for line in result.stderr.strip().split('\n'):
        if line.strip():
            print(line)

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        print("Failed to parse JSON output.")
        print(result.stdout)
        return

    print(f"\n{'Stages':<10} {'Cumulative Time (us)':<25} {'Delta (us)':<15}")
    print("-" * 50)

    prev_time = 0.0
    cpu_times = []
    for bench in data.get("benchmarks", []):
        name = bench["name"]
        try:
            stages = int(name.split('/')[-1])
        except ValueError:
            continue

        cpu_time_us = bench["cpu_time"] * 0.5
        delta = cpu_time_us - prev_time
        cpu_times.append(cpu_time_us)

        print(f"{stages:<10} {cpu_time_us:<25.4f} {delta:<15.4f}")
        prev_time = cpu_time_us

    print()
    print(cpu_times)


def main():
    parser = argparse.ArgumentParser(description="Stage Timing Benchmark for FFT")
    parser.add_argument("order", type=int, help="FFT order (size 2^order)")
    parser.add_argument("algorithm", type=str, help="Algorithm to test (e.g., simd_low_order_aosoa1)")
    parser.add_argument("--avx2", action="store_true", help="Enable AVX2 architecture")

    args = parser.parse_args()

    try:
        exe_path = build_benchmark(args.algorithm, "stage_timing", use_avx2=args.avx2)
        time.sleep(10)
        run_benchmark(exe_path, args.order, args.algorithm)
    except subprocess.CalledProcessError as e:
        print(f"Error executing benchmark: {e}")
        print(f"Stdout:\n{e.stdout}")
        print(f"Stderr:\n{e.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    main()
