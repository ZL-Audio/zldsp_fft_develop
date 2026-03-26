import argparse
import subprocess
import os
import sys
import platform
import json
import time

from build_config import build_benchmark


def run_benchmark(exe_path, order, algorithm):
    cmd = [exe_path, str(order), "--benchmark_format=json"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Benchmark failed for order {order}.\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        return None

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON output for order {order}.")
        return None

    cpu_times = []
    for bench in data.get("benchmarks", []):
        name = bench["name"]
        try:
            stages = int(name.split('/')[-1])
        except ValueError:
            continue
        cpu_time_us = bench["cpu_time"] * 0.5
        cpu_times.append(cpu_time_us)

    return cpu_times


def main():
    parser = argparse.ArgumentParser(description="Stage Timing Benchmark for FFT (all orders)")
    parser.add_argument("n0", type=int, help="Start FFT order (size 2^n)")
    parser.add_argument("n1", type=int, help="End FFT order (size 2^n)")
    parser.add_argument("algorithm", type=str, help="Algorithm to test (e.g., simd_low_order_aosoa1)")
    parser.add_argument("--avx2", action="store_true", help="Enable AVX2 architecture")

    args = parser.parse_args()

    try:
        exe_path = build_benchmark(args.algorithm, "stage_timing", use_avx2=args.avx2)
    except subprocess.CalledProcessError as e:
        print(f"Error building benchmark: {e}")
        print(f"Stdout:\n{e.stdout}")
        print(f"Stderr:\n{e.stderr}")
        sys.exit(1)

    time.sleep(10)

    results = {}
    throughputs = []
    for order in range(args.n0, args.n1 + 1):
        print(f"\nRunning stage timing for order {order}...")
        cpu_times = run_benchmark(exe_path, order, args.algorithm)
        if cpu_times is not None:
            results[order] = cpu_times

            full_time = cpu_times[-1]
            ops = 5 * (2 ** order) * order
            throughput = ops / full_time
            throughputs.append(throughput)

            print(f"  Order {order}: {len(cpu_times)} stages, throughput {throughput:.4f} MFLOPS")
            prev = 0.0
            for i, t in enumerate(cpu_times):
                delta = t - prev
                print(f"    Stage {i+1}: {t:.4f} us (delta {delta:.4f})")
                prev = t
        else:
            throughputs.append(None)

        if order < args.n1:
            time.sleep(5)

    print("\n" + "=" * 60)
    print(f"STAGE TIMING RESULTS for {args.algorithm}")
    print("=" * 60)
    print(str(results).replace("],", "],\n"))

    print(f"\n{'Order':<10} {'Time (us)':<15} {'Throughput (MFLOPS)':<20}")
    print("-" * 45)
    for i, order in enumerate(range(args.n0, args.n1 + 1)):
        if order in results:
            full_time = results[order][-1]
            tp = throughputs[i]
            print(f"{order:<10} {full_time:<15.4f} {tp:<20.4f}")

    print()
    print(throughputs)


if __name__ == "__main__":
    main()
