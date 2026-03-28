import argparse
import subprocess
import os
import sys
import platform
import shutil
import json
import time

from build_config import get_algo_list, build_benchmark, replace_result_keys


def run_benchmark(exe_path, n0, n1, algorithm):
    print(f"Running throughput benchmark for {algorithm} from order {n0} to {n1}...")
    cmd = [exe_path, str(n0), str(n1), "--benchmark_format=json"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Benchmark failed for {algorithm}.\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        return None

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON output for {algorithm}.")
        return None

    throughputs = []
    for bench in data.get("benchmarks", []):
        name = bench["name"]
        try:
            n = int(name.split('/')[-1])
        except ValueError:
            continue

        cpu_time_us = bench["cpu_time"] * 0.5

        ops = 5 * (2 ** n) * n
        throughput = ops / cpu_time_us
        throughputs.append(throughput)

    return throughputs


def main():
    parser = argparse.ArgumentParser(description="Batch Throughput Benchmark for FFT Algorithms")
    parser.add_argument("n0", type=int, help="Start FFT order (size 2^n)")
    parser.add_argument("n1", type=int, help="End FFT order (size 2^n)")
    parser.add_argument("--avx2", action="store_true", help="Enable AVX2 architecture")
    parser.add_argument("--full", action="store_true", help="Enable AVX2 architecture")

    args = parser.parse_args()

    algorithms = get_algo_list(args.full)

    results = {}

    for i, algo in enumerate(algorithms):
        if i > 0:
            print("\nSleeping 10 seconds to let the CPU cool down...")
            time.sleep(10)
            
        print(f"\n[{i+1}/{len(algorithms)}] Building {algo}...")
        
        try:
            exe_path = build_benchmark(algo, "throughput", use_avx2=args.avx2)
            time.sleep(10)
            throughput_data = run_benchmark(exe_path, args.n0, args.n1, algo)
            
            if throughput_data is not None:
                results[algo] = throughput_data
            else:
                print(f"Skipping {algo} due to execution failure.")
                
        except Exception as e:
            print(f"Error building or running benchmark for {algo}: {e}")
            continue
    shutil.rmtree("build_fft")
    print("\n" + "="*50)
    print("FINAL BENCHMARK RESULTS")
    print("="*50)
    r = replace_result_keys(results)
    print(str(r).replace("],", "],\n"))


if __name__ == "__main__":
    main()