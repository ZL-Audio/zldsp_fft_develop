import argparse
import subprocess
import os
import sys
import json
import time

from build_config import build_benchmark, get_algo_list, replace_result_keys

def run_benchmark_collect(exe_path, n0, n1):
    cmd = [exe_path, str(n0), str(n1), "--benchmark_format=json"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None

    results = {}
    for bench in data.get("benchmarks", []):
        name = bench["name"]
        try:
            n = int(name.split('/')[-1])
        except ValueError:
            continue
        
        cpu_time_us = bench["cpu_time"]
        if cpu_time_us <= 0:
            results[n] = 0.0
            continue

        ops = 2.5 * (2 ** n) * n
        throughput = ops / cpu_time_us
        results[n] = throughput
    
    throughput_list = []
    for n in range(n0, n1 + 1):
        throughput_list.append(results.get(n, 0.0))
    return throughput_list

def main():
    parser = argparse.ArgumentParser(description="Throughput All Benchmark for RFFT")
    parser.add_argument("n0", type=int, help="Start FFT order (size 2^n)")
    parser.add_argument("n1", type=int, help="End FFT order (size 2^n)")
    parser.add_argument("--avx2", action="store_true", help="Enable AVX2 architecture")
    parser.add_argument("--double", action="store_true", help="Enable Double")
    parser.add_argument("--full", action="store_true", help="Test full algorithm list")

    args = parser.parse_args()

    algo_list = get_algo_list(full=args.full)
    raw_results = {}

    for algo in algo_list:
        sys.stderr.write(f"Benchmarking {algo}...\n")
        try:
            exe_path = build_benchmark(algo, "throughput_rfft",
                                       use_avx2=args.avx2, use_double=args.double, to_print=False)
            time.sleep(5)
            th_list = run_benchmark_collect(exe_path, args.n0, args.n1)
            if th_list:
                raw_results[algo] = th_list
            else:
                sys.stderr.write(f"Failed to run benchmark for {algo}\n")
        except Exception as e:
            sys.stderr.write(f"Failed to build/run {algo}: {e}\n")

    final_results = replace_result_keys(raw_results)
    print(json.dumps(final_results, indent=4))

if __name__ == "__main__":
    main()
