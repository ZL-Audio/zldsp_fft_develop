import argparse
import subprocess
import os
import sys
import json
import time

from build_config import build_benchmark


def run_benchmark_collect(exe_path, n0, n1):
    cmd = [exe_path, str(n0), str(n1), "--benchmark_format=json"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        sys.stderr.write(f"Failed to run benchmark or parse JSON output: {e}\n")
        return None

    results = {
        "real to AoS RFFT forward": {},
        "real to SoA RFFT backward": {}
    }
    for bench in data.get("benchmarks", []):
        name = bench["name"]
        parts = name.rsplit('/', 1)
        if len(parts) != 2:
            continue
        bench_key, order_str = parts[0], parts[1]
        try:
            n = int(order_str)
        except ValueError:
            continue

        cpu_time_us = bench["cpu_time"]
        if cpu_time_us <= 0:
            throughput = 0.0
        else:
            ops = 2.5 * (2 ** n) * n
            throughput = ops / cpu_time_us

        if bench_key in results:
            results[bench_key][n] = throughput

    final_results = {}
    for bench_key in ["real to AoS RFFT forward", "real to SoA RFFT backward"]:
        th_list = []
        for n in range(n0, n1 + 1):
            th_list.append(results[bench_key].get(n, 0.0))
        final_results[bench_key] = th_list

    return final_results


def format_aligned_json(results, chunk_size=5):
    if not results:
        return "{}"

    max_key_len = max(len(str(k)) for k in results.keys())

    lines = ["{"]
    items = list(results.items())

    for i, (algo, th_list) in enumerate(items):
        key_str = f'  "{algo}":'
        padding_len = max_key_len + 6
        padded_key = f"{key_str:<{padding_len}}"

        num_strs = [f"{val:>10.4f}" for val in th_list]

        chunks = [num_strs[j:j + chunk_size] for j in range(0, len(num_strs), chunk_size)]

        for j, chunk in enumerate(chunks):
            chunk_str = ", ".join(chunk)
            is_last_chunk = (j == len(chunks) - 1)

            if j == 0:
                prefix = f"{padded_key} ["
            else:
                prefix = " " * (padding_len + 2)

            suffix = "]" if is_last_chunk else ","
            line = f"{prefix}{chunk_str}{suffix}"

            if is_last_chunk and i < len(items) - 1:
                line += ","

            lines.append(line)

    lines.append("}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Throughput Benchmark for zldsp RFFT (AoS vs SoA)")
    parser.add_argument("n0", type=int, help="Start FFT order (size 2^n)")
    parser.add_argument("n1", type=int, help="End FFT order (size 2^n)")
    parser.add_argument("--avx2", action="store_true", help="Enable AVX2 architecture")
    parser.add_argument("--double", action="store_true", help="Enable Double")

    args = parser.parse_args()

    try:
        exe_path = build_benchmark("zldsp", "throughput_zldsp_rfft",
                                   use_avx2=args.avx2, use_double=args.double, to_print=False)
        time.sleep(5)
        results = run_benchmark_collect(exe_path, args.n0, args.n1)
        if results:
            print(format_aligned_json(results, chunk_size=5))
        else:
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"Error executing benchmark: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
