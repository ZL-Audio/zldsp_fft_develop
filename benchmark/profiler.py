import argparse
import subprocess
import sys
import re
import os

from build_config import build_benchmark

def clean_sde_trace(file_path):
    print("Cleaning up SDE trace output...")
    with open(file_path, 'r') as f:
        lines = f.readlines()
    with open(file_path, 'w') as f:
        for line in lines:
            if line.startswith("INS "):
                parts = re.split(r'\s{2,}', line.strip())
                
                if len(parts) >= 3:
                    assembly = parts[-1]
                    f.write(f"{assembly}\n")
                else:
                    f.write(line)


def main():
    parser = argparse.ArgumentParser(description="Build FFT profiler binary for perf analysis")
    parser.add_argument("order", type=int, help="FFT order (size 2^order)")
    parser.add_argument("iterations", type=int, help="Number of iterations")
    parser.add_argument("algorithm", type=str, help="Algorithm to profile (e.g., simd_low_order_aosoa4)")
    parser.add_argument("--avx2", action="store_true", help="Enable AVX2 architecture")
    parser.add_argument("--build-only", action="store_true", help="Only build, do not run")

    parser.add_argument("--sde", type=str, metavar="PATH", help="Path to sde64 executable to automatically generate execution trace")

    args = parser.parse_args()

    try:
        exe_path = build_benchmark(args.algorithm, "profiler", use_avx2=args.avx2, to_print=True)
        print(f"Built: {exe_path}")

        if args.build_only:
            return

        cmd = [exe_path, str(args.order), str(args.iterations)]

        if args.sde:
            trace_file = f"trace_{args.algorithm}_order{args.order}.txt"
            sde_prefix = [
                args.sde,
                "-debugtrace",
                "-control", "start:enter_func:target_fft_execution",
                "-control", "stop:exit_func:target_fft_execution",
                "--"
            ]
            cmd = sde_prefix + cmd
            print(f"Running SDE Trace: {' '.join(cmd)}")
        else:
            print(f"Running: {' '.join(cmd)}")

        subprocess.run(cmd, check=True)

        if args.sde:
            print(f"\n[+] Trace successfully isolated and saved to: {trace_file}")
            default_out = "sde-debugtrace-out.txt"
            if os.path.exists(default_out):
                os.rename(default_out, trace_file)
                clean_sde_trace(trace_file)
                
                print(f"\n[+] Trace successfully isolated and cleaned: {trace_file}")
            else:
                print(f"\n[-] Warning: Expected SDE output file '{default_out}' not found.")

    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stdout:
            print(f"Stdout:\n{e.stdout}")
        if e.stderr:
            print(f"Stderr:\n{e.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    main()