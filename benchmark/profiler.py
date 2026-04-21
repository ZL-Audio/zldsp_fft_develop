import argparse
import subprocess
import sys

from build_config import build_benchmark


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

    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stdout:
            print(f"Stdout:\n{e.stdout}")
        if e.stderr:
            print(f"Stderr:\n{e.stderr}")
        sys.exit(1)


if __name__ == "__main__":
    main()