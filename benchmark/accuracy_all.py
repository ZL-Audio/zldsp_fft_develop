import argparse
import subprocess
import os
import sys
import platform
import shutil
import time

from build_config import get_algo_list, build_benchmark, replace_result_keys


def run_benchmark(exe_path, n0, n1, algorithm):
    print(f"Running accuracy benchmark for {algorithm} from order {n0} to {n1}...")
    
    mses = []
    for n in range(n0, n1 + 1):
        cmd = [exe_path, str(n)]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            mse = float(result.stdout.strip())
            mses.append(mse)
        except subprocess.CalledProcessError as e:
            print(f"Benchmark failed for {algorithm} at order {n}.\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
            return None
        except ValueError:
            print(f"Failed to parse MSE output for {algorithm} at order {n}. Output: {result.stdout}")
            return None

    return mses


def main():
    parser = argparse.ArgumentParser(description="Batch Accuracy Benchmark for FFT Algorithms")
    parser.add_argument("n0", type=int, help="Start FFT order (size 2^n)")
    parser.add_argument("n1", type=int, help="End FFT order (size 2^n)")
    parser.add_argument("--avx2", action="store_true", help="Enable AVX2 architecture")
    parser.add_argument("--full", action="store_true", help="Enable AVX2 architecture")

    args = parser.parse_args()

    algorithms = get_algo_list(args.full)

    results = {}

    for i, algo in enumerate(algorithms):
        print(f"\n[{i+1}/{len(algorithms)}] Building {algo}...")
        
        try:
            exe_path = build_benchmark(algo, "accuracy", use_avx2=args.avx2)
            time.sleep(1)
            accuracy_data = run_benchmark(exe_path, args.n0, args.n1, algo)
            
            if accuracy_data is not None:
                results[algo] = accuracy_data
            else:
                print(f"Skipping {algo} due to execution failure.")
                
        except Exception as e:
            print(f"Error building or running benchmark for {algo}: {e}")
            continue

    if os.path.exists("build_fft"):
        shutil.rmtree("build_fft")
        
    print("\n" + "="*50)
    print("FINAL ACCURACY RESULTS (MSE)")
    print("="*50)
    r = replace_result_keys(results)
            
    print(str(r).replace("],", "],\n"))


if __name__ == "__main__":
    main()