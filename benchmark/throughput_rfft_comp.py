import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path


DEFAULT_REPETITIONS = 6
TARGET_NAME = "zldsp_fft_throughput_rfft"
REQUIRED_SUBMODULES = ("google/benchmark", "google/highway")
TIME_UNIT_TO_US = {
    "ns": 1.0e-3,
    "us": 1.0,
    "ms": 1.0e3,
    "s": 1.0e6,
}


def run_command(cmd, cwd=None, capture_output=False):
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=capture_output,
        text=True,
    )
    if result.returncode != 0:
        stdout = result.stdout if capture_output else ""
        stderr = result.stderr if capture_output else ""
        raise RuntimeError(
            f"Command failed: {' '.join(str(arg) for arg in cmd)}\n"
            f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )
    return result


def git_output(repo_root, *args):
    result = run_command(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
    )
    return result.stdout.strip()


def repository_snapshot(repo_root):
    head = git_output(repo_root, "rev-parse", "HEAD")
    status = run_command(
        [
            "git", "-C", str(repo_root), "status", "--porcelain=v1", "-z",
            "--untracked-files=all", "--ignore-submodules=none",
        ],
        capture_output=True,
    ).stdout
    return head, status


def export_archive(repo_root, revision, destination, archive_path):
    destination.mkdir(parents=True, exist_ok=True)
    with archive_path.open("wb") as archive_file:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "archive", "--format=tar", revision],
            stdout=archive_file,
            stderr=subprocess.PIPE,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to export {revision} from {repo_root}:\n"
            f"{result.stderr.decode(errors='replace')}"
        )

    with tarfile.open(archive_path) as archive:
        if sys.version_info >= (3, 12):
            archive.extractall(destination, filter="data")
        else:
            archive.extractall(destination)
    archive_path.unlink()


def submodule_revision(repo_root, revision, submodule_path):
    entry = git_output(repo_root, "ls-tree", revision, "--", submodule_path)
    if not entry:
        raise RuntimeError(
            f"Revision {revision} does not contain submodule {submodule_path}"
        )

    fields = entry.split(None, 3)
    if len(fields) < 3 or fields[1] != "commit":
        raise RuntimeError(f"Invalid submodule entry: {entry}")
    return fields[2]


def ensure_submodule_commit(submodule_repo, submodule_path, revision):
    result = subprocess.run(
        [
            "git", "-C", str(submodule_repo), "cat-file", "-e",
            f"{revision}^{{commit}}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode == 0:
        return

    print(f"Fetching {submodule_path} revision {revision}...")
    run_command(
        [
            "git", "-C", str(submodule_repo), "fetch", "--depth=1",
            "origin", revision,
        ],
        capture_output=True,
    )


def export_revision(repo_root, revision, destination, temp_root):
    export_archive(
        repo_root,
        revision,
        destination,
        temp_root / f"{destination.name}.tar",
    )

    for submodule_path in REQUIRED_SUBMODULES:
        submodule_repo = repo_root / submodule_path
        if not submodule_repo.exists():
            raise RuntimeError(
                f"Submodule {submodule_path} is not initialized; run "
                "git submodule update --init --recursive"
            )

        submodule_commit = submodule_revision(
            repo_root, revision, submodule_path
        )
        ensure_submodule_commit(
            submodule_repo, submodule_path, submodule_commit
        )
        try:
            export_archive(
                submodule_repo,
                submodule_commit,
                destination / submodule_path,
                temp_root / f"{destination.name}-{submodule_path.replace('/', '-')}.tar",
            )
        except RuntimeError as error:
            raise RuntimeError(
                f"Failed to export {submodule_path} at {submodule_commit}. "
                "Ensure the submodule revision is available locally."
            ) from error


def build_benchmark(source_dir, build_dir, use_avx2, use_double):
    cmake_cmd = [
        "cmake",
        "-S", str(source_dir),
        "-B", str(build_dir),
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DTHROUGHPUT_RFFT_TEST=ON",
        "-DENABLE_ZLDSP=ON",
        f"-DUSE_AVX2={'ON' if use_avx2 else 'OFF'}",
        f"-DUSE_DOUBLE={'ON' if use_double else 'OFF'}",
    ]

    system = platform.system()
    if system == "Linux":
        cmake_cmd += [
            "-DCMAKE_C_COMPILER=clang",
            "-DCMAKE_CXX_COMPILER=clang++",
        ]
    elif system == "Windows":
        cmake_cmd += [
            "-DCMAKE_C_COMPILER=clang-cl",
            "-DCMAKE_CXX_COMPILER=clang-cl",
        ]

    run_command(cmake_cmd, capture_output=True)
    run_command(
        [
            "cmake", "--build", str(build_dir), "--config", "Release",
            "--target", TARGET_NAME, "--parallel",
        ],
        capture_output=True,
    )

    executable = build_dir / TARGET_NAME
    if system == "Windows":
        executable = executable.with_suffix(".exe")
    if not executable.exists():
        raise RuntimeError(f"Benchmark executable was not created: {executable}")
    return executable


def run_benchmark(executable, n0, n1):
    result = run_command(
        [str(executable), str(n0), str(n1), "--benchmark_format=json"],
        capture_output=True,
    )
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Failed to parse benchmark JSON from {executable}: {error}"
        ) from error

    expected_orders = set(range(n0, n1 + 1))
    throughputs = {}
    for benchmark in data.get("benchmarks", []):
        try:
            order = int(benchmark["name"].rsplit("/", 1)[1])
        except (KeyError, IndexError, ValueError):
            continue
        if order not in expected_orders:
            continue

        time_unit = benchmark.get("time_unit", "us")
        if time_unit not in TIME_UNIT_TO_US:
            raise RuntimeError(f"Unsupported benchmark time unit: {time_unit}")
        cpu_time_us = benchmark["cpu_time"] * TIME_UNIT_TO_US[time_unit]
        if cpu_time_us <= 0:
            raise RuntimeError(f"Non-positive CPU time for order {order}")

        operations = 2.5 * (2 ** order) * order
        throughputs[order] = operations / cpu_time_us

    missing_orders = sorted(expected_orders - throughputs.keys())
    if missing_orders:
        raise RuntimeError(
            f"Benchmark output is missing orders: {missing_orders}"
        )
    return throughputs


def print_raw_results(repetition, repetitions, current, previous):
    print(f"\nRaw throughput, repetition {repetition}/{repetitions} (MFLOPS)")
    print(f"{'Order':>7} {'Current':>15} {'Previous':>15}")
    print("-" * 39)
    for order in sorted(current):
        print(f"{order:>7} {current[order]:>15.4f} {previous[order]:>15.4f}")
    sys.stdout.flush()


def print_ratio_summary(current_results, previous_results, n0, n1):
    print("\nPer-order throughput ratio summary (current / previous)")
    header = (
        f"{'Order':>7} {'Current avg':>15} {'Previous avg':>15} "
        f"{'Geo ratio':>11} {'Avg ratio':>11} {'Median':>11} "
        f"{'MAD %':>11} {'Geo change':>12}"
    )
    print(header)
    print("-" * len(header))
    for order in range(n0, n1 + 1):
        current_values = [result[order] for result in current_results]
        previous_values = [result[order] for result in previous_results]
        ratios = [
            current / previous
            for current, previous in zip(current_values, previous_values)
        ]
        current_average = statistics.fmean(current_values)
        previous_average = statistics.fmean(previous_values)
        geometric_ratio = statistics.geometric_mean(ratios)
        average_ratio = statistics.fmean(ratios)
        median_ratio = statistics.median(ratios)
        log_ratios = [math.log(ratio) for ratio in ratios]
        median_log_ratio = statistics.median(log_ratios)
        log_ratio_mad = statistics.median(
            abs(log_ratio - median_log_ratio) for log_ratio in log_ratios
        )
        ratio_mad_percent = 100.0 * math.expm1(log_ratio_mad)
        change = (geometric_ratio - 1.0) * 100.0
        print(
            f"{order:>7} {current_average:>15.4f} {previous_average:>15.4f} "
            f"{geometric_ratio:>11.6f} {average_ratio:>11.6f} "
            f"{median_ratio:>11.6f} {ratio_mad_percent:>10.2f}% "
            f"{change:>+11.2f}%"
        )


def compare_revisions(
    repo_root, n0, n1, repetitions, use_avx2, use_double
):
    current_revision = git_output(repo_root, "rev-parse", "HEAD")
    previous_revision = git_output(repo_root, "rev-parse", "HEAD^")

    print(f"Current revision:  {current_revision}")
    print(f"Previous revision: {previous_revision}")
    print(f"Orders: {n0} through {n1}")
    print(f"Precision: {'double' if use_double else 'float'}")
    print(f"AVX2: {'enabled' if use_avx2 else 'disabled'}")
    print(f"Repetitions: {repetitions}")
    print("Both revisions are built from isolated temporary source trees.")
    sys.stdout.flush()

    with tempfile.TemporaryDirectory(prefix="zldsp-rfft-comp-") as temp_dir:
        temp_root = Path(temp_dir)
        sources = {
            "current": temp_root / "source-current",
            "previous": temp_root / "source-previous",
        }
        revisions = {
            "current": current_revision,
            "previous": previous_revision,
        }

        for label in ("current", "previous"):
            print(f"Exporting {label} revision...")
            export_revision(
                repo_root,
                revisions[label],
                sources[label],
                temp_root,
            )

        executables = {}
        for label in ("current", "previous"):
            print(f"Building {label} revision...")
            sys.stdout.flush()
            executables[label] = build_benchmark(
                sources[label],
                temp_root / f"build-{label}",
                use_avx2,
                use_double,
            )

        time.sleep(5)
        results = {"current": [], "previous": []}
        for repetition in range(1, repetitions + 1):
            if repetition % 2 == 1:
                run_order = ("current", "previous")
            else:
                run_order = ("previous", "current")
            print(f"\nRepetition {repetition} order: {' -> '.join(run_order)}")

            repetition_results = {}
            for label in run_order:
                throughput = run_benchmark(executables[label], n0, n1)
                repetition_results[label] = throughput

            print_raw_results(
                repetition,
                repetitions,
                repetition_results["current"],
                repetition_results["previous"],
            )

            for label in ("current", "previous"):
                results[label].append(repetition_results[label])

        print_ratio_summary(
            results["current"], results["previous"], n0, n1
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compare RFFT throughput for HEAD and HEAD^"
    )
    parser.add_argument("n0", type=int, help="Start FFT order (size 2^n)")
    parser.add_argument("n1", type=int, help="End FFT order (size 2^n)")
    parser.add_argument("--avx2", action="store_true", help="Enable AVX2")
    parser.add_argument("--double", action="store_true", help="Use double precision")
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPETITIONS,
        help=f"Number of comparison repetitions (default: {DEFAULT_REPETITIONS})",
    )
    args = parser.parse_args()

    if args.n0 < 1:
        parser.error("n0 must be positive")
    if args.n0 > args.n1:
        parser.error("n0 must not exceed n1")
    if args.repeats < 1:
        parser.error("repeats must be positive")

    script_dir = Path(__file__).resolve().parent
    repo_root = Path(git_output(script_dir, "rev-parse", "--show-toplevel"))
    initial_snapshot = repository_snapshot(repo_root)
    if initial_snapshot[1]:
        print(
            "Note: uncommitted changes are not benchmarked; this compares HEAD and HEAD^.",
            file=sys.stderr,
        )

    exit_code = 0
    try:
        compare_revisions(
            repo_root,
            args.n0,
            args.n1,
            args.repeats,
            args.avx2,
            args.double,
        )
    except KeyboardInterrupt:
        print("Benchmark interrupted.", file=sys.stderr)
        exit_code = 130
    except Exception as error:
        print(f"Comparison failed: {error}", file=sys.stderr)
        exit_code = 1

    final_snapshot = repository_snapshot(repo_root)
    if final_snapshot != initial_snapshot:
        print(
            "Comparison failed: repository state changed during the benchmark.",
            file=sys.stderr,
        )
        exit_code = 1
    else:
        print("\nRepository state is unchanged.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
