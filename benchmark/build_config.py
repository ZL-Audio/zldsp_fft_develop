"""Location-independent CMake builds for the benchmark command."""

import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALGOS = ("zldsp", "fftw3", "fftw3_estimate", "kfr", "vdsp",
         "pffft", "ipp", "armpl", "armpl_estimate")
BENCHMARKS = tuple(f"{mode}_{kind}" for mode in
                   ("accuracy", "accuracy_mid", "throughput", "throughput_zldsp")
                   for kind in ("cfft", "rfft"))


def get_algo_list(full=True):
    """List available backends; the standard set excludes both FFTW plans."""
    result = ["kfr", "pffft", "zldsp"]
    if full:
        result[0:0] = ["fftw3", "fftw3_estimate"]
    if platform.system() == "Darwin":
        result.insert(0, "vdsp")
    if os.environ.get("ARMPL_DIR"):
        result[0:0] = ["armpl", "armpl_estimate"]
    if platform.machine().lower() in ("x86_64", "amd64"):
        packages = Path.home() / ".nuget" / "packages"
        package = {"Darwin": "intelipp.mac", "Linux": "intelipp.static.linux-x64",
                   "Windows": "intelipp.static.win-x64"}.get(platform.system())
        if os.environ.get("IPP_ROOT") or os.environ.get("IPPROOT") or (package and (packages / package).is_dir()):
            result.insert(0, "ipp")
    return result


def run_command(command, cwd=None):
    result = subprocess.run([str(arg) for arg in command], cwd=cwd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {command}\n{result.stdout}\n{result.stderr}")
    return result


def benchmark_executable(build_dir, target):
    filename = target + (".exe" if platform.system() == "Windows" else "")
    for executable in (build_dir / filename, build_dir / "Release" / filename):
        if executable.is_file():
            return str(executable)
    raise RuntimeError(f"Missing {target} in {build_dir}; build it before using --skip-build")


def build_benchmark(algorithm, benchmark_type, use_avx2=False, use_double=False,
                    to_print=False, *, source_dir=ROOT, build_dir=None, use_sse2=False,
                    artifacts_dir=None, skip_build=False):
    if algorithm not in ALGOS:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    if benchmark_type not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {benchmark_type}")
    if use_avx2 and use_sse2:
        raise ValueError("AVX2 and SSE2 are mutually exclusive")
    source_dir = Path(source_dir).resolve()
    precision = "double" if use_double else "float"
    isa = "avx2" if use_avx2 else "sse2" if use_sse2 else "default"
    if build_dir is None:
        artifacts_dir = ROOT / "build_fft" if artifacts_dir is None else Path(artifacts_dir).expanduser()
        build_dir = artifacts_dir / f"{algorithm}-{precision}-{isa}"
    build_dir = Path(build_dir).resolve()
    target = f"zldsp_fft_{benchmark_type}"
    if skip_build:
        return benchmark_executable(build_dir, target)
    command = ["cmake", "-S", source_dir, "-B", build_dir,
               "-G", os.environ.get("CMAKE_GENERATOR", "Ninja"), "-DCMAKE_BUILD_TYPE=Release"]
    for name in BENCHMARKS:
        command.append(f"-D{name.upper()}_TEST={'ON' if name == benchmark_type else 'OFF'}")
    for name in ALGOS:
        command.append(f"-DENABLE_{name.upper()}={'ON' if name == algorithm else 'OFF'}")
    command += [f"-DUSE_AVX2={'ON' if use_avx2 else 'OFF'}",
                f"-DUSE_SSE2={'ON' if use_sse2 else 'OFF'}",
                f"-DUSE_DOUBLE={'ON' if use_double else 'OFF'}"]
    if platform.system() in ("Linux", "Windows"):
        default_c, default_cxx = ("clang", "clang++") if platform.system() == "Linux" else ("clang-cl", "clang-cl")
        command += [f"-DCMAKE_C_COMPILER={os.environ.get('CC', default_c)}",
                    f"-DCMAKE_CXX_COMPILER={os.environ.get('CXX', default_cxx)}"]
    if algorithm.startswith("armpl") and os.environ.get("ARMPL_DIR"):
        command.append(f"-DARMPL_DIR={os.environ['ARMPL_DIR']}")
    if to_print:
        print(f"Building {benchmark_type}: {algorithm}, {precision}, {isa}", flush=True)
    run_command(command)
    run_command(["cmake", "--build", build_dir, "--target", target, "--config", "Release", "--parallel"])
    return benchmark_executable(build_dir, target)
