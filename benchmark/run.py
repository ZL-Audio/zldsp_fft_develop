"""Build and run FFT accuracy, throughput, and paired commit comparisons."""

import argparse
import json
import math
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from build_config import ALGOS, ROOT, build_benchmark, get_algo_list, run_command

TIME_TO_US = {"ns": 1e-3, "us": 1.0, "ms": 1e3, "s": 1e6}
REQUIRED_SUBMODULES = ("google/benchmark", "google/highway")
DIRECTIONS = ("forward", "backward", "convolve", "roundtrip")
DEFAULT_MIN_TIME0 = 0.5
DEFAULT_MIN_TIME1 = 5.0
METRICS = {
    "throughput": (("mflops", "MFLOPS (higher is faster)", ".1f"),),
    "accuracy": (("relative_rmse", "Relative RMSE (lower is better)", ".3e"),
                 ("max_abs", "Maximum absolute error (lower is better)", ".3e")),
    "commit_comparison": (("current_us", "Current time (us)", ".6g"),
                          ("previous_us", "Previous time (us)", ".6g"),
                          ("throughput_ratio", "Throughput ratio", ".6g"),
                          ("change_percent", "Change (%)", ".6g"),
                          ("median_ratio", "Median ratio", ".6g"),
                          ("mad_percent", "MAD (%)", ".6g")),
}
DISPLAY_NAMES = {
    "fftw3": "FFTW measure", "fftw3_estimate": "FFTW estimate", "kfr": "KFR",
    "pffft": "PFFFT", "vdsp": "vDSP", "ipp": "IPP",
    "armpl": "ArmPL measure", "armpl_estimate": "ArmPL estimate",
    "aos_aos": "AoS->AoS", "aos_soa": "AoS->SoA", "soa_aos": "SoA->AoS", "soa_soa": "SoA->SoA",
    "real_aos": "real->AoS", "real_soa": "real->SoA", "real_power": "real->power",
    "aos_real": "AoS->real", "soa_real": "SoA->real", "aos": "AoS",
    "complex_aos": "complex AoS fused", "complex_soa": "complex SoA fused", "real": "real fused",
    "complex_aos_separate": "complex AoS separate", "complex_soa_separate": "complex SoA separate",
    "real_separate": "real separate",
}


def case_keys(kind, all_apis=False, convolve_baselines=False):
    if kind == "rfft":
        cases = {("forward", "real_aos"), ("backward", "aos_real")}
        if all_apis:
            cases |= {("forward", "real_soa"), ("backward", "soa_real"), ("forward", "real_power")}
            cases |= {("convolve", api) for api in ("complex_aos", "complex_soa", "real")}
            if convolve_baselines:
                cases |= {("convolve", api + "_separate") for api in ("complex_aos", "complex_soa", "real")}
        return cases
    apis = ["aos_aos", "aos_soa", "soa_aos", "soa_soa"] if all_apis else ["aos_aos"]
    return {(direction, api) for direction in ("forward", "backward") for api in apis}


def parse_throughput(data, kind, n0, n1, all_apis=False):
    """One row per order/direction/API; use median iteration time across repeats."""
    expected = {(direction, api, order) for direction, api in case_keys(kind, all_apis, convolve_baselines=True)
                for order in range(n0, n1 + 1)}
    samples, skipped = {}, {}
    for entry in data.get("benchmarks", []):
        # Aggregates include percentages (CV) as well as time. Never parse them
        # as individual timings or mix them into the repetition median.
        if entry.get("run_type") == "aggregate":
            continue
        parts = entry.get("run_name", entry.get("name", "")).split("/")
        # Google Benchmark appends the per-order MinTime setting to the name.
        if len(parts) == 4 and parts[-1].startswith("min_time:"):
            parts.pop()
        if len(parts) != 3:
            raise ValueError(f"Unexpected benchmark name: {entry.get('name')}")
        key = (parts[0], parts[1], int(parts[2]))
        if key not in expected:
            raise ValueError(f"Unexpected benchmark case: {key}")
        if entry.get("error_occurred"):
            reason = entry.get("error_message", "benchmark failed")
            if not reason.startswith("unsupported size:"):
                raise RuntimeError(f"{key}: {reason}")
            skipped[key] = reason
            continue
        convolution = key[0] == "convolve"
        if entry.get("transforms_per_iteration") != (2 if convolution else 1):
            raise ValueError(f"{key}: incorrect transforms per iteration")
        if convolution and entry.get("convolutions_per_iteration") != 1:
            raise ValueError(f"{key}: expected one convolution per iteration")
        unit = entry.get("time_unit")
        if unit not in TIME_TO_US:
            raise ValueError(f"Unsupported benchmark time unit: {unit}")
        try:
            duration = float(entry["cpu_time"]) * TIME_TO_US[unit]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{key}: invalid CPU time") from error
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"{key}: CPU time must be finite and positive")
        samples.setdefault(key, []).append(duration)
    if skipped.keys() & samples.keys():
        raise ValueError("A case was both measured and skipped")
    missing = expected - samples.keys() - skipped.keys()
    if missing:
        raise ValueError(f"Missing benchmark cases: {sorted(missing)}")
    rows = []
    for direction, api, order in sorted(expected, key=lambda key: (key[2], key[0], key[1])):
        key = (direction, api, order)
        row = dict(order=order, direction=direction, api=api)
        if key in skipped:
            row.update(status="skipped", reason=skipped[key])
        else:
            duration = statistics.median(samples[key])
            n = 2 ** order
            if direction == "convolve":
                # Same conventional work estimate for fused and separate paths;
                # this is a throughput unit, not an actual instruction count.
                operations = 5 * n * order + (n if api.startswith("real") else 3 * n)
            else:
                operations = (2.5 if kind == "rfft" else 5) * n * order
            mflops = operations / duration
            if not math.isfinite(mflops):
                raise ValueError(f"{key}: non-finite throughput")
            row.update(status="measured", cpu_time_us=duration, mflops=mflops,
                       samples=len(samples[key]))
        rows.append(row)
    return rows


def run_throughput(executable, kind, n0, n1, all_apis=False, repeats=3,
                   min_time0=DEFAULT_MIN_TIME0, min_time1=DEFAULT_MIN_TIME1):
    selected = sorted(case_keys(kind, all_apis, convolve_baselines=True))
    case_filter = ("^(" + "|".join(f"{direction}/{api}" for direction, api in selected)
                   + ")/[0-9]+(/min_time:[^/]+)?$")
    # Override an environment-supplied iteration count; each case's MinTime
    # takes precedence over this global seconds setting.
    result = run_command([executable, n0, n1, f"--min-time0={min_time0}", f"--min-time1={min_time1}",
                          "--benchmark_format=json",
                          f"--benchmark_filter={case_filter}", "--benchmark_enable_random_interleaving=false",
                          f"--benchmark_repetitions={repeats}", f"--benchmark_min_time={min_time0}s"])
    data = json.loads(result.stdout)
    return parse_throughput(data, kind, n0, n1, all_apis)


def validate_args(parser, args):
    if not 1 <= args.n0 <= args.n1 <= 30:
        parser.error("orders must satisfy 1 <= n0 <= n1 <= 30")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    for name in ("min_time0", "min_time1", "threshold"):
        value = getattr(args, name, None)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")


def print_tables(args, rows):
    """Compare one metric at a time, with libraries (or APIs) as columns."""
    column = "api" if args.apis else "library"
    groups = {(row["direction"], None if args.apis else row["api"]) for row in rows}
    for direction, api in sorted(groups, key=lambda group: (DIRECTIONS.index(group[0]), group[1] or "")):
        group = [row for row in rows if row["direction"] == direction and (args.apis or row["api"] == api)]
        columns = list(dict.fromkeys(row[column] for row in group))
        if args.apis:
            columns.sort(key=lambda name: (name == "real_power", name))
        indexed = {(row["order"], row[column]): row for row in group}
        orders = range(args.n0, args.n1 + 1)
        if len(indexed) != len(group) or set(indexed) != {(order, name) for order in orders for name in columns}:
            raise ValueError("Comparison table has duplicate or missing results")
        scope = "zldsp APIs" if args.apis else DISPLAY_NAMES.get(api, api)
        precision = "double" if args.double else "float"
        print(f"\n{args.kind.upper()} {precision} {direction} ({scope})")
        for metric, title, spec in METRICS[args.operation]:
            print(title)
            table = [["Order"] + [DISPLAY_NAMES.get(name, name) for name in columns]]
            for order in orders:
                cells = []
                for name in columns:
                    row = indexed[order, name]
                    cells.append("SKIP" if row["status"] == "skipped" else format(row[metric], spec))
                table.append([str(order)] + cells)
            widths = [max(5 if i == 0 else 12, *(len(cell) for cell in values))
                      for i, values in enumerate(zip(*table))]
            for i, cells in enumerate(table):
                print("  ".join(cell.rjust(width) for cell, width in zip(cells, widths)))
                if i == 0:
                    print("  ".join("-" * width for width in widths))
            print()
    skipped = {}
    for row in rows:
        if row["status"] == "skipped":
            key = row["library"], row["direction"], row["api"], row["reason"]
            skipped.setdefault(key, []).append(str(row["order"]))
    for (library, direction, api, reason), orders in skipped.items():
        print(f"SKIP {library} {direction}/{api}, orders {', '.join(orders)}: {reason}")


def format_json_value(value, float_spec=None):
    if float_spec and isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Out of range float values are not JSON compliant")
        return format(value, float_spec)
    return json.dumps(value, allow_nan=False)


def compact_json(value, indent=0, float_spec=None):
    """Keep numeric series in short chunks."""
    padding = " " * indent
    if isinstance(value, dict):
        if not value:
            return "{}"
        lines = [f"{padding}  {json.dumps(key)}: {compact_json(item, indent + 2, float_spec)}"
                 for key, item in value.items()]
        return "{\n" + ",\n".join(lines) + "\n" + padding + "}"
    if isinstance(value, list):
        if any(isinstance(item, (dict, list)) for item in value):
            lines = [padding + "  " + compact_json(item, indent + 2, float_spec) for item in value]
            return "[\n" + ",\n".join(lines) + "\n" + padding + "]"
        items = [format_json_value(item, float_spec) for item in value]
        if len(items) <= 5:
            return "[" + ", ".join(items) + "]"
        chunks = [", ".join(items[i:i + 5]) for i in range(0, len(items), 5)]
        return "[\n" + ",\n".join(padding + "  " + chunk for chunk in chunks) + "\n" + padding + "]"
    return format_json_value(value, float_spec)


def save_results(args, kind, operation, rows, **metadata):
    context = dict(kind=kind, operation=operation, precision="double" if args.double else "float",
                   isa="avx2" if args.avx2 else "sse2" if args.sse2 else "default",
                   n0=args.n0, n1=args.n1)
    if args.json:
        payload = dict(context, results=rows, **metadata)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    metrics = {}
    for metric, _, _ in METRICS[operation]:
        cases = metrics[metric] = {}
        for row in sorted(rows, key=lambda row: (DIRECTIONS.index(row["direction"]), row["api"] == "real_power",
                                                 row["api"], row["order"])):
            case = f"{row['direction']}/{row['api']}"
            libraries = cases.setdefault(case, {})
            values = libraries.setdefault(row.get("library", "zldsp"), [None] * (args.n1 - args.n0 + 1))
            values[row["order"] - args.n0] = (None if row.get("status") == "skipped"
                                              else float(row[metric]) if operation == "throughput" else row[metric])
    # Keep full per-case rows and paired runs in --json; the console summary
    # uses compact order-aligned arrays, as the original benchmark scripts did.
    summary = dict(context, **{key: value for key, value in metadata.items() if key != "runs"},
                   orders=list(range(args.n0, args.n1 + 1)), metrics=metrics)
    skipped = [row for row in rows if row.get("status") == "skipped"]
    if skipped:
        summary["skipped"] = skipped
    print("\nJSON summary:")
    print(compact_json(summary, float_spec=".1f" if operation == "throughput" else None))


def throughput(args):
    kind, all_apis = args.kind, args.apis
    algorithms = (get_algo_list(full=args.all_libraries) if args.standard_libraries or args.all_libraries
                  else args.libraries or ["zldsp"])
    algorithms = list(dict.fromkeys(algorithms))
    executables = {}
    for algorithm in algorithms:
        action = "Using prebuilt" if args.skip_build else "Building"
        print(f"{action} {kind.upper()} {algorithm}...", file=sys.stderr, flush=True)
        # Common zldsp cases and --apis use exactly the same optimized binary.
        target = f"throughput_{'zldsp_' if algorithm == 'zldsp' else ''}{kind}"
        executables[algorithm] = build_benchmark(algorithm, target, args.avx2, args.double,
                                               use_sse2=args.sse2, artifacts_dir=args.artifacts_dir,
                                               skip_build=args.skip_build)
    if args.build_only:
        print("Builds complete. Use --skip-build with the same library, precision, ISA and artifacts options to measure.")
        return
    rows = []
    for order in range(args.n0, args.n1 + 1):
        min_time = min_time_for_order(order, args.n0, args.n1, args.min_time0, args.min_time1)
        for algorithm, executable in executables.items():
            print(f"Measuring {kind.upper()} {algorithm}, order {order} "
                  f"({args.repeats} repetitions, minimum {min_time:g}s per case/repetition)...",
                  file=sys.stderr, flush=True)
            results = run_throughput(executable, kind, order, order, all_apis, args.repeats,
                                     min_time, min_time)
            for row in results:
                row["library"] = algorithm
            rows.extend(results)
    print_tables(args, rows)
    save_results(args, kind, "throughput", rows)


def min_time_for_order(order, n0, n1, min_time0, min_time1):
    if order == n0 or min_time0 == min_time1:
        return min_time0
    if order == n1:
        return min_time1
    fraction = (order - n0) / (n1 - n0)
    return math.exp((1 - fraction) * math.log(min_time0) + fraction * math.log(min_time1))


def timing_description(args):
    if args.n0 == args.n1 or args.min_time0 == args.min_time1:
        return f"minimum {args.min_time0:g}s per case/repetition"
    return (f"minimum {args.min_time0:g}s at order {args.n0} to {args.min_time1:g}s at order {args.n1} "
            "per case/repetition, exponential scaling")


def parse_accuracy(output, kind, order, all_apis, threshold):
    rows = [json.loads(line) for line in output.splitlines() if line.strip()]
    if not rows:
        raise ValueError("Accuracy executable returned no results")
    seen = set()
    libraries = {row.get("library") for row in rows}
    if all_apis and libraries != {"zldsp"}:
        raise ValueError("Unexpected library in API accuracy results")
    if not all_apis and libraries != {"fftw3_estimate", "kfr", "zldsp"}:
        raise ValueError("Library accuracy must contain only FFTW estimate, KFR and zldsp")
    for row in rows:
        if row.get("order") != order or row.get("status") not in ("passed", "failed"):
            raise ValueError(f"Invalid accuracy row: {row}")
        key = (row["library"], row["direction"], row["api"])
        if key in seen:
            raise ValueError(f"Duplicate accuracy result: {key}")
        seen.add(key)
        for name in ("relative_rmse", "max_abs"):
            if not math.isfinite(row[name]) or row[name] < 0:
                raise ValueError(f"Non-finite or negative accuracy error: {key}")
        if row["status"] == "failed" or row["relative_rmse"] > threshold:
            raise ValueError(f"Accuracy tolerance exceeded: {row}")
    cases = case_keys(kind, True) if all_apis else {("forward", "median"), ("backward", "median")}
    cases = cases | {("roundtrip", "aos")}
    expected = {(library, direction, api) for library in libraries for direction, api in cases}
    if seen != expected:
        raise ValueError(f"Accuracy case coverage mismatch; missing={expected - seen}, extra={seen - expected}")
    return rows


def accuracy(args):
    kind, all_apis = args.kind, args.apis
    rows = []
    scope = "zldsp APIs" if all_apis else "library agreement"
    print(f"Checking {kind.upper()} {scope}, orders {args.n0}-{args.n1}...", file=sys.stderr, flush=True)
    executable = build_benchmark("zldsp", f"accuracy_{'' if all_apis else 'mid_'}{kind}",
                                 args.avx2, args.double, use_sse2=args.sse2, artifacts_dir=args.artifacts_dir)
    for order in range(args.n0, args.n1 + 1):
        result = run_command([executable, order, args.repeats, 42, args.threshold])
        results = parse_accuracy(result.stdout, kind, order, all_apis, args.threshold)
        rows.extend(results)
        print(f"Order {order}: passed", file=sys.stderr, flush=True)
    print_tables(args, rows)
    print("All accuracy checks passed.")
    reference = "aos API" if all_apis else "component-wise library median"
    if all_apis and kind == "rfft":
        reference += "; convolution: separate path and small direct DFT"
    save_results(args, kind, "accuracy", rows, threshold=args.threshold,
                 reference=reference)


def git_output(repo, *args):
    return run_command(["git", "-C", repo, *args]).stdout.strip()


def repository_snapshot(repo):
    return (git_output(repo, "rev-parse", "HEAD"),
            git_output(repo, "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"))


def export_archive(repo, revision, destination, archive_path):
    destination.mkdir(parents=True, exist_ok=True)
    with archive_path.open("wb") as output:
        result = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", revision],
                                stdout=output, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(f"Cannot export {revision} from {repo}: {result.stderr.decode(errors='replace')}")
    with tarfile.open(archive_path) as archive:
        if hasattr(tarfile, "data_filter"):
            archive.extractall(destination, filter="data")
        else:
            # Older Python versions lack the safe extraction filter. Git archives
            # here need only regular files/directories; reject links and escapes.
            for member in archive.getmembers():
                target = (destination / member.name).resolve()
                if not (member.isfile() or member.isdir()) or destination.resolve() not in target.parents:
                    raise RuntimeError(f"Unsafe archive member: {member.name}")
            archive.extractall(destination)
    archive_path.unlink()


def export_revision(repo, revision, destination, temp_root, harness):
    export_archive(repo, revision, destination, temp_root / f"{destination.name}.tar")
    for relative in REQUIRED_SUBMODULES:
        entry = git_output(repo, "ls-tree", revision, "--", relative).split()
        if len(entry) != 4 or entry[0] != "160000" or entry[1] != "commit":
            raise RuntimeError(f"Missing submodule {relative} in {revision}")
        submodule = repo / relative
        if not (submodule / ".git").exists():
            raise RuntimeError(f"Initialize {relative} before comparing commits")
        export_archive(submodule, entry[2], destination / relative,
                       temp_root / f"{destination.name}-{relative.replace('/', '-')}.tar")
    # Use the same runner, build flags and case names on both revisions. Only
    # the FFT implementation and its pinned dependencies come from each commit.
    shutil.copy2(harness / "CMakeLists.txt", destination / "CMakeLists.txt")
    for source in (harness / "benchmark").iterdir():
        if source.suffix in (".cpp", ".hpp"):
            shutil.copy2(source, destination / "benchmark" / source.name)


def summarize_pairs(current_runs, previous_runs):
    def index(rows):
        if any(row["status"] != "measured" for row in rows):
            raise ValueError("Commit comparisons require measured results for every case")
        return {(row["order"], row["direction"], row["api"]): row for row in rows}
    current = [index(rows) for rows in current_runs]
    previous = [index(rows) for rows in previous_runs]
    if not current or len(current) != len(previous):
        raise ValueError("A comparison needs paired runs")
    keys = current[0].keys()
    if any(run.keys() != keys for run in current + previous):
        raise ValueError("Compared revisions have different case coverage")
    summary = []
    for order, direction, api in sorted(keys):
        key = order, direction, api
        ratios = [old[key]["cpu_time_us"] / new[key]["cpu_time_us"] for new, old in zip(current, previous)]
        if any(not math.isfinite(ratio) or ratio <= 0 for ratio in ratios):
            raise ValueError("Comparison contains invalid ratios")
        ratio = statistics.geometric_mean(ratios)
        logs = [math.log(value) for value in ratios]
        center = statistics.median(logs)
        mad = statistics.median(abs(value - center) for value in logs)
        summary.append(dict(order=order, direction=direction, api=api, throughput_ratio=ratio,
                            change_percent=100 * (ratio - 1),
                            median_ratio=statistics.median(ratios), mad_percent=100 * math.expm1(mad),
                            current_us=statistics.median(run[key]["cpu_time_us"] for run in current),
                            previous_us=statistics.median(run[key]["cpu_time_us"] for run in previous)))
    return summary


def compare(args):
    kind = args.kind
    initial = repository_snapshot(ROOT)
    revisions = {"current": git_output(ROOT, "rev-parse", "--verify", f"{args.current}^{{commit}}"),
                 "previous": git_output(ROOT, "rev-parse", "--verify", f"{args.previous}^{{commit}}")}
    print(f"Current:  {revisions['current']}\nPrevious: {revisions['previous']}")
    print("Both commits use this checkout's benchmark harness; uncommitted FFT changes are excluded.", flush=True)
    print(f"Timing: {timing_description(args)}", flush=True)
    runs = {"current": [], "previous": []}
    if args.artifacts_dir is not None:
        args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="zldsp-fft-compare-", dir=args.artifacts_dir) as temporary:
        temp_root = Path(temporary)
        # Freeze the harness once so edits during a long build cannot make
        # the two exported revisions use different measurement code.
        harness = temp_root / "harness"
        (harness / "benchmark").mkdir(parents=True)
        shutil.copy2(ROOT / "CMakeLists.txt", harness / "CMakeLists.txt")
        for source in (ROOT / "benchmark").iterdir():
            if source.suffix in (".cpp", ".hpp"):
                shutil.copy2(source, harness / "benchmark" / source.name)
        executables = {}
        target = f"throughput_zldsp_{kind}"
        for label, revision in revisions.items():
            source = temp_root / f"source-{label}"
            export_revision(ROOT, revision, source, temp_root, harness)
            print(f"Building {label}...", flush=True)
            executables[label] = build_benchmark("zldsp", target, args.avx2, args.double,
                                                 source_dir=source, build_dir=temp_root / f"build-{label}",
                                                 use_sse2=args.sse2)
        for repetition in range(args.repeats):
            labels = ("current", "previous") if repetition % 2 == 0 else ("previous", "current")
            print(f"Pair {repetition + 1}/{args.repeats}: {' then '.join(labels)}", flush=True)
            for label in labels:
                runs[label].append(run_throughput(executables[label], kind, args.n0, args.n1,
                                                 args.apis, repeats=1,
                                                 min_time0=args.min_time0, min_time1=args.min_time1))
    summary = summarize_pairs(runs["current"], runs["previous"])
    print(f"{'Order':>5} {'Direction':<9} {'API':<12} {'Current us':>12} {'Previous us':>12} {'Ratio':>9} {'Change':>9} {'MAD':>9}")
    for row in summary:
        print(f"{row['order']:>5} {row['direction']:<9} {row['api']:<12} {row['current_us']:12.6f} "
              f"{row['previous_us']:12.6f} {row['throughput_ratio']:9.4f} {row['change_percent']:+8.2f}% {row['mad_percent']:8.2f}%")
    if repository_snapshot(ROOT) != initial:
        raise RuntimeError("Repository state changed during the comparison")
    save_results(args, kind, "commit_comparison", summary, revisions=revisions, runs=runs,
                 harness="current checkout", all_apis=args.apis)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    for operation, handler, description in (
        ("accuracy", accuracy, "Check library median agreement or zldsp API agreement"),
        ("throughput", throughput, "Measure libraries or zldsp APIs in both directions"),
        ("compare", compare, "Compare throughput before/after a commit"),
    ):
        command = commands.add_parser(operation, help=description, description=description)
        command.set_defaults(handler=handler)
        command.add_argument("kind", choices=("cfft", "rfft"))
        command.add_argument("n0", type=int, help="First order (size = 2^order)")
        command.add_argument("n1", type=int, help="Last order, inclusive")
        isa = command.add_mutually_exclusive_group()
        isa.add_argument("--avx2", action="store_true")
        isa.add_argument("--sse2", action="store_true")
        command.add_argument("--double", action="store_true", help="Use double precision")
        command.add_argument("--artifacts-dir", type=Path, metavar="DIR",
                             help="Root for build artifacts and temporary comparison files "
                                  "(default: <repo>/build_fft for accuracy/throughput, system temp for compare)")
        command.add_argument("--json", type=Path, help="Save structured results to this file")
        scope = command.add_mutually_exclusive_group()
        scope.add_argument("--apis", action="store_true", help="Compare every zldsp API")
        if operation == "throughput":
            scope.add_argument("--libraries", nargs="+", choices=ALGOS,
                               help="Libraries to measure (default: zldsp)")
            scope.add_argument("--standard-libraries", action="store_true",
                               help="Measure available libraries except FFTW measure and estimate")
            scope.add_argument("--all-libraries", action="store_true",
                               help="Measure available libraries, including FFTW measure and estimate")
            build_mode = command.add_mutually_exclusive_group()
            build_mode.add_argument("--build-only", action="store_true",
                                    help="Build all selected libraries without measuring")
            build_mode.add_argument("--skip-build", action="store_true",
                                    help="Measure existing binaries without configuring or building")
        command.add_argument("--repeats", type=int, default=6 if operation == "compare" else 3,
                             help={"accuracy": "Random inputs in addition to five fixed signals (default: 3)",
                                   "throughput": "Timing repetitions per case (default: 3)",
                                   "compare": "Paired runs, alternating which revision runs first (default: 6)"}[operation])
        if operation == "accuracy":
            command.add_argument("--threshold", type=float,
                                 help="Maximum relative RMSE (default: float 2e-5, double 2e-12)")
        else:
            command.add_argument("--min-time0", type=float,
                                 help="Minimum seconds per case/repetition at n0 (default: 0.5); "
                                      "scales exponentially to --min-time1 at n1")
            command.add_argument("--min-time1", type=float,
                                 help="Minimum seconds per case/repetition at n1 (default: 5); "
                                      "a single-order run uses --min-time0")
            command.add_argument("--min-time", type=float,
                                 help="Set both timing endpoints to the same value; "
                                      "cannot be combined with --min-time0 or --min-time1")
        if operation == "compare":
            command.add_argument("--current", default="HEAD", help="New revision (default: HEAD)")
            command.add_argument("--previous", default="HEAD^", help="Baseline revision (default: HEAD^)")
    args = parser.parse_args(argv)
    if args.operation != "accuracy":
        if args.min_time is not None:
            if args.min_time0 is not None or args.min_time1 is not None:
                parser.error("--min-time cannot be combined with --min-time0 or --min-time1")
            args.min_time0 = args.min_time1 = args.min_time
        if args.min_time0 is None:
            args.min_time0 = DEFAULT_MIN_TIME0
        if args.min_time1 is None:
            args.min_time1 = DEFAULT_MIN_TIME1
    if args.operation == "accuracy":
        if args.threshold is None:
            args.threshold = 2e-12 if args.double else 2e-5
        if args.repeats > 1000:
            parser.error("accuracy --repeats must not exceed 1000")
    validate_args(parser, args)
    if args.artifacts_dir is not None:
        args.artifacts_dir = args.artifacts_dir.expanduser().resolve()
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        args.handler(args)
        return 0
    except (RuntimeError, ValueError, OSError, tarfile.TarError) as error:
        print(str(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Benchmark interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
