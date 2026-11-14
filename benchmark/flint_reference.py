"""Validate streamed FFT samples against independent FLINT/Arb transforms.

The exporter writes one JSON header followed by interleaved binary64 vectors
in native byte order (specified in the header). For each pattern, forward then
backward: input, then each library's output and (forward only) round trip.
Outputs already use the conventional unscaled DFT/inverse DFT convention;
round trips are normalized. Float samples are promoted exactly to binary64.
"""

import argparse
from array import array
from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

try:
    import flint
    from flint import acb, arb, ctx
except ImportError as error:
    raise RuntimeError("Library accuracy requires python-flint. Install with Python 3.10+: "
                       "python -m pip install -r benchmark/requirements-accuracy.txt") from error


LIBRARIES = ("fftw3_estimate", "kfr", "zldsp")
PRECISIONS = (192, 384, 768)
FORMAT = "zldsp-fft-accuracy-v1"


def uncertainty_limit(threshold):
    return min(2.0 ** -80, threshold / 1024)


def upper_float(value):
    """Convert a nonnegative Arb bound to binary64, rounding outward."""
    upper = value.upper()
    result = float(upper)
    if not math.isfinite(result) or result < 0:
        raise ValueError("Non-finite or negative accuracy bound")
    if arb(result) < upper:
        result = math.nextafter(result, math.inf)
    if not math.isfinite(result):
        raise ValueError("Accuracy bound exceeds binary64 range")
    return result


def read_vector(stream, count, byte_order):
    data = stream.read(16 * count)
    if len(data) != 16 * count:
        raise ValueError("Truncated accuracy sample stream")
    values = array("d")
    values.frombytes(data)
    if byte_order != sys.byteorder:
        values.byteswap()
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Non-finite accuracy sample")
    return values


def complex_values(values):
    return [acb(values[i], values[i + 1]) for i in range(0, len(values), 2)]


@dataclass
class Reference:
    values: list
    denominator: arb
    precision: int
    uncertainty: float

    @classmethod
    def from_values(cls, values, precision):
        energy, radius = arb(0), arb(0)
        for value in values:
            re, im = value.real, value.imag
            energy += re * re + im * im
            radius = max(radius, value.rad().upper())
        # Exactly zero references use absolute RMS, as the previous test did.
        denominator = arb(len(values)) if energy.is_zero() else energy.lower()
        if not denominator.is_finite() or not denominator > 0:
            return None
        uncertainty = (len(values) * radius * radius / denominator).sqrt()
        return cls(values, denominator, precision, upper_float(uncertainty))

    def errors(self, actual):
        if len(actual) != 2 * len(self.values):
            raise ValueError("Accuracy output size mismatch")
        with ctx.workprec(self.precision):
            energy, maximum = arb(0), arb(0)
            for i, expected in enumerate(self.values):
                difference = acb(actual[2 * i], actual[2 * i + 1]) - expected
                bound = difference.abs_upper()
                energy += bound * bound
                maximum = max(maximum, bound)
            # The numerator is an upper bound and the denominator a lower
            # bound. Keep subtraction and accumulation in reference precision;
            # only the final reported metrics are converted to binary64.
            rmse = (energy.upper() / self.denominator).sqrt()
            return upper_float(rmse), upper_float(maximum)


def make_reference(values, kind, backward, n, threshold):
    count = n // 2 + 1 if kind == "rfft" and backward else n
    if len(values) != 2 * count:
        raise ValueError("Accuracy input size mismatch")
    if kind == "rfft":
        if backward and (values[1] != 0 or values[-1] != 0):
            raise ValueError("RFFT DC and Nyquist inputs must be real")
        if not backward and any(values[i] != 0 for i in range(1, len(values), 2)):
            raise ValueError("RFFT time input must be real")
    # Nine decimal orders below typical double roundoff, and also well below
    # the user's tolerance. Refinement has a finite limit and never silently
    # falls back to a library consensus or a machine-precision reference.
    limit = uncertainty_limit(threshold)
    for precision in PRECISIONS:
        with ctx.workprec(precision):
            source = complex_values(values)
            if kind == "rfft" and backward:
                source += [value.conjugate() for value in source[-2:0:-1]]
            expected = acb.dft(source, inverse=backward)
            if backward:
                # FLINT's inverse includes 1/N; the exported inverse is unscaled.
                expected = [value * n for value in expected]
            if kind == "rfft":
                expected = ([acb(value.real) for value in expected] if backward
                            else expected[:n // 2 + 1])
            reference = Reference.from_values(expected, precision)
            if reference is not None and reference.uncertainty <= limit:
                return reference
    raise RuntimeError(f"FLINT reference uncertainty exceeds {limit:g} at {PRECISIONS[-1]} bits")


def read_results(stream, kind, order, repetitions, seed, threshold, use_double):
    header = json.loads(stream.readline(8192))
    if not isinstance(header, dict):
        raise ValueError("Unexpected accuracy stream header")
    expected_header = dict(format=FORMAT, kind=kind, order=order, repetitions=repetitions,
                           seed=seed, precision="double" if use_double else "float",
                           libraries=list(LIBRARIES), byte_order=header.get("byte_order"))
    if header != expected_header or header["byte_order"] not in ("little", "big"):
        raise ValueError(f"Unexpected accuracy stream header: {header}")
    n, rows = 1 << order, {}

    def record(library, direction, reference, actual):
        error, maximum = reference.errors(actual)
        row = rows.setdefault((library, direction), dict(
            order=order, library=library, direction=direction,
            api="aos" if direction == "roundtrip" else "flint",
            relative_rmse=0.0, max_abs=0.0, reference_precision_bits=0,
            reference_relative_uncertainty=0.0, status="passed"))
        row["relative_rmse"] = max(row["relative_rmse"], error)
        row["max_abs"] = max(row["max_abs"], maximum)
        row["reference_precision_bits"] = max(row["reference_precision_bits"], reference.precision)
        row["reference_relative_uncertainty"] = max(row["reference_relative_uncertainty"],
                                                     reference.uncertainty)
        if error > threshold:
            row["status"] = "failed"

    for _ in range(5 + repetitions):
        for backward in (False, True):
            input_count = n // 2 + 1 if kind == "rfft" and backward else n
            output_count = n // 2 + 1 if kind == "rfft" and not backward else n
            values = read_vector(stream, input_count, header["byte_order"])
            reference = make_reference(values, kind, backward, n, threshold)
            roundtrip = None
            if not backward:
                with ctx.workprec(PRECISIONS[0]):
                    roundtrip = Reference.from_values(complex_values(values), PRECISIONS[0])
            for library in LIBRARIES:
                actual = read_vector(stream, output_count, header["byte_order"])
                record(library, "backward" if backward else "forward", reference, actual)
                if not backward:
                    recovered = read_vector(stream, n, header["byte_order"])
                    record(library, "roundtrip", roundtrip, recovered)
            del reference, roundtrip
    if stream.read(1):
        raise ValueError("Unexpected trailing accuracy samples")
    return [rows[library, direction] for library in LIBRARIES
            for direction in ("forward", "backward", "roundtrip")]


def check_libraries(executable, kind, order, repetitions, seed, threshold, use_double):
    command = [str(executable), "--export", str(order), str(repetitions), str(seed)]
    # File-backed stderr cannot fill a pipe while the reference calculation
    # applies backpressure to the binary stdout stream.
    with tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr)
        try:
            rows = read_results(process.stdout, kind, order, repetitions, seed, threshold, use_double)
            if process.wait() != 0:
                raise RuntimeError("Accuracy sample exporter failed")
            return rows
        except (ValueError, RuntimeError, OSError) as error:
            if process.poll() is None:
                process.kill()
            process.wait()
            stderr.seek(0)
            detail = stderr.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"{error}" + (f"\n{detail}" if detail else "")) from error
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()


def metadata(threshold):
    return dict(reference="FLINT/Arb", reference_precision_bits=list(PRECISIONS),
                reference_relative_uncertainty_limit=uncertainty_limit(threshold),
                metric_semantics="conservative upper bounds against the unrounded reference",
                python_flint_version=flint.__version__, flint_version=flint.__FLINT_VERSION__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("kind", choices=("cfft", "rfft"))
    parser.add_argument("order", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--double", action="store_true")
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    threshold = args.threshold if args.threshold is not None else (2e-12 if args.double else 2e-5)
    if not 1 <= args.order <= 30 or not 1 <= args.repeats <= 1000:
        parser.error("order must be in [1, 30] and repetitions in [1, 1000]")
    if not math.isfinite(threshold) or threshold <= 0:
        parser.error("threshold must be finite and positive")
    try:
        rows = check_libraries(args.executable.resolve(), args.kind, args.order,
                               args.repeats, 42, threshold, args.double)
        for row in rows:
            print(json.dumps(row, allow_nan=False))
        return int(any(row["status"] == "failed" for row in rows))
    except (RuntimeError, ValueError, OSError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
