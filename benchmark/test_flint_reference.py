"""Checks of error calculations, precision handling and sample transport."""

from array import array
import io
import json
import math
import sys
import unittest
from unittest.mock import patch

from flint import acb, arb, ctx

import flint_reference as reference
from run import parse_accuracy


def components(values):
    return array("d", (part for value in values for part in (value.real, value.imag)))


def sample_stream(kind="cfft", byte_order=sys.byteorder, corrupt=False):
    n, order = 4, 2
    stream = io.BytesIO()
    header = dict(format=reference.FORMAT, kind=kind, order=order, repetitions=1,
                  seed=42, precision="double", byte_order=byte_order,
                  libraries=list(reference.LIBRARIES))
    stream.write(json.dumps(header).encode() + b"\n")

    def write(values):
        data = components(values)
        if byte_order != sys.byteorder:
            data.byteswap()
        stream.write(data.tobytes())

    for pattern in range(6):
        for backward in (False, True):
            input_count = n // 2 + 1 if kind == "rfft" and backward else n
            output_count = n // 2 + 1 if kind == "rfft" and not backward else n
            # Zero and impulse/DC pairs have independently known transforms.
            value = 0 if pattern == 0 else 1
            original = [complex(value)] + [0j] * (input_count - 1)
            write(original)
            for library in reference.LIBRARIES:
                output = [complex(value)] * output_count
                if corrupt and pattern == 1 and library == "zldsp" and not backward:
                    output[1] += 0.01j
                write(output)
                if not backward:
                    write(original)
    stream.seek(0)
    return stream


class ReferenceTests(unittest.TestCase):
    def test_error_subtraction_preserves_sub_double_precision(self):
        with ctx.workprec(192):
            expected = acb(arb(1) + arb(2) ** -80)
            result = reference.Reference.from_values([expected], 192)
            rmse, maximum = result.errors(array("d", [1, 0]))
        self.assertGreater(rmse, 0)
        self.assertEqual(maximum, 2 ** -80)

    def test_known_metrics_and_zero_reference(self):
        with ctx.workprec(192):
            result = reference.Reference.from_values([acb(3, 4), acb(0)], 192)
            self.assertEqual(result.errors(array("d", [6, 8, 0, 0])), (1.0, 5.0))
            zero = reference.Reference.from_values([acb(0), acb(0)], 192)
            self.assertEqual(zero.errors(array("d", [0, 0, 0, 0])), (0.0, 0.0))
            error, maximum = zero.errors(array("d", [3, 4, 0, 0]))
            self.assertAlmostEqual(error, math.sqrt(25 / 2))
            self.assertEqual(maximum, 5)

    def test_reference_uncertainty_is_included_in_error_bound(self):
        with ctx.workprec(192):
            radius = arb(2) ** -100
            result = reference.Reference.from_values([acb(arb(1, radius))], 192)
            error, maximum = result.errors(array("d", [1, 0]))
            self.assertGreaterEqual(maximum, 2 ** -100)
            self.assertGreater(error, 0)

    def test_refinement_and_exhaustion_restore_precision(self):
        values = components([complex(i / 7, i / 11) for i in range(8)])
        previous = ctx.prec
        with patch.object(reference, "PRECISIONS", (32, 192)):
            result = reference.make_reference(values, "cfft", False, 8, 2e-12)
            self.assertEqual(result.precision, 192)
        with patch.object(reference, "PRECISIONS", (32,)):
            with self.assertRaisesRegex(RuntimeError, "uncertainty"):
                reference.make_reference(values, "cfft", False, 8, 2e-12)
        self.assertEqual(ctx.prec, previous)

    def test_real_input_contract(self):
        with self.assertRaisesRegex(ValueError, "DC and Nyquist"):
            reference.make_reference(components([1j, 0j, 0j]), "rfft", True, 4, 2e-12)
        with self.assertRaisesRegex(ValueError, "must be real"):
            reference.make_reference(components([1j] * 4), "rfft", False, 4, 2e-12)


class StreamTests(unittest.TestCase):
    def test_binary64_and_byte_order(self):
        values = array("d", [math.nextafter(1.0, 2.0), -0.125, 2 ** -1074, -1e200])
        for byte_order in ("little", "big"):
            data = array("d", values)
            if byte_order != sys.byteorder:
                data.byteswap()
            actual = reference.read_vector(io.BytesIO(data.tobytes()), 2, byte_order)
            self.assertEqual(actual, values)

    def test_complete_reports_and_corruption_detection(self):
        for kind in ("cfft", "rfft"):
            for corrupt in (False, True):
                rows = reference.read_results(sample_stream(kind, corrupt=corrupt), kind, 2, 1, 42, 2e-12, True)
                output = "\n".join(json.dumps(row) for row in rows)
                if corrupt:
                    with self.assertRaisesRegex(ValueError, "tolerance exceeded"):
                        parse_accuracy(output, kind, 2, False, 2e-12)
                else:
                    self.assertEqual(len(parse_accuracy(output, kind, 2, False, 2e-12)), 9)
                    self.assertTrue(all(row["relative_rmse"] == 0 for row in rows))

    def test_invalid_streams_fail(self):
        stream = sample_stream()
        with self.assertRaisesRegex(ValueError, "header"):
            reference.read_results(stream, "rfft", 2, 1, 42, 2e-12, True)
        for suffix in (b"", b"extra"):
            data = sample_stream().getvalue()
            data = data[:-8] if not suffix else data + suffix
            with self.assertRaisesRegex(ValueError, "Truncated|trailing"):
                reference.read_results(io.BytesIO(data), "cfft", 2, 1, 42, 2e-12, True)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            reference.read_vector(io.BytesIO(array("d", [math.nan, 0]).tobytes()), 1, sys.byteorder)

    def test_exporter_failure_includes_stderr(self):
        # Python is deliberately not an exporter: it rejects --export on every
        # supported OS. This checks cleanup and propagation of producer errors.
        with self.assertRaisesRegex(RuntimeError, "export"):
            reference.check_libraries(sys.executable, "cfft", 2, 1, 42, 2e-12, True)


if __name__ == "__main__":
    unittest.main()
