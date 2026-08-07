from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "runtime" / "benchmark" / "kimi_k3_inference_calculator.py"


def load_calculator():
    spec = importlib.util.spec_from_file_location("_standalone_k3_calculator", LAUNCHER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {LAUNCHER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CALCULATOR = load_calculator()
ANALYZER = CALCULATOR._load_analyzer()


class StandaloneRuntimeTest(unittest.TestCase):
    def test_manifest_defaults_are_a_valid_request(self):
        manifest = CALCULATOR.manifest_payload()
        result = CALCULATOR.calculate_payload(manifest["defaults"])
        self.assertEqual(len(result["results"]), 3)
        self.assertEqual(result["results"][0]["workload"]["phase"], "prefill")

    def test_h200_tp16_marlin_memory_is_padded(self):
        result = CALCULATOR.calculate_payload(
            {
                "phase": "decode",
                "hardware": ["h200"],
                "tp_size": 16,
                "batch_size": 8,
                "context_length": 4096,
            }
        )["results"][0]
        self.assertAlmostEqual(
            result["memory"]["static_weight_bytes_per_rank"] / 1024**3,
            129.5701426193118,
        )

    def test_supported_matrix_has_finite_complete_traces(self):
        configurations = 0
        for phase in ("prefill", "decode"):
            workload = ANALYZER.Workload(
                phase=phase,
                batch_size=1 if phase == "prefill" else 8,
                sequence_length=4096 if phase == "prefill" else None,
                context_length=4096 if phase == "decode" else None,
            )
            for family in ANALYZER.CALCULATOR_HARDWARE_FAMILIES:
                for tp_size in (8, 16, 32):
                    result = ANALYZER.estimate(
                        hardware=ANALYZER.make_tp_hardware(family, tp_size),
                        workload=workload,
                    )
                    self.assertEqual(len(result.layers), 96)
                    self.assertTrue(
                        math.isclose(
                            sum(layer.latency_seconds for layer in result.layers),
                            result.total_seconds,
                            rel_tol=1e-12,
                            abs_tol=1e-15,
                        )
                    )
                    for layer in result.layers:
                        for operation in layer.operations:
                            for calculation in operation.calculations.values():
                                self.assertTrue(calculation.formula)
                                self.assertTrue(calculation.substitution)
                                self.assertTrue(calculation.units)
                                self.assertTrue(math.isfinite(float(calculation.result)))
                    configurations += 1
        self.assertEqual(configurations, 18)


if __name__ == "__main__":
    unittest.main()
