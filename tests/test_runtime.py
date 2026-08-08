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


def _routed_expert_operation(result):
    for layer in result.layers:
        for operation in layer.operations:
            if operation.id == "moe_routed_experts":
                return operation
    raise AssertionError("Missing routed-expert operation.")


class StandaloneRuntimeTest(unittest.TestCase):
    def test_manifest_defaults_are_a_valid_request(self):
        manifest = CALCULATOR.manifest_payload()
        result = CALCULATOR.calculate_payload(manifest["defaults"])
        self.assertEqual(len(result["results"]), 3)
        self.assertEqual(result["results"][0]["workload"]["phase"], "prefill")

    def test_h200_tp16_selects_ep16_recipe(self):
        result = CALCULATOR.calculate_payload(
            {
                "phase": "decode",
                "hardware": ["h200"],
                "tp_size": 16,
                "batch_size": 8,
                "context_length": 4096,
            }
        )["results"][0]

        self.assertEqual(result["hardware"]["id"], "h200-tpep16")
        self.assertEqual(result["hardware"]["tp_size"], 16)
        self.assertEqual(result["hardware"]["ep_size"], 16)
        self.assertEqual(result["hardware"]["moe_sharding"], "ep")

    def test_h200_tp32_selects_ep32_recipe(self):
        result = CALCULATOR.calculate_payload(
            {
                "phase": "decode",
                "hardware": ["h200"],
                "tp_size": 32,
                "batch_size": 1,
                "context_length": 4096,
            }
        )["results"][0]

        self.assertEqual(result["hardware"]["id"], "h200-tpep32")
        self.assertEqual(result["hardware"]["gpu_count"], 32)
        self.assertEqual(result["hardware"]["node_count"], 4)
        self.assertEqual(result["hardware"]["tp_size"], 32)
        self.assertEqual(result["hardware"]["ep_size"], 32)
        self.assertEqual(result["hardware"]["moe_sharding"], "ep")
        self.assertEqual(result["hardware"]["local_routed_experts"], 28)
        self.assertEqual(
            result["hardware"]["routed_expert_intermediate_size_per_partition"],
            3072,
        )

    def test_h200_ep_marlin_counts_global_pair_buffers(self):
        config = ANALYZER.KIMI_K3_TEXT_CONFIG
        for preset_id in ("h200-tpep16", "h200-tpep32"):
            with self.subTest(preset_id=preset_id):
                hardware = ANALYZER.HARDWARE_PRESETS[preset_id]
                result = ANALYZER.estimate(
                    hardware=hardware,
                    workload=ANALYZER.Workload(
                        phase="decode", batch_size=1, context_length=4096
                    ),
                )
                operation = _routed_expert_operation(result)

                pairs = config.num_experts_per_token
                critical_rank_pairs = math.ceil(pairs / hardware.ep_size)
                hidden = config.routed_expert_hidden_size
                intermediate = config.moe_intermediate_size
                critical_rank_unique_experts = 1.0
                weight_bytes = critical_rank_unique_experts * (
                    3
                    * hidden
                    * intermediate
                    * config.mxfp4_weight_bytes_per_parameter
                    + (2 * intermediate + hidden) * 2
                )
                # Marlin allocates/activates all global pairs on every EP rank.
                # Only the two GEMMs' critical-rank rows scale down with EP.
                global_pair_bytes = pairs * (
                    max(2 * intermediate, hidden)
                    + 3 * intermediate
                    + 2 * hidden
                ) * 2
                local_gemm_bytes = critical_rank_pairs * (
                    2 * hidden + 3 * intermediate
                ) * 2
                output_bytes = hidden * 2
                metadata_bytes = pairs * (4 + 4)
                expected = (
                    weight_bytes
                    + global_pair_bytes
                    + local_gemm_bytes
                    + output_bytes
                    + metadata_bytes
                )

                self.assertEqual(operation.hbm_bytes_per_rank, expected)

    def test_h200_ep32_b1_uses_an_active_critical_rank(self):
        result = ANALYZER.estimate(
            hardware=ANALYZER.HARDWARE_PRESETS["h200-tpep32"],
            workload=ANALYZER.Workload(
                phase="decode", batch_size=1, context_length=4096
            ),
        )
        operation = _routed_expert_operation(result)
        self.assertIn("critical active EP rank", " ".join(operation.notes))

    def test_h200_marlin_flops_include_m_block_padding(self):
        config = ANALYZER.KIMI_K3_TEXT_CONFIG
        cases = (
            (ANALYZER.HARDWARE_PRESETS["h200-tpep16"], 8.0),
            (ANALYZER.HARDWARE_PRESETS["h200-tpep32"], 8.0),
            (ANALYZER.make_tp_hardware("h200", 16), 128.0),
        )
        for hardware, padded_rows in cases:
            with self.subTest(hardware=hardware.id):
                result = ANALYZER.estimate(
                    hardware=hardware,
                    workload=ANALYZER.Workload(
                        phase="decode", batch_size=1, context_length=4096
                    ),
                )
                operation = _routed_expert_operation(result)
                intermediate = (
                    config.moe_intermediate_size
                    if hardware.moe_sharding == "ep"
                    else hardware.routed_expert_intermediate_size_per_partition
                )
                pairs = config.num_experts_per_token
                expected = (
                    padded_rows
                    * 6
                    * config.routed_expert_hidden_size
                    * intermediate
                    + pairs * 8 * intermediate
                )
                self.assertEqual(operation.flops_per_rank, expected)

    def test_h200_marlin_m_block_boundaries(self):
        config = ANALYZER.KIMI_K3_TEXT_CONFIG
        hardware = ANALYZER.make_tp_hardware("h200", 16)
        cases = (
            (403, 8),
            (404, 16),
            (806, 16),
            (807, 32),
            (1612, 32),
            (1613, 48),
            (2419, 48),
            (2420, 64),
        )
        for tokens, block_size in cases:
            with self.subTest(tokens=tokens, block_size=block_size):
                result = ANALYZER.estimate(
                    hardware=hardware,
                    workload=ANALYZER.Workload(
                        phase="decode", batch_size=tokens, context_length=1
                    ),
                    assumptions=ANALYZER.EstimatorAssumptions(
                        decode_cuda_graph=False
                    ),
                )
                operation = _routed_expert_operation(result)
                pairs = tokens * config.num_experts_per_token
                active_experts = config.num_experts * (
                    1
                    - math.pow(
                        1 - config.num_experts_per_token / config.num_experts,
                        tokens,
                    )
                )
                padded_rows = max(pairs, active_experts * block_size)
                intermediate = (
                    hardware.routed_expert_intermediate_size_per_partition
                )
                expected = (
                    padded_rows
                    * 6
                    * config.routed_expert_hidden_size
                    * intermediate
                    + pairs * 8 * intermediate
                )
                self.assertAlmostEqual(
                    operation.flops_per_rank, expected, delta=1e-5
                )

    def test_blackwell_expert_flops_do_not_use_marlin_padding(self):
        config = ANALYZER.KIMI_K3_TEXT_CONFIG
        hardware = ANALYZER.make_tp_hardware("b300", 8)
        result = ANALYZER.estimate(
            hardware=hardware,
            workload=ANALYZER.Workload(
                phase="decode", batch_size=1, context_length=4096
            ),
        )
        operation = _routed_expert_operation(result)
        pairs = config.num_experts_per_token
        intermediate = hardware.routed_expert_intermediate_size_per_partition
        expected = pairs * (
            6 * config.routed_expert_hidden_size * intermediate + 8 * intermediate
        )

        self.assertEqual(operation.flops_per_rank, expected)

    def test_roofline_coordinates_match_operation_floors(self):
        for phase, workload_fields in (
            ("prefill", {"sequence_length": 4096}),
            ("decode", {"context_length": 4096}),
        ):
            with self.subTest(phase=phase):
                result = CALCULATOR.calculate_payload(
                    {
                        "phase": phase,
                        "hardware": ["h200"],
                        "tp_size": 32,
                        "batch_size": 1,
                        **workload_fields,
                    }
                )["results"][0]
                hardware = result["hardware"]
                plotted = 0
                communication_only = 0
                observed_bottlenecks = set()
                for layer in result["layers"]:
                    for operation in layer["operations"]:
                        floors = {
                            "compute": operation["compute_seconds"],
                            "hbm": operation["hbm_seconds"],
                            "communication": operation["communication_seconds"],
                        }
                        expected_bottleneck = max(
                            floors.items(), key=lambda item: item[1]
                        )[0]
                        self.assertEqual(
                            operation["bottleneck"], expected_bottleneck
                        )
                        observed_bottlenecks.add(expected_bottleneck)

                        flops = operation["flops_per_rank"]
                        hbm_bytes = operation["hbm_bytes_per_rank"]
                        duration = operation["duration_seconds"]
                        if flops > 0 and hbm_bytes > 0 and duration > 0:
                            intensity = flops / hbm_bytes
                            performance = flops / duration
                            self.assertEqual(
                                operation[
                                    "arithmetic_intensity_flops_per_hbm_byte"
                                ],
                                intensity,
                            )
                            self.assertEqual(
                                operation["roofline_flops_per_second"],
                                performance,
                            )
                            peak = (
                                hardware["k3_expert_flops_per_s"]
                                if operation["compute_kind"] == "k3_expert"
                                else hardware["dense_bf16_flops_per_s"]
                            )
                            self.assertLessEqual(performance, peak * (1 + 1e-12))
                            memory_roof = (
                                intensity
                                * hardware["hbm_bandwidth_bytes_per_s"]
                            )
                            self.assertLessEqual(
                                performance, memory_roof * (1 + 1e-12)
                            )
                            plotted += 1
                        else:
                            self.assertIsNone(
                                operation[
                                    "arithmetic_intensity_flops_per_hbm_byte"
                                ]
                            )
                            self.assertIsNone(
                                operation["roofline_flops_per_second"]
                            )
                        if operation["communication_seconds"] > 0 and flops == 0:
                            communication_only += 1

                self.assertGreater(plotted, 0)
                self.assertGreater(communication_only, 0)
                self.assertIn("hbm", observed_bottlenecks)
                self.assertIn("communication", observed_bottlenecks)
                if phase == "prefill":
                    self.assertIn("compute", observed_bottlenecks)

    def test_site_exposes_interactive_roofline_explorer(self):
        html = (REPO_ROOT / "site" / "index.html").read_text()
        javascript = (REPO_ROOT / "site" / "app.js").read_text()
        styles = (REPO_ROOT / "site" / "styles.css").read_text()

        self.assertIn('id="roofline-chart"', html)
        self.assertIn('id="roofline-reset"', html)
        self.assertIn('id="roofline-hardware"', html)
        self.assertIn("Expert parallel size", html)
        self.assertEqual(html.count('class="parallel-pair"'), 4)
        self.assertEqual(html.count('class="parallel-dimension"'), 8)
        self.assertEqual(html.count('data-dimension="tp"'), 4)
        self.assertEqual(html.count('data-dimension="ep"'), 4)
        self.assertIn(
            ".parallel-pair {\n  display: grid !important;\n  grid-template-columns: 1fr;",
            styles,
        )
        self.assertIn('id="roofline-topology"', html)
        self.assertIn('id="roofline-communication-list"', html)
        self.assertIn("Memory bandwidth-bound", html)
        self.assertNotIn("HBM-bound", html)
        self.assertNotIn("Each standard", html)
        self.assertIn('addEventListener("pointerdown"', javascript)
        self.assertIn('addEventListener("wheel"', javascript)
        self.assertIn("arithmetic_intensity_flops_per_hbm_byte", javascript)
        self.assertIn("roofline_flops_per_second", javascript)
        self.assertIn("function standardRooflinePerformance", javascript)
        self.assertIn(
            "Math.min(bandwidthBytesPerSecond * intensity, peakFlopsPerSecond)",
            javascript,
        )
        self.assertIn('hbm: "Memory bandwidth"', javascript)
        self.assertNotIn('"Standard"', javascript)
        self.assertIn('"EP off"', javascript)
        self.assertIn("local_routed_experts", javascript)

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
                        hardware=ANALYZER.make_calculator_hardware(
                            family, tp_size
                        ),
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
