from __future__ import annotations

import importlib
import importlib.util
import math
import sys
import unittest
from dataclasses import replace
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
ESTIMATOR = importlib.import_module(f"{ANALYZER.__name__}.estimator")


def _routed_expert_operation(result):
    for layer in result.layers:
        for operation in layer.operations:
            if operation.id == "moe_routed_experts":
                return operation
    raise AssertionError("Missing routed-expert operation.")


def _blackwell_decode_padding_oracle(
    *, rank_count: int, batch_size: int, cuda_graph: bool
) -> dict[str, object]:
    """Independent integer oracle for the pinned K3 Blackwell recipe."""

    attention_tp = 8
    routed_top_k = 16
    attention_dp = rank_count // attention_tp
    base, remainder = divmod(batch_size, attention_dp)
    real_rows = tuple(
        base + (dp_rank < remainder) for dp_rank in range(attention_dp)
    )
    aligned_rows = tuple(
        ((rows + attention_tp - 1) // attention_tp) * attention_tp
        if rows
        else 0
        for rows in real_rows
    )
    maximum_rows = max(aligned_rows)
    if cuda_graph:
        capture_buckets = (
            [1, 2, 4, 8, 12]
            + list(range(16, 257, 8))
            + list(range(272, 512, 16))
            + [512]
        )
        captured_rows = next(
            bucket for bucket in capture_buckets if bucket >= maximum_rows
        )
        model_rows = (captured_rows,) * attention_dp
        padding_mode = "max_len_cuda_graph"
    elif 2 * sum(aligned_rows) >= maximum_rows * attention_dp:
        model_rows = (maximum_rows,) * attention_dp
        padding_mode = "max_len"
    else:
        model_rows = aligned_rows
        padding_mode = "sum_len"

    source_rows = tuple(rows // attention_tp for rows in model_rows)
    global_source_rows = attention_tp * sum(source_rows)
    routed_pairs = global_source_rows * routed_top_k
    if routed_pairs % rank_count:
        raise AssertionError("Oracle routed pairs must divide the EP group.")
    return {
        "real_rows": real_rows,
        "aligned_rows": aligned_rows,
        "model_rows": model_rows,
        "padding_mode": padding_mode,
        "source_rows": source_rows,
        "global_source_rows": global_source_rows,
        "routed_pairs": routed_pairs,
        "balanced_received_pairs": routed_pairs // rank_count,
    }


class StandaloneRuntimeTest(unittest.TestCase):
    def test_manifest_defaults_are_a_valid_request(self):
        manifest = CALCULATOR.manifest_payload()
        result = CALCULATOR.calculate_payload(manifest["defaults"])
        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(result["schema_version"], 2)
        self.assertEqual(len(result["results"]), 3)
        self.assertEqual(result["results"][0]["workload"]["phase"], "prefill")

    def test_manifest_exposes_six_explicit_sharding_scenarios(self):
        manifest = CALCULATOR.manifest_payload()
        options = manifest["sharding_options"]

        self.assertEqual(
            [
                (
                    option["id"],
                    option["tp_size"],
                    option["moe_sharding"],
                    option["ep_size"],
                )
                for option in options
            ],
            [
                ("tp8", 8, "tp", 1),
                ("tp16", 16, "tp", 1),
                ("tp32", 32, "tp", 1),
                ("tp8+ep8", 8, "ep", 8),
                ("tp16+ep16", 16, "ep", 16),
                ("tp32+ep32", 32, "ep", 32),
            ],
        )
        self.assertEqual(manifest["defaults"]["moe_sharding"], "tp")
        for option in options:
            self.assertEqual(
                {support["status"] for support in option["families"].values()},
                {"modeled"},
            )

    def test_explicit_h200_tp16_does_not_select_ep_recipe(self):
        result = CALCULATOR.calculate_payload(
            {
                "phase": "decode",
                "hardware": ["h200"],
                "tp_size": 16,
                "moe_sharding": "tp",
                "batch_size": 1,
                "context_length": 4096,
            }
        )["results"][0]

        self.assertEqual(result["hardware"]["id"], "h200-tp16")
        self.assertEqual(result["hardware"]["ep_size"], 1)
        self.assertEqual(result["hardware"]["moe_sharding"], "tp")

    def test_h200_tp8_ep8_is_modeled_but_memory_infeasible(self):
        result = CALCULATOR.calculate_payload(
            {
                "phase": "decode",
                "hardware": ["h200"],
                "tp_size": 8,
                "moe_sharding": "ep",
                "batch_size": 1,
                "context_length": 4096,
            }
        )["results"][0]

        self.assertEqual(result["hardware"]["id"], "h200-tpep8")
        self.assertEqual(result["hardware"]["local_routed_experts"], 112)
        self.assertIsNone(
            result["hardware"]["scaleout_bytes_per_s_per_gpu_per_direction"]
        )
        self.assertNotIn("NDR400", " ".join(result["hardware"]["derivations"]))
        self.assertIn(
            "one eight-GPU NVLink domain",
            " ".join(result["hardware"]["derivations"]),
        )
        self.assertEqual(
            result["hardware"]["routed_expert_intermediate_size_per_partition"],
            3072,
        )
        self.assertFalse(result["memory"]["fits_nominal_capacity"])
        self.assertGreater(result["total_seconds"], 0)
        for preset_id in ("h200-tpep16", "h200-tpep32"):
            preset = ANALYZER.HARDWARE_PRESETS[preset_id]
            self.assertEqual(
                preset.scaleout_bytes_per_s_per_gpu_per_direction, 50e9
            )
            self.assertIn("NDR400", " ".join(preset.derivations))

    def test_blackwell_ep_recipes_match_public_contract(self):
        for family in ("b300", "gb300"):
            for tp_size in (8, 16, 32):
                with self.subTest(family=family, tp_size=tp_size):
                    hardware = ANALYZER.make_ep_hardware(family, tp_size)
                    self.assertEqual(hardware.tp_size, tp_size)
                    self.assertEqual(hardware.ep_size, tp_size)
                    self.assertEqual(hardware.attention_tp_size, 8)
                    self.assertEqual(hardware.attention_dp_size, tp_size // 8)
                    self.assertEqual(hardware.local_attention_heads, 12)
                    self.assertEqual(hardware.moe_a2a_backend, "MegaMoE")
                    self.assertEqual(hardware.moe_backend, "DeepGEMM MegaMoE MXFP4 W4A8")
                    self.assertTrue(hardware.sp_moe)
                    self.assertEqual(hardware.dp_lm_head, tp_size > 8)
                    self.assertEqual(hardware.kv_cache_bytes_per_element, 1.0)
                    self.assertEqual(hardware.kda_state_bytes_per_element, 2.0)
                    self.assertFalse(hardware.k3_fused_all_reduce_capable)
                    self.assertEqual(
                        hardware.execution_contract_id,
                        "kimi_k3_blackwell_megamoe_sp_v1",
                    )
                    self.assertEqual(
                        hardware.node_count,
                        tp_size // (8 if family == "b300" else 4),
                    )
                    if tp_size == 8:
                        self.assertIn(
                            "public SGLang live-generator TP8+EP8 scenario",
                            hardware.recipe_status,
                        )
                        self.assertIn(
                            "omitted from the rendered 16-64 GPU table",
                            hardware.recipe_status,
                        )
                    else:
                        self.assertIn(
                            "Peak Throughput recipe", hardware.recipe_status
                        )

    def test_blackwell_execution_contract_is_enforced_by_hardware_validation(self):
        hardware = ANALYZER.make_ep_hardware("b300", 32)
        invalid_overrides = (
            {"attention_tp_size": 16, "attention_dp_size": 2},
            {"moe_backend": "FlashInfer MXFP4 W4A8"},
            {"moe_a2a_backend": "Other A2A"},
            {"sp_moe": False},
            {"dp_lm_head": False},
            {"kv_cache_bytes_per_element": 2.0},
            {"kda_state_bytes_per_element": 1.0},
            {"gpus_per_node": 4, "node_count": 8},
        )
        for override in invalid_overrides:
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    replace(hardware, **override).validate()

    def test_mixed_hardware_ep_request_returns_complete_comparison(self):
        response = CALCULATOR.calculate_payload(
            {
                "phase": "decode",
                "hardware": ["h200", "b300", "gb300"],
                "tp_size": 16,
                "moe_sharding": "ep",
                "batch_size": 16,
                "context_length": 128,
                "decode_cuda_graph": False,
            },
            ANALYZER,
        )
        self.assertEqual(
            [result["hardware"]["family"] for result in response["results"]],
            ["h200", "b300", "gb300"],
        )

    def test_blackwell_ep_eager_padding_matches_mlp_sync_boundary_oracle(self):
        # Boundary theorem: mandatory per-DP TP8 alignment happens before the
        # eager MAX_LEN/SUM_LEN decision. These cases therefore all take the
        # recipe's RS + AG path; the defensive AR fallback is unreachable.
        cases = (
            (16, 14, (7, 7), (8, 8), (8, 8), 16, 256, 16),
            (16, 15, (8, 7), (8, 8), (8, 8), 16, 256, 16),
            (16, 16, (8, 8), (8, 8), (8, 8), 16, 256, 16),
            (16, 17, (9, 8), (16, 8), (16, 16), 32, 512, 32),
            (
                32,
                31,
                (8, 8, 8, 7),
                (8, 8, 8, 8),
                (8, 8, 8, 8),
                32,
                512,
                16,
            ),
            (
                32,
                32,
                (8, 8, 8, 8),
                (8, 8, 8, 8),
                (8, 8, 8, 8),
                32,
                512,
                16,
            ),
            (
                32,
                33,
                (9, 8, 8, 8),
                (16, 8, 8, 8),
                (16, 16, 16, 16),
                64,
                1024,
                32,
            ),
        )
        for (
            tp_size,
            batch,
            real_rows,
            aligned_rows,
            model_rows,
            global_source_rows,
            routed_pairs,
            balanced_received_pairs,
        ) in cases:
            with self.subTest(tp_size=tp_size, batch=batch):
                oracle = _blackwell_decode_padding_oracle(
                    rank_count=tp_size,
                    batch_size=batch,
                    cuda_graph=False,
                )
                self.assertEqual(oracle["real_rows"], real_rows)
                self.assertEqual(oracle["aligned_rows"], aligned_rows)
                self.assertEqual(oracle["model_rows"], model_rows)
                self.assertEqual(oracle["padding_mode"], "max_len")
                self.assertEqual(
                    oracle["global_source_rows"], global_source_rows
                )
                self.assertEqual(oracle["routed_pairs"], routed_pairs)
                self.assertEqual(
                    oracle["balanced_received_pairs"],
                    balanced_received_pairs,
                )

                ledger = ESTIMATOR._parallel_work_ledger(
                    workload=ANALYZER.Workload(
                        phase="decode", batch_size=batch, context_length=128
                    ),
                    hardware=ANALYZER.make_ep_hardware("b300", tp_size),
                    decode_cuda_graph_replay=False,
                )
                self.assertIsNotNone(ledger)
                self.assertEqual(ledger.dp_real_requests, oracle["real_rows"])
                self.assertEqual(
                    ledger.dp_mlp_aligned_rows, oracle["aligned_rows"]
                )
                self.assertEqual(ledger.dp_model_rows, oracle["model_rows"])
                self.assertEqual(
                    ledger.dp_padding_mode, oracle["padding_mode"]
                )
                self.assertEqual(
                    ledger.source_rows_per_attention_rank,
                    oracle["source_rows"],
                )
                self.assertEqual(
                    ledger.global_model_rows, oracle["global_source_rows"]
                )
                self.assertEqual(
                    ledger.routed_pair_instances, oracle["routed_pairs"]
                )
                self.assertEqual(
                    ledger.balanced_received_pairs_per_ep_rank,
                    oracle["balanced_received_pairs"],
                )

    def test_blackwell_ep_cuda_graph_uses_one_common_max_len_shape(self):
        cases = (
            (16, 14, (7, 7), (8, 8), (8, 8), 16, 256, 16),
            (16, 17, (9, 8), (16, 8), (16, 16), 32, 512, 32),
            (
                32,
                31,
                (8, 8, 8, 7),
                (8, 8, 8, 8),
                (8, 8, 8, 8),
                32,
                512,
                16,
            ),
            (
                32,
                1,
                (1, 0, 0, 0),
                (8, 0, 0, 0),
                (8, 8, 8, 8),
                32,
                512,
                16,
            ),
            (
                16,
                513,
                (257, 256),
                (264, 256),
                (272, 272),
                544,
                8704,
                544,
            ),
        )
        for (
            tp_size,
            batch,
            real_rows,
            aligned_rows,
            model_rows,
            global_source_rows,
            routed_pairs,
            balanced_received_pairs,
        ) in cases:
            with self.subTest(tp_size=tp_size, batch=batch):
                oracle = _blackwell_decode_padding_oracle(
                    rank_count=tp_size,
                    batch_size=batch,
                    cuda_graph=True,
                )
                self.assertEqual(oracle["real_rows"], real_rows)
                self.assertEqual(oracle["aligned_rows"], aligned_rows)
                self.assertEqual(oracle["model_rows"], model_rows)
                self.assertEqual(
                    oracle["padding_mode"], "max_len_cuda_graph"
                )
                self.assertEqual(
                    oracle["global_source_rows"], global_source_rows
                )
                self.assertEqual(oracle["routed_pairs"], routed_pairs)
                self.assertEqual(
                    oracle["balanced_received_pairs"],
                    balanced_received_pairs,
                )

                ledger = ESTIMATOR._parallel_work_ledger(
                    workload=ANALYZER.Workload(
                        phase="decode", batch_size=batch, context_length=128
                    ),
                    hardware=ANALYZER.make_ep_hardware("gb300", tp_size),
                    decode_cuda_graph_replay=True,
                )
                self.assertIsNotNone(ledger)
                self.assertEqual(ledger.dp_real_requests, oracle["real_rows"])
                self.assertEqual(
                    ledger.dp_mlp_aligned_rows, oracle["aligned_rows"]
                )
                self.assertEqual(ledger.dp_model_rows, oracle["model_rows"])
                self.assertEqual(
                    ledger.dp_padding_mode, oracle["padding_mode"]
                )
                self.assertEqual(
                    ledger.global_model_rows, oracle["global_source_rows"]
                )
                self.assertEqual(
                    ledger.routed_pair_instances, oracle["routed_pairs"]
                )
                self.assertEqual(
                    ledger.balanced_received_pairs_per_ep_rank,
                    oracle["balanced_received_pairs"],
                )

    def test_blackwell_ep_operation_inventory_matches_execution_dag(self):
        hardware = ANALYZER.make_ep_hardware("b300", 16)
        result = ANALYZER.estimate(
            hardware=hardware,
            workload=ANALYZER.Workload(
                phase="decode", batch_size=16, context_length=128
            ),
            assumptions=ANALYZER.EstimatorAssumptions(decode_cuda_graph=False),
        )
        dense_ids = {operation.id for operation in result.layers[1].operations}
        dense_operations = {
            operation.id: operation for operation in result.layers[1].operations
        }
        moe_operations = {
            operation.id: operation for operation in result.layers[2].operations
        }

        self.assertIn("dense_dp_gather_tp_reduce_scatter", dense_ids)
        self.assertIn("dense_dp_gather_global_all_gather", dense_ids)
        self.assertNotIn("dense_dp_gather_all_reduce", dense_ids)
        self.assertEqual(
            dense_operations[
                "dense_dp_gather_global_all_gather"
            ].dependencies,
            ("dense_dp_gather_tp_reduce_scatter",),
        )
        self.assertIn("dense_all_reduce", dense_ids)
        self.assertIn("dense_dp_scatter", dense_ids)
        self.assertIn("attention_reduce_scatter", moe_operations)
        self.assertIn("moe_ep_router", moe_operations)
        self.assertIn("moe_ep_latent_down", moe_operations)
        self.assertIn("moe_sp_all_gather", moe_operations)
        self.assertIn("moe_routed_experts", moe_operations)
        self.assertNotIn("moe_combined_all_reduce", moe_operations)
        self.assertNotIn("moe_shared_all_reduce", moe_operations)
        self.assertEqual(
            moe_operations["moe_topk"].dependencies, ("moe_ep_router",)
        )
        self.assertEqual(
            moe_operations["moe_shared_gate_up_tp1"].dependencies,
            ("moe_topk",),
        )
        self.assertEqual(
            set(moe_operations["moe_predispatch_quant"].dependencies),
            {"moe_topk", "moe_ep_latent_down"},
        )
        self.assertEqual(
            moe_operations["moe_routed_experts"].dependencies,
            ("moe_predispatch_quant",),
        )
        self.assertEqual(
            set(moe_operations["moe_tail_add"].dependencies),
            {"moe_latent_up_replicated", "moe_shared_down_tp1"},
        )

    def test_blackwell_dense_dp_gather_matches_pinned_collectives(self):
        hidden_size = ANALYZER.KIMI_K3_TEXT_CONFIG.hidden_size
        for tp_size, batch_size, global_rows in ((16, 16, 16), (32, 32, 32)):
            with self.subTest(tp_size=tp_size, mode="max_len"):
                result = ANALYZER.estimate(
                    hardware=ANALYZER.make_ep_hardware("b300", tp_size),
                    workload=ANALYZER.Workload(
                        phase="decode",
                        batch_size=batch_size,
                        context_length=128,
                    ),
                    assumptions=ANALYZER.EstimatorAssumptions(
                        decode_cuda_graph=False
                    ),
                )
                operations = {
                    operation.id: operation
                    for operation in result.layers[1].operations
                }
                reduce_scatter = operations[
                    "dense_dp_gather_tp_reduce_scatter"
                ]
                all_gather = operations[
                    "dense_dp_gather_global_all_gather"
                ]

                local_bytes = 8 * hidden_size * 2
                global_bytes = global_rows * hidden_size * 2
                rs_link_bytes = 7 / 8 * local_bytes
                domains = tp_size // 8
                ag_local_bytes = 7 / 8 * global_bytes
                ag_remote_bytes = (
                    (domains - 1) / domains * global_bytes / 8
                )
                self.assertEqual(
                    reduce_scatter.logical_collective_bytes, local_bytes
                )
                self.assertEqual(
                    reduce_scatter.link_bytes_per_rank, rs_link_bytes
                )
                self.assertEqual(
                    reduce_scatter.communication_seconds,
                    rs_link_bytes / 900e9,
                )
                self.assertEqual(
                    all_gather.logical_collective_bytes, global_bytes
                )
                self.assertEqual(
                    all_gather.link_bytes_per_rank,
                    ag_local_bytes + ag_remote_bytes,
                )
                self.assertEqual(
                    all_gather.communication_seconds,
                    max(ag_local_bytes / 900e9, ag_remote_bytes / 100e9),
                )

        result = ANALYZER.estimate(
            hardware=ANALYZER.make_ep_hardware("b300", 32),
            workload=ANALYZER.Workload(
                phase="decode", batch_size=1, context_length=128
            ),
            assumptions=ANALYZER.EstimatorAssumptions(decode_cuda_graph=False),
        )
        operations = {
            operation.id: operation for operation in result.layers[1].operations
        }
        self.assertIn("dense_dp_gather_all_reduce", operations)
        self.assertNotIn("dense_dp_gather_tp_reduce_scatter", operations)
        self.assertNotIn("dense_dp_gather_global_all_gather", operations)
        all_reduce = operations["dense_dp_gather_all_reduce"]
        global_bytes = 8 * hidden_size * 2
        local_bytes = global_bytes
        remote_bytes = global_bytes / 8
        self.assertEqual(all_reduce.logical_collective_bytes, global_bytes)
        self.assertEqual(
            all_reduce.link_bytes_per_rank, local_bytes + remote_bytes
        )
        self.assertEqual(
            all_reduce.communication_seconds,
            max(local_bytes / 900e9, remote_bytes / 100e9),
        )

    def test_blackwell_ep_recipe_padding_keeps_rs_and_post_moe_ag(self):
        result = ANALYZER.estimate(
            hardware=ANALYZER.make_ep_hardware("b300", 16),
            workload=ANALYZER.Workload(
                phase="decode", batch_size=14, context_length=128
            ),
            assumptions=ANALYZER.EstimatorAssumptions(decode_cuda_graph=False),
        )
        operations = {
            operation.id: operation for operation in result.layers[2].operations
        }
        operation_ids = set(operations)
        self.assertIn("attention_reduce_scatter", operation_ids)
        self.assertIn("moe_sp_all_gather", operation_ids)
        self.assertNotIn("attention_all_reduce", operation_ids)
        self.assertNotIn("attention_pending_prefix_add", operation_ids)
        reduce_scatter = operations["attention_reduce_scatter"]
        hidden_size = ANALYZER.KIMI_K3_TEXT_CONFIG.hidden_size
        logical_bytes = 8 * hidden_size * 2
        self.assertEqual(reduce_scatter.flops_per_rank, hidden_size)
        self.assertEqual(
            reduce_scatter.hbm_bytes_per_rank,
            logical_bytes * (1 + 2 / 8),
        )
        self.assertIn(
            "SP lower-bound bundle",
            " ".join(reduce_scatter.notes),
        )

    def test_all_reduce_uses_full_tensor_information_floor(self):
        hardware = ANALYZER.make_tp_hardware("h200", 8)
        assumptions = ANALYZER.EstimatorAssumptions()
        logical_bytes = 12_345

        all_reduce = ESTIMATOR._collective_cost(
            hardware=hardware,
            kind="all_reduce",
            logical_bytes=logical_bytes,
            assumptions=assumptions,
            group_size=8,
            local_domain_size=8,
        )
        self.assertEqual(all_reduce.link_bytes_per_rank, logical_bytes)
        self.assertEqual(
            all_reduce.seconds,
            logical_bytes / hardware.nvlink_bytes_per_s_per_direction,
        )

        single_rank = ESTIMATOR._collective_cost(
            hardware=hardware,
            kind="all_reduce",
            logical_bytes=logical_bytes,
            assumptions=assumptions,
            group_size=1,
            local_domain_size=1,
        )
        self.assertEqual(single_rank.link_bytes_per_rank, 0)
        self.assertEqual(single_rank.seconds, 0)

    def test_blackwell_megamoe_pair_and_fabric_oracles(self):
        config = ANALYZER.KIMI_K3_TEXT_CONFIG
        received_pairs = 16
        source_rows = 1
        routed_width = config.routed_expert_hidden_size
        expert_width = config.moe_intermediate_size
        expected_flops = received_pairs * 6 * routed_width * expert_width
        expected_hbm = (
            3
            * routed_width
            * expert_width
            * config.mxfp4_weight_bytes_per_parameter
            + received_pairs * routed_width
            + source_rows * routed_width * 2
        )
        for family in ("b300", "gb300"):
            with self.subTest(family=family):
                result = ANALYZER.estimate(
                    hardware=ANALYZER.make_ep_hardware(family, 16),
                    workload=ANALYZER.Workload(
                        phase="decode", batch_size=16, context_length=128
                    ),
                    assumptions=ANALYZER.EstimatorAssumptions(
                        decode_cuda_graph=False
                    ),
                )
                operation = _routed_expert_operation(result)
                self.assertEqual(operation.flops_per_rank, expected_flops)
                self.assertEqual(operation.hbm_bytes_per_rank, expected_hbm)
                if family == "b300":
                    local_bytes = 21 * routed_width
                    remote_bytes = 24 * routed_width
                else:
                    local_bytes = 45 * routed_width
                    remote_bytes = 0
                oracle_bytes = ESTIMATOR._blackwell_a2a_link_bytes(
                    sent_pairs_by_dp=(16, 16),
                    attention_tp_size=8,
                    ep_size=16,
                    local_domain_size=8 if family == "b300" else 16,
                    routed_width=routed_width,
                )
                self.assertEqual(oracle_bytes, (local_bytes, remote_bytes))
                self.assertEqual(
                    operation.logical_collective_bytes,
                    48 * routed_width,
                )
                self.assertEqual(
                    operation.link_bytes_per_rank,
                    local_bytes + remote_bytes,
                )
                self.assertIn(
                    "One-rank logical send/receive payload",
                    operation.calculations["logical_collective_bytes"].note,
                )
                self.assertIn(
                    "need not belong to one physical rank",
                    operation.calculations["link_bytes_per_rank"].note,
                )
                expected_seconds = max(
                    local_bytes / 900e9,
                    remote_bytes / 100e9 if remote_bytes else 0,
                )
                self.assertEqual(operation.communication_seconds, expected_seconds)

    def test_blackwell_n32_b1_directional_a2a_oracle(self):
        config = ANALYZER.KIMI_K3_TEXT_CONFIG
        hardware = ANALYZER.make_ep_hardware("gb300", 32)
        workload = ANALYZER.Workload(
            phase="decode", batch_size=1, context_length=128
        )
        ledger = ESTIMATOR._parallel_work_ledger(
            workload=workload,
            hardware=hardware,
            decode_cuda_graph_replay=False,
        )
        self.assertIsNotNone(ledger)

        # One active DP replica sends 1 row × top-16 = 16 pairs. The global
        # 8 source rows create 128 pairs, so ideal-balanced EP32 receives 4.
        # Dispatch is FP8 and combine is BF16. Fabric-local and remote
        # directional maxima must be derived independently because their
        # critical ranks need not be the same rank.
        sent_pairs = 16
        balanced_received_pairs = 4
        routed_width = config.routed_expert_hidden_size
        self.assertEqual(ledger.dp_real_requests, (1, 0, 0, 0))
        self.assertEqual(ledger.dp_mlp_aligned_rows, (8, 0, 0, 0))
        self.assertEqual(ledger.dp_model_rows, (8, 0, 0, 0))
        self.assertEqual(ledger.dp_padding_mode, "sum_len")
        self.assertEqual(ledger.source_rows_per_attention_rank, (1, 0, 0, 0))
        self.assertEqual(ledger.global_model_rows, 8)
        self.assertEqual(ledger.routed_pair_instances, 128)
        self.assertEqual(
            ledger.sent_pairs_per_attention_rank_by_dp, (16, 0, 0, 0)
        )
        self.assertEqual(
            ledger.critical_sent_pairs_per_source_rank, sent_pairs
        )
        self.assertEqual(
            ledger.balanced_received_pairs_per_ep_rank,
            balanced_received_pairs,
        )
        self.assertEqual(
            ledger.bound_condition_id,
            "balanced_dp_fractional_uniform_ep_routing",
        )
        self.assertIn("fractional ideal-routing relaxation", ledger.bound_condition)
        self.assertIn("SGLANG_K3_SP_ATTN_RES=0", ledger.bound_condition)
        self.assertNotIn("routing_skew", ledger.excluded_positive_term_ids)
        self.assertIn("collective_startup", ledger.excluded_positive_term_ids)

        for family in ("b300", "gb300"):
            with self.subTest(family=family):
                if family == "b300":
                    local_bytes = 10.5 * routed_width
                    remote_bytes = 24 * routed_width
                    local_domain_size = 8
                else:
                    local_bytes = 34.5 * routed_width
                    remote_bytes = 0
                    local_domain_size = 32
                oracle_bytes = ESTIMATOR._blackwell_a2a_link_bytes(
                    sent_pairs_by_dp=(16, 0, 0, 0),
                    attention_tp_size=8,
                    ep_size=32,
                    local_domain_size=local_domain_size,
                    routed_width=routed_width,
                )
                self.assertEqual(oracle_bytes, (local_bytes, remote_bytes))
                result = ANALYZER.estimate(
                    hardware=ANALYZER.make_ep_hardware(family, 32),
                    workload=workload,
                    assumptions=ANALYZER.EstimatorAssumptions(
                        decode_cuda_graph=False
                    ),
                )
                operation = _routed_expert_operation(result)
                self.assertEqual(
                    operation.logical_collective_bytes,
                    36 * routed_width,
                )
                self.assertEqual(
                    operation.link_bytes_per_rank,
                    local_bytes + remote_bytes,
                )
                self.assertEqual(
                    operation.communication_seconds,
                    max(
                        local_bytes / 900e9,
                        remote_bytes / 100e9 if remote_bytes else 0,
                    ),
                )

    def test_lower_bound_mode_rejects_empirical_adjustment_knobs(self):
        direct_overrides = (
            {"compute_efficiency": 0.5},
            {"hbm_efficiency": 0.5},
            {"collective_efficiency": 0.5},
            {"collective_startup_seconds": 1e-6},
            {"mla_kv_read_amplification": 2.0},
        )
        for override in direct_overrides:
            with self.subTest(api="python", override=override):
                with self.assertRaisesRegex(
                    ValueError, "Lower-bound certificates require"
                ):
                    ANALYZER.estimate(
                        hardware=ANALYZER.make_tp_hardware("h200", 8),
                        workload=ANALYZER.Workload(
                            phase="decode", batch_size=1, context_length=128
                        ),
                        assumptions=ANALYZER.EstimatorAssumptions(**override),
                    )

        request_overrides = (
            {"compute_efficiency": 0.5},
            {"hbm_efficiency": 0.5},
            {"collective_efficiency": 0.5},
            {"collective_startup_us": 1.0},
            {"mla_kv_read_amplification": 2.0},
        )
        request = {
            "phase": "decode",
            "hardware": ["h200"],
            "tp_size": 8,
            "moe_sharding": "tp",
            "batch_size": 1,
            "context_length": 128,
        }
        for override in request_overrides:
            with self.subTest(api="http", override=override):
                with self.assertRaisesRegex(
                    CALCULATOR.ApiError, "Lower-bound certificate mode requires"
                ):
                    CALCULATOR.calculate_payload({**request, **override})

    def test_blackwell_ep_weight_sharding_is_axis_specific(self):
        _, weights16 = ESTIMATOR._weight_memory(
            ANALYZER.KIMI_K3_TEXT_CONFIG,
            ANALYZER.make_ep_hardware("b300", 16),
        )
        _, weights32 = ESTIMATOR._weight_memory(
            ANALYZER.KIMI_K3_TEXT_CONFIG,
            ANALYZER.make_ep_hardware("b300", 32),
        )
        for component in (
            "embedding",
            "lm_head",
            "kda_attention",
            "mla_attention",
            "moe_router_latent_shared",
        ):
            self.assertEqual(weights16[component], weights32[component])
        self.assertEqual(weights16["dense_ffn"], 2 * weights32["dense_ffn"])
        self.assertEqual(
            weights16["routed_experts_mxfp4"],
            2 * weights32["routed_experts_mxfp4"],
        )
    def test_blackwell_ep_memory_fit_is_inconclusive_without_workspace(self):
        result = ANALYZER.estimate(
            hardware=ANALYZER.make_ep_hardware("gb300", 16),
            workload=ANALYZER.Workload(
                phase="decode", batch_size=16, context_length=128
            ),
            assumptions=ANALYZER.EstimatorAssumptions(decode_cuda_graph=False),
        )
        self.assertIsNone(result.memory.fits_nominal_capacity)
        self.assertEqual(
            result.memory.capacity_status,
            "inconclusive_megamoe_workspace_excluded",
        )

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

    def test_layer_certificates_preserve_latency_and_report_exact_ties(self):
        result = CALCULATOR.calculate_payload(
            {
                "phase": "decode",
                "hardware": ["h200"],
                "tp_size": 16,
                "moe_sharding": "ep",
                "batch_size": 1,
                "context_length": 4096,
            }
        )["results"][0]
        for layer in result["layers"]:
            candidates = (
                (
                    "critical_path",
                    layer["critical_path_lower_bound_seconds"],
                ),
                ("compute", layer["compute_resource_seconds"]),
                ("hbm", layer["hbm_resource_seconds"]),
                ("communication", layer["communication_resource_seconds"]),
            )
            expected_latency = max(value for _, value in candidates)
            expected_certificates = [
                name for name, value in candidates if value == expected_latency
            ]
            self.assertEqual(layer["latency_seconds"], expected_latency)
            self.assertEqual(
                layer["limiting_certificates"], expected_certificates
            )
            self.assertNotIn("dependency_path_seconds", layer)
            self.assertNotIn("limiting_floor", layer)

        self.assertEqual(
            result["total_seconds"],
            sum(layer["latency_seconds"] for layer in result["layers"]),
        )

    def test_layer_certificates_distinguish_exact_and_near_ties(self):
        hardware = ANALYZER.make_tp_hardware("h200", 8)
        assumptions = ANALYZER.EstimatorAssumptions()

        tied_builder = ESTIMATOR._LayerBuilder(
            name="synthetic_tie",
            number=None,
            attention=None,
            ffn=None,
            hardware=hardware,
            assumptions=assumptions,
        )
        tied_builder.add_roofline(
            op_id="hbm",
            name="HBM-only operation",
            category="synthetic",
            hbm_bytes=hardware.hbm_bandwidth_bytes_per_s,
            hbm_formula="one second of HBM demand",
            hbm_substitution=str(hardware.hbm_bandwidth_bytes_per_s),
        )
        tied = tied_builder.finish()
        self.assertEqual(tied.limiting_certificates, ("critical_path", "hbm"))

        near_builder = ESTIMATOR._LayerBuilder(
            name="synthetic_near_tie",
            number=None,
            attention=None,
            ffn=None,
            hardware=hardware,
            assumptions=assumptions,
        )
        compute = near_builder.add_roofline(
            op_id="compute",
            name="Compute-only operation",
            category="synthetic",
            flops=hardware.dense_bf16_flops_per_s,
            flops_formula="one second of compute demand",
            flops_substitution=str(hardware.dense_bf16_flops_per_s),
        )
        near_builder.add_roofline(
            op_id="hbm",
            name="Near-zero HBM operation",
            category="synthetic",
            dependencies=(compute,),
            hbm_bytes=(
                hardware.hbm_bandwidth_bytes_per_s * math.ulp(1.0)
            ),
            hbm_formula="one ULP-second of HBM demand",
            hbm_substitution=(
                f"{hardware.hbm_bandwidth_bytes_per_s} × {math.ulp(1.0)}"
            ),
        )
        near = near_builder.finish()
        self.assertGreater(near.dependency_path_seconds, near.compute_resource_seconds)
        self.assertEqual(near.limiting_certificates, ("critical_path",))

    def test_h200_ep_marlin_counts_global_pair_buffers(self):
        config = ANALYZER.KIMI_K3_TEXT_CONFIG
        hardware_specs = (
            ANALYZER.make_ep_hardware("h200", 8),
            ANALYZER.HARDWARE_PRESETS["h200-tpep16"],
            ANALYZER.HARDWARE_PRESETS["h200-tpep32"],
        )
        for hardware in hardware_specs:
            with self.subTest(hardware=hardware.id):
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
                critical_rank_unique_experts = math.ceil(
                    config.num_experts_per_token / hardware.ep_size
                )
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

    def test_tp_expert_weight_floor_allows_repeated_topk_sets(self):
        config = ANALYZER.KIMI_K3_TEXT_CONFIG
        hardware = ANALYZER.make_tp_hardware("h200", 16)
        tokens = 2
        result = ANALYZER.estimate(
            hardware=hardware,
            workload=ANALYZER.Workload(
                phase="decode", batch_size=tokens, context_length=128
            ),
            assumptions=ANALYZER.EstimatorAssumptions(decode_cuda_graph=False),
        )
        operation = _routed_expert_operation(result)
        routed_width = config.routed_expert_hidden_size
        intermediate = hardware.routed_expert_intermediate_size_per_partition
        pairs = tokens * config.num_experts_per_token
        minimum_unique = config.num_experts_per_token
        weight_bytes = minimum_unique * (
            3
            * routed_width
            * intermediate
            * config.mxfp4_weight_bytes_per_parameter
            + (2 * intermediate + routed_width) * 2
        )
        activation_bytes = (
            pairs * (4 * routed_width + 6 * intermediate) * 2
            + tokens * routed_width * 2
            + pairs * (4 + 4)
        )

        self.assertEqual(
            operation.hbm_bytes_per_rank, weight_bytes + activation_bytes
        )
        self.assertIn(
            "deterministically touches at least the same 16",
            " ".join(operation.notes),
        )

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
                self.assertEqual(
                    ESTIMATOR._marlin_m_block_size(
                        tokens=tokens,
                        top_k=config.num_experts_per_token,
                        local_experts=config.num_experts,
                    ),
                    block_size,
                )
                padded_rows = max(
                    pairs, config.num_experts_per_token * block_size
                )
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
                self.assertEqual(operation.flops_per_rank, expected)

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
        self.assertIn("Sharding scenario", html)
        self.assertEqual(html.count('name="sharding"'), 6)
        self.assertEqual(html.count('class="parallel-pair"'), 6)
        self.assertEqual(html.count('class="parallel-dimension"'), 12)
        self.assertEqual(html.count('data-dimension="tp"'), 6)
        self.assertEqual(html.count('data-dimension="ep"'), 6)
        self.assertEqual(
            html.count('aria-describedby="sharding-option-'), 6
        )
        self.assertNotIn('name="tp-size"', html)
        self.assertNotIn("TP64", html)
        self.assertNotIn("Model time", html)
        self.assertIn("Latency lower bound", html)
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
        self.assertIn("hardware.attention_tp_size", javascript)
        self.assertIn("hardware.attention_dp_size", javascript)
        self.assertIn("hardware.moe_a2a_backend", javascript)
        self.assertIn("result.parallel_work_ledger", javascript)
        self.assertIn("MegaMoE workspace excluded", javascript)
        self.assertIn("SP-MoE, MegaMoE A2A, and DeepGEMM", html)
        self.assertIn("moe_sharding: sharding.moeSharding", javascript)
        self.assertIn("updateShardingCompatibility", javascript)
        self.assertIn("validateCalculatorResponse(payload, body)", javascript)

    def test_site_exposes_non_additive_certificates_without_winner_counts(self):
        html = (REPO_ROOT / "site" / "index.html").read_text()
        javascript = (REPO_ROOT / "site" / "app.js").read_text()

        self.assertNotIn('id="floor-chart"', html)
        self.assertNotIn("Layer limiting floors", html)
        self.assertNotIn("floorCounts", javascript)
        self.assertIn("Whole-model lower-bound certificates", html)
        self.assertIn("Overlapping · non-additive", html)
        self.assertIn('id="resource-bars"', html)
        self.assertIn('id="certificate-latency-total"', html)
        self.assertIn("layer.limiting_certificates", javascript)
        self.assertIn("certificateChips(layer)", javascript)
        for field in (
            "critical_path_lower_bound_seconds",
            "compute_resource_seconds",
            "hbm_resource_seconds",
            "communication_resource_seconds",
        ):
            self.assertIn(field, javascript)

    def test_manifest_matrix_matches_public_api_and_topology(self):
        manifest = CALCULATOR.manifest_payload(ANALYZER)
        modeled = 0
        capacity_limited = set()
        for phase in ("prefill", "decode"):
            for option in manifest["sharding_options"]:
                for family, support in option["families"].items():
                    with self.subTest(
                        phase=phase, option=option["id"], family=family
                    ):
                        request = {
                            "phase": phase,
                            "hardware": [family],
                            "tp_size": option["tp_size"],
                            "moe_sharding": option["moe_sharding"],
                            "batch_size": 1 if phase == "prefill" else 8,
                        }
                        if phase == "prefill":
                            request["sequence_length"] = 4096
                        else:
                            request["context_length"] = 4096

                        if support["status"] != "modeled":
                            self.fail(f"Unknown support status: {support!r}")

                        response = CALCULATOR.calculate_payload(request, ANALYZER)
                        self.assertEqual(len(response["results"]), 1)
                        result = response["results"][0]
                        hardware = result["hardware"]
                        self.assertEqual(hardware["family"], family)
                        self.assertEqual(hardware["gpu_count"], option["tp_size"])
                        self.assertEqual(hardware["tp_size"], option["tp_size"])
                        self.assertEqual(
                            hardware["moe_sharding"], option["moe_sharding"]
                        )
                        self.assertEqual(hardware["ep_size"], option["ep_size"])
                        if option["moe_sharding"] == "ep":
                            self.assertEqual(
                                {
                                    hardware["gpu_count"],
                                    hardware["tp_size"],
                                    hardware["ep_size"],
                                },
                                {option["tp_size"]},
                            )
                            if family in ("b300", "gb300"):
                                self.assertEqual(hardware["attention_tp_size"], 8)
                                self.assertEqual(
                                    hardware["attention_dp_size"],
                                    option["tp_size"] // 8,
                                )
                                self.assertIsNotNone(
                                    result["parallel_work_ledger"]
                                )
                        else:
                            self.assertEqual(hardware["ep_size"], 1)

                        memory = result["memory"]
                        has_capacity_hint = (
                            support.get("capacity_hint")
                            == "static_weights_exceed_nominal_hbm"
                        )
                        self.assertEqual(
                            has_capacity_hint,
                            memory["static_weight_bytes_per_rank"]
                            > memory["nominal_hbm_capacity_bytes_per_rank"],
                        )
                        if has_capacity_hint:
                            capacity_limited.add((family, option["id"]))

                        self.assertEqual(len(result["layers"]), 96)
                        self.assertTrue(
                            math.isclose(
                                sum(
                                    layer["latency_seconds"]
                                    for layer in result["layers"]
                                ),
                                result["total_seconds"],
                                rel_tol=1e-12,
                                abs_tol=1e-15,
                            )
                        )
                        for layer in result["layers"]:
                            for operation in layer["operations"]:
                                for calculation in operation["calculations"].values():
                                    self.assertTrue(calculation["formula"])
                                    self.assertTrue(calculation["substitution"])
                                    self.assertTrue(calculation["units"])
                                    self.assertTrue(
                                        math.isfinite(float(calculation["result"]))
                                    )
                        modeled += 1
        self.assertEqual(modeled, 36)
        self.assertEqual(
            capacity_limited,
            {("h200", "tp8"), ("h200", "tp8+ep8")},
        )


if __name__ == "__main__":
    unittest.main()
