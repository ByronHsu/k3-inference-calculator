"""Roofline and topology lower-bound estimator for the Kimi-K3 text model.

The estimator counts the operations selected by the scoped SGLang recipes and
then schedules their known dependencies.  It does not claim to predict real
latency: efficiencies are fixed at 100%, collective startup is excluded, and
HBM-demand certificates are conditional on counted logical reads and writes
materializing through HBM.  Those conditions keep the accounting explicit
without presenting it as a benchmark result.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Literal

from .specs import (
    HARDWARE_PRESETS,
    KIMI_K3_TEXT_CONFIG,
    DecoderLayerSpec,
    HardwareSpec,
    KimiK3TextConfig,
)

Phase = Literal["prefill", "decode"]
ComputeKind = Literal["bf16", "k3_expert", "none"]
CollectiveKind = Literal["all_reduce", "all_gather", "reduce_scatter", "all_to_all"]

BF16_BYTES = 2
FP8_BYTES = 1
FP32_BYTES = 4


@dataclass(frozen=True)
class Workload:
    phase: Phase
    batch_size: int
    sequence_length: int | None = None
    context_length: int | None = None
    execution_batch_size: int | None = None

    def validate(self) -> None:
        if self.phase not in ("prefill", "decode"):
            raise ValueError(f"Unsupported phase {self.phase!r}.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.phase == "prefill":
            if self.sequence_length is None or self.sequence_length <= 0:
                raise ValueError("Prefill requires a positive sequence_length.")
            if self.context_length is not None:
                raise ValueError("Cold prefill does not accept context_length.")
            if self.execution_batch_size is not None:
                raise ValueError("Cold prefill does not use execution_batch_size.")
        else:
            if self.context_length is None or self.context_length <= 0:
                raise ValueError(
                    "Decode requires a positive context_length; batch size alone "
                    "cannot determine MLA cache-read cost."
                )
            if self.sequence_length is not None:
                raise ValueError("Decode does not accept sequence_length.")
            if (
                self.execution_batch_size is not None
                and self.execution_batch_size < self.batch_size
            ):
                raise ValueError(
                    "execution_batch_size cannot be smaller than requested batch_size."
                )

    @property
    def model_batch_size(self) -> int:
        return self.execution_batch_size or self.batch_size

    @property
    def useful_token_count(self) -> int:
        if self.phase == "prefill":
            assert self.sequence_length is not None
            return self.batch_size * self.sequence_length
        return self.batch_size

    @property
    def token_count(self) -> int:
        if self.phase == "prefill":
            assert self.sequence_length is not None
            return self.batch_size * self.sequence_length
        return self.model_batch_size

    @property
    def attention_pair_count(self) -> int:
        if self.phase == "prefill":
            assert self.sequence_length is not None
            # Cold causal prefill: sum_{q=1..L} q, independently per sequence.
            return (
                self.batch_size * self.sequence_length * (self.sequence_length + 1) // 2
            )
        assert self.context_length is not None
        # The new query attends the existing context plus its own newly written
        # key/value. Existing-cache HBM reads are counted separately from that
        # write in the MLA core.
        # Captured decode pads unused graph rows with sequence length 1. Real
        # rows attend C existing tokens plus self; padded rows attend one dummy
        # slot, so only the model's dense token axis is fully padded.
        return self.batch_size * self.context_length + self.model_batch_size

    @property
    def logits_token_count(self) -> int:
        # SGLang prunes cold-prefill hidden states to one sample position per
        # request when input logprobs/full logits are not requested.
        return self.batch_size if self.phase == "prefill" else self.model_batch_size

    def to_dict(self) -> dict:
        result = asdict(self)
        result["token_count"] = self.token_count
        result["useful_token_count"] = self.useful_token_count
        result["model_batch_size"] = self.model_batch_size
        result["attention_pair_count"] = self.attention_pair_count
        result["logits_token_count"] = self.logits_token_count
        return result


@dataclass(frozen=True)
class ParallelWorkLedger:
    """Per-rank work identities for the Blackwell DP-attention/EP recipe."""

    attention_tp_size: int
    attention_dp_size: int
    dp_real_requests: tuple[int, ...]
    dp_mlp_aligned_rows: tuple[int, ...]
    dp_model_rows: tuple[int, ...]
    dp_padding_mode: str
    source_rows_per_attention_rank: tuple[int, ...]
    critical_attention_rows: int
    critical_model_rows: int
    global_model_rows: int
    routed_pair_instances: int
    sent_pairs_per_attention_rank_by_dp: tuple[int, ...]
    critical_sent_pairs_per_source_rank: int
    balanced_received_pairs_per_ep_rank: int
    bound_condition_id: str
    bound_condition: str
    topology_contract: str
    excluded_positive_term_ids: tuple[str, ...]
    notes: tuple[str, ...]

    @property
    def critical_source_rows_per_rank(self) -> int:
        return max(self.source_rows_per_attention_rank, default=0)

    def to_dict(self) -> dict:
        result = asdict(self)
        result["contract_id"] = "kimi_k3_blackwell_parallel_ledger_v1"
        result["critical_source_rows_per_rank"] = (
            self.critical_source_rows_per_rank
        )
        return result


@dataclass(frozen=True)
class EstimatorAssumptions:
    compute_efficiency: float = 1.0
    hbm_efficiency: float = 1.0
    collective_efficiency: float = 1.0
    collective_startup_seconds: float = 0.0
    mla_kv_read_amplification: float = 1.0
    decode_cuda_graph: bool = True
    blackwell_k3_fused_all_reduce: bool = True

    def validate(self) -> None:
        for name in (
            "compute_efficiency",
            "hbm_efficiency",
            "collective_efficiency",
        ):
            value = getattr(self, name)
            if not (0 < value <= 1):
                raise ValueError(f"{name} must be in (0, 1].")
        if self.collective_startup_seconds < 0:
            raise ValueError("collective_startup_seconds cannot be negative.")
        if self.mla_kv_read_amplification < 1:
            raise ValueError("mla_kv_read_amplification must be at least 1.")
        certificate_values = (
            self.compute_efficiency,
            self.hbm_efficiency,
            self.collective_efficiency,
            self.collective_startup_seconds,
            self.mla_kv_read_amplification,
        )
        if certificate_values != (1.0, 1.0, 1.0, 0.0, 1.0):
            raise ValueError(
                "Lower-bound certificates require unit efficiencies, zero "
                "collective startup, and unit MLA KV read amplification."
            )


_DEFAULT_ESTIMATOR_ASSUMPTIONS = EstimatorAssumptions()


@dataclass(frozen=True)
class CalculationProvenance:
    """Human-readable derivation for one reported operation number."""

    label: str
    formula: str
    substitution: str
    units: str
    result: float
    note: str | None = None

    def to_dict(self) -> dict:
        result = asdict(self)
        if self.note is None:
            result.pop("note")
        return result


@dataclass(frozen=True)
class OperationEstimate:
    id: str
    name: str
    category: str
    dependencies: tuple[str, ...]
    flops_per_rank: float
    hbm_bytes_per_rank: float
    logical_collective_bytes: float
    link_bytes_per_rank: float
    compute_kind: ComputeKind
    compute_seconds: float
    hbm_seconds: float
    communication_seconds: float
    duration_seconds: float
    start_seconds: float
    end_seconds: float
    bottleneck: str
    notes: tuple[str, ...]
    calculations: dict[str, CalculationProvenance]

    @property
    def arithmetic_intensity_flops_per_hbm_byte(self) -> float | None:
        if (
            self.flops_per_rank <= 0
            or self.hbm_bytes_per_rank <= 0
            or self.duration_seconds <= 0
        ):
            return None
        return self.flops_per_rank / self.hbm_bytes_per_rank

    @property
    def roofline_flops_per_second(self) -> float | None:
        if (
            self.flops_per_rank <= 0
            or self.hbm_bytes_per_rank <= 0
            or self.duration_seconds <= 0
        ):
            return None
        return self.flops_per_rank / self.duration_seconds

    def to_dict(self) -> dict:
        result = asdict(self)
        result["calculations"] = {
            field: calculation.to_dict()
            for field, calculation in self.calculations.items()
        }
        result["arithmetic_intensity_flops_per_hbm_byte"] = (
            self.arithmetic_intensity_flops_per_hbm_byte
        )
        result["roofline_flops_per_second"] = self.roofline_flops_per_second
        return result


@dataclass(frozen=True)
class LayerEstimate:
    name: str
    number: int | None
    attention: str | None
    ffn: str | None
    operations: tuple[OperationEstimate, ...]
    dependency_path_seconds: float
    compute_resource_seconds: float
    hbm_resource_seconds: float
    communication_resource_seconds: float
    latency_seconds: float
    limiting_certificates: tuple[str, ...]

    @property
    def dominant_operation(self) -> OperationEstimate:
        return max(self.operations, key=lambda op: op.duration_seconds)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "number": self.number,
            "attention": self.attention,
            "ffn": self.ffn,
            "critical_path_lower_bound_seconds": self.dependency_path_seconds,
            "compute_resource_seconds": self.compute_resource_seconds,
            "hbm_resource_seconds": self.hbm_resource_seconds,
            "communication_resource_seconds": self.communication_resource_seconds,
            "latency_seconds": self.latency_seconds,
            "limiting_certificates": list(self.limiting_certificates),
            "dominant_operation": self.dominant_operation.id,
            "operations": [op.to_dict() for op in self.operations],
        }


@dataclass(frozen=True)
class MemoryEstimate:
    static_weight_bytes_per_rank: float
    kda_state_bytes_per_rank: float
    mla_kv_cache_bytes_per_rank: float
    model_and_cache_bytes_per_rank: float
    attention_residual_bank_bytes_per_rank: float
    total_accounted_peak_bytes_per_rank: float
    nominal_hbm_capacity_bytes_per_rank: float
    fits_nominal_capacity: bool | None
    capacity_status: str
    weight_breakdown_bytes_per_rank: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EstimateResult:
    scope: str
    model_revision: str
    hardware: HardwareSpec
    workload: Workload
    assumptions: EstimatorAssumptions
    decode_cuda_graph_replay: bool
    parallel_work_ledger: ParallelWorkLedger | None
    layers: tuple[LayerEstimate, ...]
    total_seconds: float
    memory: MemoryEstimate
    warnings: tuple[str, ...]

    @property
    def ideal_tokens_per_second(self) -> float:
        return self.workload.useful_token_count / self.total_seconds

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "model_revision": self.model_revision,
            "hardware": self.hardware.to_dict(),
            "workload": self.workload.to_dict(),
            "assumptions": asdict(self.assumptions),
            "decode_cuda_graph_replay": self.decode_cuda_graph_replay,
            "parallel_work_ledger": (
                self.parallel_work_ledger.to_dict()
                if self.parallel_work_ledger is not None
                else None
            ),
            "total_seconds": self.total_seconds,
            "ideal_tokens_per_second": self.ideal_tokens_per_second,
            "memory": self.memory.to_dict(),
            "warnings": list(self.warnings),
            "layers": [layer.to_dict() for layer in self.layers],
        }


@dataclass(frozen=True)
class _PendingOperation:
    id: str
    name: str
    category: str
    dependencies: tuple[str, ...]
    flops_per_rank: float
    hbm_bytes_per_rank: float
    logical_collective_bytes: float
    link_bytes_per_rank: float
    compute_kind: ComputeKind
    compute_seconds: float
    hbm_seconds: float
    communication_seconds: float
    communication_local_seconds: float
    communication_remote_seconds: float
    notes: tuple[str, ...]
    calculations: dict[str, CalculationProvenance]


_CALCULATION_LABELS = {
    "flops_per_rank": "FLOPs per rank",
    "hbm_bytes_per_rank": "HBM bytes per rank",
    "logical_collective_bytes": "Logical collective payload",
    "link_bytes_per_rank": "Fabric-byte diagnostic",
    "compute_seconds": "Compute floor",
    "hbm_seconds": "HBM floor",
    "communication_seconds": "Communication floor",
    "duration_seconds": "Operator duration",
}

_CALCULATION_UNITS = {
    "flops_per_rank": "FLOP/rank",
    "hbm_bytes_per_rank": "bytes/rank",
    "logical_collective_bytes": "bytes",
    "link_bytes_per_rank": "bytes/rank",
    "compute_seconds": "seconds",
    "hbm_seconds": "seconds",
    "communication_seconds": "seconds",
    "duration_seconds": "seconds",
}


def _display_number(value: float) -> str:
    """Keep substitutions compact while preserving useful numeric precision."""

    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.15g}"


def _calculation(
    field: str,
    *,
    formula: str,
    substitution: str,
    result: float,
    note: str | None = None,
) -> CalculationProvenance:
    return CalculationProvenance(
        label=_CALCULATION_LABELS[field],
        formula=formula,
        substitution=substitution,
        units=_CALCULATION_UNITS[field],
        result=result,
        note=note,
    )


class _LayerBuilder:
    def __init__(
        self,
        *,
        name: str,
        number: int | None,
        attention: str | None,
        ffn: str | None,
        hardware: HardwareSpec,
        assumptions: EstimatorAssumptions,
    ) -> None:
        self.name = name
        self.number = number
        self.attention = attention
        self.ffn = ffn
        self.hardware = hardware
        self.assumptions = assumptions
        self._operations: list[_PendingOperation] = []

    def _peak(self, kind: ComputeKind) -> float:
        if kind == "bf16":
            return self.hardware.dense_bf16_flops_per_s
        if kind == "k3_expert":
            return self.hardware.k3_expert_flops_per_s
        return math.inf

    def add_roofline(
        self,
        *,
        op_id: str,
        name: str,
        category: str,
        dependencies: Sequence[str] = (),
        flops: float = 0,
        hbm_bytes: float = 0,
        compute_kind: ComputeKind = "bf16",
        flops_formula: str | None = None,
        flops_substitution: str | None = None,
        hbm_formula: str | None = None,
        hbm_substitution: str | None = None,
        notes: Sequence[str] = (),
    ) -> str:
        if flops and (not flops_formula or not flops_substitution):
            raise ValueError(f"{op_id} must provide an explicit FLOP derivation.")
        if hbm_bytes and (not hbm_formula or not hbm_substitution):
            raise ValueError(f"{op_id} must provide an explicit HBM derivation.")
        peak = self._peak(compute_kind)
        compute_seconds = (
            flops / (peak * self.assumptions.compute_efficiency) if flops else 0.0
        )
        hbm_seconds = (
            hbm_bytes
            / (
                self.hardware.hbm_bandwidth_bytes_per_s
                * self.assumptions.hbm_efficiency
            )
            if hbm_bytes
            else 0.0
        )
        calculations = {
            "flops_per_rank": _calculation(
                "flops_per_rank",
                formula=(flops_formula or "0; no floating-point work is counted"),
                substitution=(flops_substitution or "0"),
                result=flops,
            ),
            "hbm_bytes_per_rank": _calculation(
                "hbm_bytes_per_rank",
                formula=(hbm_formula or "0; no HBM traffic is counted"),
                substitution=(hbm_substitution or "0"),
                result=hbm_bytes,
            ),
            "logical_collective_bytes": _calculation(
                "logical_collective_bytes",
                formula="0; this is not a collective operator",
                substitution="0",
                result=0,
            ),
            "link_bytes_per_rank": _calculation(
                "link_bytes_per_rank",
                formula="0; this is not a collective operator",
                substitution="0",
                result=0,
            ),
            "compute_seconds": _calculation(
                "compute_seconds",
                formula=(
                    "FLOPs per rank / (peak FLOP/s per rank × compute efficiency)"
                    if flops
                    else "0; no floating-point work is counted"
                ),
                substitution=(
                    f"{_display_number(flops)} / "
                    f"({_display_number(peak)} × "
                    f"{_display_number(self.assumptions.compute_efficiency)})"
                    if flops
                    else "0"
                ),
                result=compute_seconds,
            ),
            "hbm_seconds": _calculation(
                "hbm_seconds",
                formula=(
                    "HBM bytes per rank / (HBM bytes/s per rank × HBM efficiency)"
                    if hbm_bytes
                    else "0; no HBM traffic is counted"
                ),
                substitution=(
                    f"{_display_number(hbm_bytes)} / "
                    f"({_display_number(self.hardware.hbm_bandwidth_bytes_per_s)} × "
                    f"{_display_number(self.assumptions.hbm_efficiency)})"
                    if hbm_bytes
                    else "0"
                ),
                result=hbm_seconds,
            ),
            "communication_seconds": _calculation(
                "communication_seconds",
                formula="0; this is not a collective operator",
                substitution="0",
                result=0,
            ),
        }
        self._operations.append(
            _PendingOperation(
                id=op_id,
                name=name,
                category=category,
                dependencies=tuple(dependencies),
                flops_per_rank=flops,
                hbm_bytes_per_rank=hbm_bytes,
                logical_collective_bytes=0,
                link_bytes_per_rank=0,
                compute_kind=compute_kind,
                compute_seconds=compute_seconds,
                hbm_seconds=hbm_seconds,
                communication_seconds=0,
                communication_local_seconds=0,
                communication_remote_seconds=0,
                notes=tuple(notes),
                calculations=calculations,
            )
        )
        return op_id

    def add_gemm(
        self,
        *,
        op_id: str,
        name: str,
        category: str,
        m: int,
        k: int,
        n: int,
        weight_bytes_per_parameter: float,
        dependencies: Sequence[str] = (),
        compute_kind: ComputeKind = "bf16",
        input_bytes_per_element: float = BF16_BYTES,
        output_bytes_per_element: float = BF16_BYTES,
        weight_batch_count: int = 1,
        notes: Sequence[str] = (),
    ) -> str:
        flops = 2.0 * m * k * n
        hbm_bytes = (
            weight_batch_count * k * n * weight_bytes_per_parameter
            + m * k * input_bytes_per_element
            + m * n * output_bytes_per_element
        )
        return self.add_roofline(
            op_id=op_id,
            name=name,
            category=category,
            dependencies=dependencies,
            flops=flops,
            hbm_bytes=hbm_bytes,
            compute_kind=compute_kind,
            flops_formula="2 × M × K × N",
            flops_substitution=(
                f"2 × {_display_number(m)} × {_display_number(k)} × {_display_number(n)}"
            ),
            hbm_formula=(
                "weight batches × K × N × weight bytes + M × K × input bytes "
                "+ M × N × output bytes"
            ),
            hbm_substitution=(
                f"{_display_number(weight_batch_count)} × {_display_number(k)} × "
                f"{_display_number(n)} × {_display_number(weight_bytes_per_parameter)} "
                f"+ {_display_number(m)} × {_display_number(k)} × "
                f"{_display_number(input_bytes_per_element)} + {_display_number(m)} × "
                f"{_display_number(n)} × {_display_number(output_bytes_per_element)}"
            ),
            notes=notes,
        )

    def add_collective(
        self,
        *,
        op_id: str,
        name: str,
        category: str,
        kind: CollectiveKind,
        logical_bytes: float,
        logical_bytes_formula: str,
        logical_bytes_substitution: str,
        group_size: int | None = None,
        local_domain_size: int | None = None,
        intra_link_bytes_per_rank: float | None = None,
        inter_link_bytes_per_rank: float | None = None,
        dependencies: Sequence[str] = (),
        flops: float = 0,
        hbm_bytes: float | None = None,
        compute_kind: ComputeKind = "none",
        flops_formula: str | None = None,
        flops_substitution: str | None = None,
        hbm_formula: str | None = None,
        hbm_substitution: str | None = None,
        notes: Sequence[str] = (),
    ) -> str:
        if flops and (not flops_formula or not flops_substitution):
            raise ValueError(f"{op_id} must provide an explicit FLOP derivation.")
        collective = _collective_cost(
            hardware=self.hardware,
            kind=kind,
            logical_bytes=logical_bytes,
            assumptions=self.assumptions,
            group_size=group_size,
            local_domain_size=local_domain_size,
            intra_link_bytes_per_rank=intra_link_bytes_per_rank,
            inter_link_bytes_per_rank=inter_link_bytes_per_rank,
        )
        communication_seconds = collective.seconds
        link_bytes = collective.link_bytes_per_rank
        peak = self._peak(compute_kind)
        compute_seconds = (
            flops / (peak * self.assumptions.compute_efficiency) if flops else 0.0
        )
        if hbm_bytes is None:
            if kind == "all_to_all":
                raise ValueError(f"{op_id} must provide explicit fused A2A HBM bytes.")
            # Logical device-memory floor: all-reduce reads and overwrites a
            # full local tensor; gather/scatter reads one side and writes the
            # other. Backend algorithms can move more.
            collective_size = group_size or self.hardware.tp_size
            hbm_bytes = (
                2 * logical_bytes
                if kind == "all_reduce"
                else logical_bytes * (1 + 1 / collective_size)
            )
            hbm_formula = (
                "2 × logical collective bytes"
                if kind == "all_reduce"
                else "logical collective bytes × (1 + 1 / collective group size)"
            )
            hbm_substitution = (
                f"2 × {_display_number(logical_bytes)}"
                if kind == "all_reduce"
                else (
                    f"{_display_number(logical_bytes)} × "
                    f"(1 + 1 / {_display_number(collective_size)})"
                )
            )
        elif hbm_bytes and (not hbm_formula or not hbm_substitution):
            raise ValueError(f"{op_id} must provide an explicit HBM derivation.")
        hbm_seconds = (
            hbm_bytes
            / (
                self.hardware.hbm_bandwidth_bytes_per_s
                * self.assumptions.hbm_efficiency
            )
            if hbm_bytes
            else 0.0
        )
        calculations = {
            "flops_per_rank": _calculation(
                "flops_per_rank",
                formula=(flops_formula or "0; no floating-point work is counted"),
                substitution=(flops_substitution or "0"),
                result=flops,
            ),
            "hbm_bytes_per_rank": _calculation(
                "hbm_bytes_per_rank",
                formula=(hbm_formula or "0; no HBM traffic is counted"),
                substitution=(hbm_substitution or "0"),
                result=hbm_bytes,
            ),
            "logical_collective_bytes": _calculation(
                "logical_collective_bytes",
                formula=logical_bytes_formula,
                substitution=logical_bytes_substitution,
                result=logical_bytes,
                note=(
                    "One-rank logical send/receive payload before locality; the "
                    "communication floor uses topology-resolved fabric bytes."
                    if kind == "all_to_all"
                    else "Full logical tensor size after the collective, before topology expansion."
                ),
            ),
            "link_bytes_per_rank": _calculation(
                "link_bytes_per_rank",
                formula=collective.link_formula,
                substitution=collective.link_substitution,
                result=link_bytes,
                note=(
                    "Sum of independent local- and remote-fabric directional "
                    "maxima; it is a non-additive diagnostic and need not belong "
                    "to one physical rank."
                    if kind == "all_to_all"
                    else None
                ),
            ),
            "compute_seconds": _calculation(
                "compute_seconds",
                formula=(
                    "FLOPs per rank / (peak FLOP/s per rank × compute efficiency)"
                    if flops
                    else "0; no floating-point work is counted"
                ),
                substitution=(
                    f"{_display_number(flops)} / "
                    f"({_display_number(peak)} × "
                    f"{_display_number(self.assumptions.compute_efficiency)})"
                    if flops
                    else "0"
                ),
                result=compute_seconds,
            ),
            "hbm_seconds": _calculation(
                "hbm_seconds",
                formula=(
                    "HBM bytes per rank / (HBM bytes/s per rank × HBM efficiency)"
                    if hbm_bytes
                    else "0; no HBM traffic is counted"
                ),
                substitution=(
                    f"{_display_number(hbm_bytes)} / "
                    f"({_display_number(self.hardware.hbm_bandwidth_bytes_per_s)} × "
                    f"{_display_number(self.assumptions.hbm_efficiency)})"
                    if hbm_bytes
                    else "0"
                ),
                result=hbm_seconds,
            ),
            "communication_seconds": _calculation(
                "communication_seconds",
                formula=collective.communication_formula,
                substitution=collective.communication_substitution,
                result=communication_seconds,
            ),
        }
        self._operations.append(
            _PendingOperation(
                id=op_id,
                name=name,
                category=category,
                dependencies=tuple(dependencies),
                flops_per_rank=flops,
                hbm_bytes_per_rank=hbm_bytes,
                logical_collective_bytes=logical_bytes,
                link_bytes_per_rank=link_bytes,
                compute_kind=compute_kind,
                compute_seconds=compute_seconds,
                hbm_seconds=hbm_seconds,
                communication_seconds=communication_seconds,
                communication_local_seconds=collective.local_seconds,
                communication_remote_seconds=collective.remote_seconds,
                notes=tuple(notes) + (collective.note,),
                calculations=calculations,
            )
        )
        return op_id

    def finish(self) -> LayerEstimate:
        known: dict[str, OperationEstimate] = {}
        result: list[OperationEstimate] = []
        for pending in self._operations:
            if pending.id in known:
                raise ValueError(f"Duplicate operation id {pending.id!r}.")
            missing = [dep for dep in pending.dependencies if dep not in known]
            if missing:
                raise ValueError(
                    f"Operation {pending.id!r} has unknown/out-of-order deps {missing}."
                )
            start = max(
                (known[dep].end_seconds for dep in pending.dependencies), default=0.0
            )
            floors = {
                "compute": pending.compute_seconds,
                "hbm": pending.hbm_seconds,
                "communication": pending.communication_seconds,
            }
            bottleneck, duration = max(floors.items(), key=lambda item: item[1])
            operation = OperationEstimate(
                id=pending.id,
                name=pending.name,
                category=pending.category,
                dependencies=pending.dependencies,
                flops_per_rank=pending.flops_per_rank,
                hbm_bytes_per_rank=pending.hbm_bytes_per_rank,
                logical_collective_bytes=pending.logical_collective_bytes,
                link_bytes_per_rank=pending.link_bytes_per_rank,
                compute_kind=pending.compute_kind,
                compute_seconds=pending.compute_seconds,
                hbm_seconds=pending.hbm_seconds,
                communication_seconds=pending.communication_seconds,
                duration_seconds=duration,
                start_seconds=start,
                end_seconds=start + duration,
                bottleneck=bottleneck,
                notes=pending.notes,
                calculations={
                    **pending.calculations,
                    "duration_seconds": _calculation(
                        "duration_seconds",
                        formula="max(compute floor, HBM floor, communication floor)",
                        substitution=(
                            f"max({_display_number(pending.compute_seconds)}, "
                            f"{_display_number(pending.hbm_seconds)}, "
                            f"{_display_number(pending.communication_seconds)})"
                        ),
                        result=duration,
                        note="Roofline floors overlap ideally; the slowest resource sets duration.",
                    ),
                },
            )
            known[pending.id] = operation
            result.append(operation)
        dependency_path = max((op.end_seconds for op in result), default=0.0)
        # A dependency DAG alone is not a valid GPU roofline: two overlapping
        # branches cannot each consume 100% of tensor-core or HBM capacity. Sum
        # each shared physical resource's normalized work to form independent
        # conservation floors, then take the maximum with the dependency path.
        compute_resource = sum(op.compute_seconds for op in result)
        hbm_resource = sum(op.hbm_seconds for op in result)
        communication_resource = max(
            sum(op.communication_local_seconds for op in self._operations),
            sum(op.communication_remote_seconds for op in self._operations),
        )
        certificates = (
            ("critical_path", dependency_path),
            ("compute", compute_resource),
            ("hbm", hbm_resource),
            ("communication", communication_resource),
        )
        latency = max(value for _, value in certificates)
        # Co-limiting means exact equality of the derived lower bounds. Do not
        # turn merely close values into a semantic tie with a display tolerance.
        limiting_certificates = tuple(
            name for name, value in certificates if value == latency
        )
        return LayerEstimate(
            name=self.name,
            number=self.number,
            attention=self.attention,
            ffn=self.ffn,
            operations=tuple(result),
            dependency_path_seconds=dependency_path,
            compute_resource_seconds=compute_resource,
            hbm_resource_seconds=hbm_resource,
            communication_resource_seconds=communication_resource,
            latency_seconds=latency,
            limiting_certificates=limiting_certificates,
        )


@dataclass(frozen=True)
class _CollectiveCost:
    seconds: float
    local_seconds: float
    remote_seconds: float
    link_bytes_per_rank: float
    note: str
    link_formula: str
    link_substitution: str
    communication_formula: str
    communication_substitution: str


def _collective_cost(
    *,
    hardware: HardwareSpec,
    kind: CollectiveKind,
    logical_bytes: float,
    assumptions: EstimatorAssumptions,
    group_size: int | None = None,
    local_domain_size: int | None = None,
    intra_link_bytes_per_rank: float | None = None,
    inter_link_bytes_per_rank: float | None = None,
) -> _CollectiveCost:
    """Return idealized per-rank collective time and link traffic.

    For reductions and gathers, ``logical_bytes`` is the full logical tensor.
    A2A callers provide topology-resolved critical directional bytes for each
    fabric. NVLink and scale-out rates are one-directional.
    """

    p = group_size or hardware.tp_size
    if p <= 0:
        raise ValueError("Collective group size must be positive.")
    local_p = min(local_domain_size or hardware.nvlink_domain_size, p)
    if p % local_p:
        raise ValueError("Collective group must contain whole local domains.")
    efficiency = assumptions.collective_efficiency
    alpha = assumptions.collective_startup_seconds
    nv_bw = hardware.nvlink_bytes_per_s_per_direction * efficiency

    if kind == "all_to_all":
        if intra_link_bytes_per_rank is None or inter_link_bytes_per_rank is None:
            raise ValueError("A2A requires topology-resolved directional bytes.")
        intra_bytes = intra_link_bytes_per_rank
        inter_bytes = inter_link_bytes_per_rank
        net_raw = hardware.scaleout_bytes_per_s_per_gpu_per_direction
        if inter_bytes and net_raw is None:
            raise ValueError("Cross-domain A2A requires a scale-out fabric.")
        local_seconds = intra_bytes / nv_bw
        remote_seconds = (
            inter_bytes / (net_raw * efficiency)
            if inter_bytes and net_raw is not None
            else 0.0
        )
        seconds = max(local_seconds, remote_seconds) + (alpha if p > 1 else 0.0)
        return _CollectiveCost(
            seconds=seconds,
            local_seconds=local_seconds,
            remote_seconds=remote_seconds,
            link_bytes_per_rank=intra_bytes + inter_bytes,
            note=(
                "Domain-resolved ideal-balanced A2A; independent NVLink and "
                "scale-out resource floors overlap."
            ),
            link_formula="critical local directional bytes + critical remote directional bytes",
            link_substitution=(
                f"{_display_number(intra_bytes)} + {_display_number(inter_bytes)}"
            ),
            communication_formula=(
                "max(local-link bytes / NVLink rate, remote bytes / scale-out "
                "rate) + startup"
                if inter_bytes
                else "local-link bytes / NVLink rate + startup"
            ),
            communication_substitution=(
                f"max({_display_number(intra_bytes)} / "
                f"({_display_number(hardware.nvlink_bytes_per_s_per_direction)} × "
                f"{_display_number(efficiency)}), {_display_number(inter_bytes)} / "
                f"({_display_number(net_raw)} × {_display_number(efficiency)})) + "
                f"{_display_number(alpha)}"
                if inter_bytes and net_raw is not None
                else (
                    f"{_display_number(intra_bytes)} / "
                    f"({_display_number(hardware.nvlink_bytes_per_s_per_direction)} × "
                    f"{_display_number(efficiency)}) + {_display_number(alpha)}"
                )
            ),
        )

    if p > local_p:
        # Explicit two-level resource floor. Chunked implementations may
        # pipeline the local and scale-out legs, so time is their maximum,
        # not their sum; this does not claim an exact NCCL algorithm.
        domains = p // local_p
        net_raw = hardware.scaleout_bytes_per_s_per_gpu_per_direction
        if net_raw is None:
            raise ValueError("Cross-domain collective requires a scale-out fabric.")
        net_bw = net_raw * efficiency
        if kind == "all_reduce":
            # Each rank must recover one full-tensor aggregate not present in
            # its input.  Under this explicit two-level contract, that is one
            # full tensor on the local fabric and one local shard on scale-out.
            intra_bytes = logical_bytes
            inter_bytes = logical_bytes / local_p
            note = (
                "Two-level all-reduce information floor across "
                f"{domains} domains and {local_p} local ranks."
            )
            link_formula = (
                "logical bytes + logical bytes / local ranks"
            )
            link_substitution = (
                f"{_display_number(logical_bytes)} + "
                f"{_display_number(logical_bytes)} / {local_p}"
            )
        else:
            # Gather corresponding rank shards across domains first, then
            # gather the larger shards within each local NVLink domain. The
            # byte floor is the same for reduce-scatter in reverse.
            inter_bytes = (domains - 1) / domains * logical_bytes / local_p
            intra_bytes = (local_p - 1) / local_p * logical_bytes
            collective_name = (
                "reduce-scatter" if kind == "reduce_scatter" else "all-gather"
            )
            note = (
                f"Hierarchical {collective_name} across {domains} domains and "
                f"{local_p} local ranks."
            )
            link_formula = (
                "(local ranks − 1) / local ranks × logical bytes + "
                "(domains − 1) / domains × logical bytes / local ranks"
            )
            link_substitution = (
                f"({local_p} − 1) / {local_p} × {_display_number(logical_bytes)} "
                f"+ ({domains} − 1) / {domains} × "
                f"{_display_number(logical_bytes)} / {local_p}"
            )
        seconds = max(intra_bytes / nv_bw, inter_bytes / net_bw) + alpha
        local_seconds = intra_bytes / nv_bw
        remote_seconds = inter_bytes / net_bw
        return _CollectiveCost(
            seconds=seconds,
            local_seconds=local_seconds,
            remote_seconds=remote_seconds,
            link_bytes_per_rank=intra_bytes + inter_bytes,
            note=note,
            link_formula=link_formula,
            link_substitution=link_substitution,
            communication_formula=(
                "max(intra-domain bytes / (NVLink bytes/s × collective efficiency), "
                "inter-domain bytes / (scale-out bytes/s × collective efficiency)) + startup"
            ),
            communication_substitution=(
                f"max({_display_number(intra_bytes)} / "
                f"({_display_number(hardware.nvlink_bytes_per_s_per_direction)} × "
                f"{_display_number(efficiency)}), {_display_number(inter_bytes)} / "
                f"({_display_number(net_raw)} × {_display_number(efficiency)})) + "
                f"{_display_number(alpha)}"
            ),
        )

    if kind == "all_reduce":
        link_bytes = logical_bytes if p > 1 else 0.0
        note = (
            "All-reduce receive-information floor inside one "
            "nonblocking NVLink domain."
        )
        link_formula = "logical bytes if group ranks > 1, otherwise 0"
        link_substitution = (
            _display_number(logical_bytes) if p > 1 else "0 (single rank)"
        )
    else:
        link_bytes = (p - 1) / p * logical_bytes
        collective_name = (
            "reduce-scatter" if kind == "reduce_scatter" else "all-gather"
        )
        note = (
            f"Ideal ring {collective_name} inside one nonblocking NVLink domain."
        )
        link_formula = "(group ranks − 1) / group ranks × logical bytes"
        link_substitution = f"({p} − 1) / {p} × {_display_number(logical_bytes)}"
    return _CollectiveCost(
        seconds=link_bytes / nv_bw + alpha,
        local_seconds=link_bytes / nv_bw,
        remote_seconds=0,
        link_bytes_per_rank=link_bytes,
        note=note,
        link_formula=link_formula,
        link_substitution=link_substitution,
        communication_formula=(
            "link bytes per rank / (NVLink bytes/s × collective efficiency) + startup"
        ),
        communication_substitution=(
            f"{_display_number(link_bytes)} / "
            f"({_display_number(hardware.nvlink_bytes_per_s_per_direction)} × "
            f"{_display_number(efficiency)}) + {_display_number(alpha)}"
        ),
    )


def _residual_aggregate(
    builder: _LayerBuilder,
    *,
    op_id: str,
    dependencies: Sequence[str],
    token_count: int,
    hidden_size: int,
    previous_blocks: int,
    write_snapshot: bool,
    final_output: bool = False,
) -> str:
    h200_unfused = builder.hardware.family == "h200"
    # Count only one compulsory arithmetic operation per output element. The
    # larger logical score/norm estimate is intentionally excluded because it
    # is not an instruction-count lower bound for every fused implementation.
    flops = token_count * hidden_size
    flops_formula = "tokens × hidden size compulsory output operations"
    flops_substitution = f"{token_count} × {hidden_size}"
    if previous_blocks == 0:
        hbm_bytes = token_count * hidden_size * 2 * BF16_BYTES
        hbm_formula = "tokens × hidden size × 2 passes × BF16 bytes"
        hbm_substitution = f"{token_count} × {hidden_size} × 2 × {BF16_BYTES}"
    else:
        rows = previous_blocks + 1
        # Lower-bound traffic reads each candidate row in score and combine
        # passes, then writes the normalized result.
        hbm_bytes = (
            2 * rows * token_count * hidden_size * BF16_BYTES
            + token_count * hidden_size * BF16_BYTES
        )
        hbm_formula = (
            "2 × rows × tokens × hidden size × BF16 bytes + "
            "tokens × hidden size × BF16 bytes"
        )
        hbm_substitution = (
            f"2 × {rows} × {token_count} × {hidden_size} × {BF16_BYTES} + "
            f"{token_count} × {hidden_size} × {BF16_BYTES}"
        )
        if h200_unfused:
            # Triton score and mix write/read FP32 score scratch; the mixture is
            # then written and reread by a separate RMSNorm kernel.
            hbm_bytes += (
                2 * token_count * rows * FP32_BYTES
                + 2 * token_count * hidden_size * BF16_BYTES
            )
            hbm_formula += (
                " + 2 × tokens × rows × FP32 scratch bytes + "
                "2 × tokens × hidden size × BF16 bytes"
            )
            hbm_substitution += (
                f" + 2 × {token_count} × {rows} × {FP32_BYTES} + "
                f"2 × {token_count} × {hidden_size} × {BF16_BYTES}"
            )
    if write_snapshot:
        snapshot_passes = 2 if h200_unfused else 1
        hbm_bytes += snapshot_passes * token_count * hidden_size * BF16_BYTES
        hbm_formula += " + snapshot passes × tokens × hidden size × BF16 bytes"
        hbm_substitution += (
            f" + {snapshot_passes} × {token_count} × {hidden_size} × {BF16_BYTES}"
        )
    return builder.add_roofline(
        op_id=op_id,
        name=(
            "Final attention-residual aggregation and RMSNorm"
            if final_output
            else "Attention-residual aggregation and RMSNorm"
        ),
        category="attention_residual",
        dependencies=dependencies,
        flops=flops,
        hbm_bytes=hbm_bytes,
        flops_formula=flops_formula,
        flops_substitution=flops_substitution,
        hbm_formula=hbm_formula,
        hbm_substitution=hbm_substitution,
        notes=(
            f"Uses {previous_blocks} frozen snapshot rows plus the current stream.",
            (
                "H200 counts separate Triton score/mix, RMSNorm, and snapshot-copy traffic."
                if h200_unfused
                else "Blackwell counts the fused SM100+ TMA aggregation kernel."
            ),
            "Compute counts only one compulsory output operation per element; score, softmax, and normalization arithmetic are excluded.",
            "HBM traffic is a logical materialization condition, not measured transactions.",
        ),
    )


def _attention_all_reduce(
    builder: _LayerBuilder,
    *,
    dependency: str,
    token_count: int,
    hidden_size: int,
    fuse_pending_prefix: bool,
    reduce_scatter: bool = False,
) -> str:
    logical_bytes = token_count * hidden_size * BF16_BYTES
    group_size = builder.hardware.attention_tp_size
    output_tokens = token_count // group_size if reduce_scatter else token_count
    if reduce_scatter and token_count % group_size:
        raise ValueError("SP-MoE reduce-scatter rows must divide attention TP.")
    fusion_note = (
        "SP lower-bound bundle consumes the pending BF16 prefix in the "
        "reduce-scatter epilogue; GB300 custom SP implements this, while other "
        "paths may materialize more work."
        if reduce_scatter
        else "K3 fused AR consumes the pending BF16 prefix in its epilogue."
    )
    return builder.add_collective(
        op_id=("attention_reduce_scatter" if reduce_scatter else "attention_all_reduce"),
        name=(
            "Attention output TP reduce-scatter for SP-MoE"
            if reduce_scatter and not fuse_pending_prefix
            else (
                "Attention output TP reduce-scatter + fused pending-prefix add"
                if reduce_scatter
                else (
                    "Attention output TP all-reduce + fused pending-prefix add"
                    if fuse_pending_prefix
                    else "Attention output TP all-reduce"
                )
            )
        ),
        category="communication",
        kind="reduce_scatter" if reduce_scatter else "all_reduce",
        logical_bytes=logical_bytes,
        logical_bytes_formula="tokens × hidden size × BF16 bytes",
        logical_bytes_substitution=(f"{token_count} × {hidden_size} × {BF16_BYTES}"),
        group_size=group_size,
        dependencies=(dependency,),
        flops=output_tokens * hidden_size if fuse_pending_prefix else 0,
        hbm_bytes=(
            logical_bytes * (1 + 2 / group_size)
            if fuse_pending_prefix and reduce_scatter
            else (3 * logical_bytes if fuse_pending_prefix else None)
        ),
        compute_kind="bf16" if fuse_pending_prefix else "none",
        flops_formula=(
            (
                "output-shard tokens × hidden size (fused prefix add)"
                if reduce_scatter
                else "tokens × hidden size (fused prefix add)"
            )
            if fuse_pending_prefix
            else None
        ),
        flops_substitution=(
            f"{output_tokens} × {hidden_size}" if fuse_pending_prefix else None
        ),
        hbm_formula=(
            "logical input bytes + 2 × reduce-scatter output-shard bytes"
            if fuse_pending_prefix and reduce_scatter
            else "3 × logical collective bytes (output, prefix, and result)"
            if fuse_pending_prefix
            else None
        ),
        hbm_substitution=(
            f"{_display_number(logical_bytes)} × (1 + 2 / {group_size})"
            if fuse_pending_prefix and reduce_scatter
            else f"3 × {_display_number(logical_bytes)}"
            if fuse_pending_prefix
            else None
        ),
        notes=(fusion_note,) if fuse_pending_prefix else (),
    )


def _kda_attention(
    builder: _LayerBuilder,
    *,
    root: str,
    workload: Workload,
    config: KimiK3TextConfig,
    hardware: HardwareSpec,
    assumptions: EstimatorAssumptions,
    fuse_pending_prefix: bool,
    collective_token_count: int | None = None,
    reduce_scatter: bool = False,
) -> str:
    tokens = workload.token_count
    tp = hardware.attention_tp_size
    heads = hardware.local_attention_heads
    projection = config.projection_size

    wide = builder.add_gemm(
        op_id="kda_qkvg",
        name="Fused KDA Q/K/V/full-rank-gate projection",
        category="kda_projection",
        m=tokens,
        k=config.hidden_size,
        n=4 * projection // tp,
        weight_bytes_per_parameter=BF16_BYTES,
        dependencies=(root,),
        notes=("Attention weights are BF16 and column-sharded by TP.",),
    )
    overlap_limit = hardware.decode_overlap_token_limit
    overlap_tiny_projection = (
        workload.phase == "decode"
        and assumptions.decode_cuda_graph
        and tokens <= overlap_limit
    )
    tiny_deps = (root,) if overlap_tiny_projection else (wide,)
    tiny_logical_width = config.head_dim + heads
    tiny_padded_width = math.ceil(tiny_logical_width / 8) * 8
    tiny = builder.add_gemm(
        op_id="kda_bfa",
        name="Merged KDA f_a and beta projection",
        category="kda_projection",
        m=tokens,
        k=config.hidden_size,
        n=tiny_padded_width,
        weight_bytes_per_parameter=BF16_BYTES,
        dependencies=tiny_deps,
        notes=(
            f"Logical width {tiny_logical_width} is padded to {tiny_padded_width} for the GEMM.",
            (
                "Overlaps the wide projection under captured decode within the "
                f"{overlap_limit}-token hardware limit."
                if overlap_tiny_projection
                else "Serialized after the wide projection for this workload."
            ),
        ),
    )
    forget = builder.add_gemm(
        op_id="kda_forget",
        name="KDA low-rank forget-gate projection",
        category="kda_projection",
        m=tokens,
        k=config.head_dim,
        n=projection // tp,
        weight_bytes_per_parameter=BF16_BYTES,
        dependencies=(tiny,),
    )

    k = config.head_dim
    v = config.v_head_dim
    # CUDA-graph padding executes dummy KDA rows too, including their state and
    # convolution-state traffic. Persistent capacity below still uses real B.
    state_sequences = workload.model_batch_size
    recurrence_flops = tokens * heads * k * v
    recurrence_formula = "tokens × local heads × K × V compulsory state operations"
    recurrence_substitution = f"{tokens} × {heads} × {k} × {v}"
    recurrence_note = (
        "The compute certificate counts only one compulsory arithmetic operation "
        "per dense K×V state element; benchmark approximation coefficients are excluded."
    )
    activation_bytes = tokens * heads * (3 * k + v) * BF16_BYTES
    gate_bytes = tokens * heads * k * FP32_BYTES
    beta_bytes = tokens * heads * FP32_BYTES
    state_bytes = (
        2
        * state_sequences
        * heads
        * k
        * v
        * hardware.kda_state_bytes_per_element
    )
    conv_state_bytes = (
        2
        * state_sequences
        * 3
        * heads
        * k
        * (config.short_conv_kernel_size - 1)
        * BF16_BYTES
    )
    fused_blackwell_decode = (
        workload.phase == "decode" and hardware.kda_fused_decode_capable
    )
    if fused_blackwell_decode:
        core = builder.add_roofline(
            op_id="kda_fused_decode",
            name="Fused KDA conv update, recurrence, and gated RMSNorm",
            category="kda_core",
            dependencies=(wide, forget),
            flops=(
                recurrence_flops
                + 2 * tokens * 3 * heads * k * config.short_conv_kernel_size
                + 8 * tokens * heads * v
            ),
            hbm_bytes=(
                activation_bytes
                + gate_bytes
                + beta_bytes
                + state_bytes
                + conv_state_bytes
            ),
            flops_formula=(
                "recurrence FLOPs + 2 × tokens × 3 convs × local heads × K × "
                "kernel width + 8 × tokens × local heads × V"
            ),
            flops_substitution=(
                f"{_display_number(recurrence_flops)} + 2 × {tokens} × 3 × "
                f"{heads} × {k} × {config.short_conv_kernel_size} + "
                f"8 × {tokens} × {heads} × {v}"
            ),
            hbm_formula=(
                "activation bytes + gate bytes + beta bytes + state read/write bytes + "
                "convolution-state read/write bytes"
            ),
            hbm_substitution=(
                f"{_display_number(activation_bytes)} + {_display_number(gate_bytes)} + "
                f"{_display_number(beta_bytes)} + {_display_number(state_bytes)} + "
                f"{_display_number(conv_state_bytes)}"
            ),
            notes=(
                recurrence_note,
                "TP8 uses SGLang's shape-gated fused KDA decode kernel.",
                "State traffic assumes one recipe-dtype KDA state read and write per sequence.",
            ),
        )
    else:
        conv = builder.add_roofline(
            op_id="kda_short_convs",
            name=(
                "Three KDA causal-convolution updates"
                if workload.phase == "decode"
                else "Three width-4 KDA causal convolutions"
            ),
            category="kda_core",
            dependencies=(wide, forget),
            flops=(2 * tokens * 3 * heads * k * config.short_conv_kernel_size),
            hbm_bytes=(6 * tokens * heads * k * BF16_BYTES + conv_state_bytes),
            flops_formula=(
                "2 × tokens × 3 convolutions × local heads × K × kernel width"
            ),
            flops_substitution=(
                f"2 × {tokens} × 3 × {heads} × {k} × {config.short_conv_kernel_size}"
            ),
            hbm_formula=(
                "6 activation passes × tokens × local heads × K × BF16 bytes + "
                "convolution-state read/write bytes"
            ),
            hbm_substitution=(
                f"6 × {tokens} × {heads} × {k} × {BF16_BYTES} + "
                f"{_display_number(conv_state_bytes)}"
            ),
            notes=(
                (
                    "H200 decode uses the fallback convolution update; cold prefill "
                    "uses separate convolution kernels on all three presets."
                ),
            ),
        )
        recurrence = builder.add_roofline(
            op_id="kda_recurrence",
            name=(
                "Packed Triton KDA recurrence"
                if workload.phase == "decode"
                else "Chunked Triton KDA recurrence"
            ),
            category="kda_core",
            dependencies=(conv, forget),
            flops=recurrence_flops,
            hbm_bytes=(activation_bytes + gate_bytes + beta_bytes + state_bytes),
            flops_formula=recurrence_formula,
            flops_substitution=recurrence_substitution,
            hbm_formula=(
                "activation bytes + gate bytes + beta bytes + state read/write bytes"
            ),
            hbm_substitution=(
                f"{_display_number(activation_bytes)} + {_display_number(gate_bytes)} + "
                f"{_display_number(beta_bytes)} + {_display_number(state_bytes)}"
            ),
            notes=(
                recurrence_note,
                "State traffic assumes one recipe-dtype KDA state read and write per sequence.",
                "The prefill recurrence is itself multi-kernel; its internal workspace traffic is not known here.",
            ),
        )
        core = builder.add_roofline(
            op_id="kda_gated_rmsnorm",
            name="KDA sigmoid-gated RMSNorm",
            category="normalization",
            dependencies=(recurrence,),
            flops=8 * tokens * heads * v,
            hbm_bytes=3 * tokens * heads * v * BF16_BYTES,
            flops_formula="8 × tokens × local heads × V",
            flops_substitution=f"8 × {tokens} × {heads} × {v}",
            hbm_formula="3 passes × tokens × local heads × V × BF16 bytes",
            hbm_substitution=(f"3 × {tokens} × {heads} × {v} × {BF16_BYTES}"),
        )
    o_proj = builder.add_gemm(
        op_id="kda_o_proj",
        name="KDA output projection",
        category="attention_output",
        m=tokens,
        k=projection // tp,
        n=config.hidden_size,
        weight_bytes_per_parameter=BF16_BYTES,
        dependencies=(core,),
    )
    return _attention_all_reduce(
        builder,
        dependency=o_proj,
        token_count=collective_token_count or tokens,
        hidden_size=config.hidden_size,
        fuse_pending_prefix=fuse_pending_prefix,
        reduce_scatter=reduce_scatter,
    )


def _mla_attention(
    builder: _LayerBuilder,
    *,
    root: str,
    workload: Workload,
    config: KimiK3TextConfig,
    hardware: HardwareSpec,
    assumptions: EstimatorAssumptions,
    fuse_pending_prefix: bool,
    collective_token_count: int | None = None,
    reduce_scatter: bool = False,
) -> str:
    tokens = workload.token_count
    local_heads = hardware.local_attention_heads
    a_out = config.q_lora_rank + config.kv_lora_rank + config.qk_rope_head_dim
    overlap_limit = hardware.decode_overlap_token_limit
    overlap_output_gate = False
    kv_cache_bytes = hardware.kv_cache_bytes_per_element

    a_proj = builder.add_gemm(
        op_id="mla_a_proj",
        name="Replicated fused MLA q/kv latent projection",
        category="mla_projection",
        m=tokens,
        k=config.hidden_size,
        n=a_out,
        weight_bytes_per_parameter=BF16_BYTES,
        dependencies=(root,),
        notes=("This projection is replicated, not divided by TP.",),
    )
    norms = builder.add_roofline(
        op_id="mla_latent_norms",
        name="MLA q-latent and kv-latent RMSNorm",
        category="normalization",
        dependencies=(a_proj,),
        flops=5.0 * tokens * (config.q_lora_rank + config.kv_lora_rank),
        hbm_bytes=(
            2 * tokens * (config.q_lora_rank + config.kv_lora_rank) * BF16_BYTES
        ),
        flops_formula="5 × tokens × (q LoRA rank + kv LoRA rank)",
        flops_substitution=(
            f"5 × {tokens} × ({config.q_lora_rank} + {config.kv_lora_rank})"
        ),
        hbm_formula=("2 passes × tokens × (q LoRA rank + kv LoRA rank) × BF16 bytes"),
        hbm_substitution=(
            f"2 × {tokens} × ({config.q_lora_rank} + {config.kv_lora_rank}) × "
            f"{BF16_BYTES}"
        ),
    )
    q_proj = builder.add_gemm(
        op_id="mla_q_b",
        name="MLA q_b projection",
        category="mla_projection",
        m=tokens,
        k=config.q_lora_rank,
        n=local_heads * config.mla_qk_head_dim,
        weight_bytes_per_parameter=BF16_BYTES,
        dependencies=(norms,),
    )

    if workload.phase == "prefill":
        # The scoped recipes use eager cold prefill. H200 FA3 and Blackwell
        # TRT-LLM MLA handlers both dispatch it to materialized MHA rather than
        # the absorbed decode path.
        kv_proj = builder.add_gemm(
            op_id="mla_kv_b",
            name="MLA materialized K/V projection for cold prefill",
            category="mla_projection",
            m=tokens,
            k=config.kv_lora_rank,
            n=local_heads * (config.qk_nope_head_dim + config.v_head_dim),
            weight_bytes_per_parameter=BF16_BYTES,
            dependencies=(q_proj,),
            notes=("Default eager cold prefill dispatches MHA and materializes K/V.",),
        )
        pair_flops = (
            workload.attention_pair_count
            * local_heads
            * 2
            * (config.mla_qk_head_dim + config.v_head_dim)
        )
        qkvo_bytes = (
            tokens
            * local_heads
            * 2
            * (config.mla_qk_head_dim + config.v_head_dim)
            * BF16_BYTES
        )
        cache_write = tokens * config.mla_latent_cache_dim * kv_cache_bytes
        core = builder.add_roofline(
            op_id="mla_mha_core",
            name="Causal MHA core for cold prefill",
            category="mla_attention_core",
            dependencies=(kv_proj,),
            flops=pair_flops,
            hbm_bytes=qkvo_bytes + cache_write,
            flops_formula=(
                "causal attention pairs × local heads × 2 × "
                "(MLA QK width + value width)"
            ),
            flops_substitution=(
                f"{workload.attention_pair_count} × {local_heads} × 2 × "
                f"({config.mla_qk_head_dim} + {config.v_head_dim})"
            ),
            hbm_formula="Q/K/V/O logical bytes + latent-cache write bytes",
            hbm_substitution=(
                f"{tokens} × {local_heads} × 2 × "
                f"({config.mla_qk_head_dim} + {config.v_head_dim}) × {BF16_BYTES} "
                f"+ {tokens} × {config.mla_latent_cache_dim} × {kv_cache_bytes}"
            ),
            notes=(
                "HBM bytes are a FlashAttention-style logical minimum: Q/K/V/O plus latent-cache write once.",
                "K3 constructs MLA with skip_rope=True, so no RoPE operation is charged.",
            ),
        )
        gate_deps = (core,)
        output_input = core
    else:
        absorb = builder.add_gemm(
            op_id="mla_q_absorb",
            name="MLA absorbed q-nope batched projection",
            category="mla_projection",
            m=tokens * local_heads,
            k=config.qk_nope_head_dim,
            n=config.kv_lora_rank,
            weight_bytes_per_parameter=BF16_BYTES,
            weight_batch_count=local_heads,
            dependencies=(q_proj,),
            notes=("Uses one distinct absorbed K matrix per local head.",),
        )
        pair_flops = (
            workload.attention_pair_count
            * local_heads
            * 2
            * (config.mla_latent_cache_dim + config.kv_lora_rank)
        )
        assert workload.context_length is not None
        cache_read = (
            workload.batch_size
            * workload.context_length
            * config.mla_latent_cache_dim
            * kv_cache_bytes
            * assumptions.mla_kv_read_amplification
        )
        query_output_bytes = (
            tokens
            * local_heads
            * (config.mla_latent_cache_dim + config.kv_lora_rank)
            * BF16_BYTES
        )
        cache_write = tokens * config.mla_latent_cache_dim * kv_cache_bytes
        core = builder.add_roofline(
            op_id="mla_absorbed_core",
            name="Absorbed MLA decode attention core",
            category="mla_attention_core",
            dependencies=(absorb,),
            flops=pair_flops,
            hbm_bytes=cache_read + query_output_bytes + cache_write,
            flops_formula=(
                "(real batch × context + padded model tokens) × local heads × 2 × "
                "(latent-cache width + kv LoRA rank)"
            ),
            flops_substitution=(
                f"({workload.batch_size} × {workload.context_length} + {tokens}) × "
                f"{local_heads} × 2 × "
                f"({config.mla_latent_cache_dim} + {config.kv_lora_rank})"
            ),
            hbm_formula=(
                "real batch × context × latent-cache width × cache-element bytes × read "
                "amplification + padded tokens × local heads × (latent-cache width + "
                "kv LoRA rank) × BF16 bytes + padded tokens × latent-cache width × cache-element bytes"
            ),
            hbm_substitution=(
                f"{workload.batch_size} × {workload.context_length} × "
                f"{config.mla_latent_cache_dim} × {kv_cache_bytes} × "
                f"{_display_number(assumptions.mla_kv_read_amplification)} + "
                f"{tokens} × {local_heads} × ({config.mla_latent_cache_dim} + "
                f"{config.kv_lora_rank}) × {BF16_BYTES} + {tokens} × "
                f"{config.mla_latent_cache_dim} × {kv_cache_bytes}"
            ),
            notes=(
                "KV-cache HBM traffic assumes the configured read-amplification factor; 1.0 is a logical minimum.",
                "K3 constructs MLA with skip_rope=True, so no RoPE operation is charged.",
            ),
        )
        output_input = builder.add_gemm(
            op_id="mla_v_deabsorb",
            name="MLA latent-value de-absorption projection",
            category="mla_projection",
            m=tokens * local_heads,
            k=config.kv_lora_rank,
            n=config.v_head_dim,
            weight_bytes_per_parameter=BF16_BYTES,
            weight_batch_count=local_heads,
            dependencies=(core,),
            notes=("Uses one distinct de-absorption V matrix per local head.",),
        )
        # The output-gate GEMM is issued on an alternate stream at the start of
        # attention and joins immediately before the gate multiply/o_proj.
        overlap_output_gate = assumptions.decode_cuda_graph and tokens <= overlap_limit
        gate_deps = (root,) if overlap_output_gate else (output_input,)

    gate = builder.add_gemm(
        op_id="mla_output_gate",
        name="MLA full-rank output-gate projection",
        category="mla_projection",
        m=tokens,
        k=config.hidden_size,
        n=local_heads * config.v_head_dim,
        weight_bytes_per_parameter=BF16_BYTES,
        dependencies=gate_deps,
        notes=(
            (
                "Overlaps the main MLA chain under non-breakable captured decode "
                f"within the {overlap_limit}-token hardware limit."
                if workload.phase == "decode" and overlap_output_gate
                else "Runs on the critical path immediately before gate apply."
            ),
        ),
    )
    gated = builder.add_roofline(
        op_id="mla_gate_apply",
        name="MLA sigmoid output gate",
        category="elementwise",
        dependencies=(output_input, gate),
        flops=5.0 * tokens * local_heads * config.v_head_dim,
        hbm_bytes=3 * tokens * local_heads * config.v_head_dim * BF16_BYTES,
        flops_formula="5 × tokens × local heads × value width",
        flops_substitution=(f"5 × {tokens} × {local_heads} × {config.v_head_dim}"),
        hbm_formula="3 passes × tokens × local heads × value width × BF16 bytes",
        hbm_substitution=(
            f"3 × {tokens} × {local_heads} × {config.v_head_dim} × {BF16_BYTES}"
        ),
    )
    o_proj = builder.add_gemm(
        op_id="mla_o_proj",
        name="MLA output projection",
        category="attention_output",
        m=tokens,
        k=local_heads * config.v_head_dim,
        n=config.hidden_size,
        weight_bytes_per_parameter=BF16_BYTES,
        dependencies=(gated,),
    )
    return _attention_all_reduce(
        builder,
        dependency=o_proj,
        token_count=collective_token_count or tokens,
        hidden_size=config.hidden_size,
        fuse_pending_prefix=fuse_pending_prefix,
        reduce_scatter=reduce_scatter,
    )


def _dense_ffn(
    builder: _LayerBuilder,
    *,
    root: str,
    workload: Workload,
    config: KimiK3TextConfig,
    hardware: HardwareSpec,
    work_ledger: ParallelWorkLedger | None = None,
) -> str:
    tokens = (
        work_ledger.global_model_rows
        if work_ledger is not None
        else workload.token_count
    )
    dense_root = root
    if work_ledger is not None and hardware.attention_dp_size > 1:
        global_bytes = tokens * config.hidden_size * BF16_BYTES
        global_local_domain = (
            min(8, hardware.tp_size)
            if hardware.family == "b300"
            else hardware.tp_size
        )
        if work_ledger.dp_padding_mode == "sum_len":
            dense_root = builder.add_collective(
                op_id="dense_dp_gather_all_reduce",
                name="SUM_LEN DP gather via global-TP all-reduce",
                category="communication",
                kind="all_reduce",
                logical_bytes=global_bytes,
                logical_bytes_formula=(
                    "global model rows × hidden size × BF16 bytes"
                ),
                logical_bytes_substitution=(
                    f"{tokens} × {config.hidden_size} × {BF16_BYTES}"
                ),
                group_size=hardware.tp_size,
                local_domain_size=global_local_domain,
                dependencies=(root,),
                notes=(
                    "Pinned SGLang SUM_LEN writes DP-local slices into one global buffer and all-reduces it across global TP.",
                ),
            )
        else:
            local_bytes = (
                work_ledger.critical_model_rows
                * config.hidden_size
                * BF16_BYTES
            )
            local_scatter = builder.add_collective(
                op_id="dense_dp_gather_tp_reduce_scatter",
                name="MAX_LEN attention-TP row reduce-scatter",
                category="communication",
                kind="reduce_scatter",
                logical_bytes=local_bytes,
                logical_bytes_formula=(
                    "critical DP rows × hidden size × BF16 bytes"
                ),
                logical_bytes_substitution=(
                    f"{work_ledger.critical_model_rows} × {config.hidden_size} × "
                    f"{BF16_BYTES}"
                ),
                group_size=hardware.attention_tp_size,
                local_domain_size=hardware.attention_tp_size,
                dependencies=(root,),
                notes=(
                    "Pinned SGLang first reduce-scatters each padded DP-local tensor across attention TP8.",
                ),
            )
            dense_root = builder.add_collective(
                op_id="dense_dp_gather_global_all_gather",
                name="MAX_LEN sharded-row global-TP all-gather",
                category="communication",
                kind="all_gather",
                logical_bytes=global_bytes,
                logical_bytes_formula=(
                    "global model rows × hidden size × BF16 bytes"
                ),
                logical_bytes_substitution=(
                    f"{tokens} × {config.hidden_size} × {BF16_BYTES}"
                ),
                group_size=hardware.tp_size,
                local_domain_size=global_local_domain,
                dependencies=(local_scatter,),
                notes=(
                    "Pinned SGLang then all-gathers the TP8 row shards across global TP.",
                ),
            )
    local_intermediate = config.dense_intermediate_size // hardware.tp_size
    gate_up = builder.add_gemm(
        op_id="dense_gate_up",
        name="Dense SiTU gate/up projection",
        category="dense_ffn",
        m=tokens,
        k=config.hidden_size,
        n=2 * local_intermediate,
        weight_bytes_per_parameter=BF16_BYTES,
        dependencies=(dense_root,),
    )
    activation = builder.add_roofline(
        op_id="dense_situ",
        name="Dense SiTU activation",
        category="elementwise",
        dependencies=(gate_up,),
        flops=8.0 * tokens * local_intermediate,
        hbm_bytes=3 * tokens * local_intermediate * BF16_BYTES,
        flops_formula="8 × tokens × TP-local intermediate width",
        flops_substitution=f"8 × {tokens} × {local_intermediate}",
        hbm_formula="3 passes × tokens × TP-local intermediate width × BF16 bytes",
        hbm_substitution=(f"3 × {tokens} × {local_intermediate} × {BF16_BYTES}"),
    )
    down = builder.add_gemm(
        op_id="dense_down",
        name="Dense down projection",
        category="dense_ffn",
        m=tokens,
        k=local_intermediate,
        n=config.hidden_size,
        weight_bytes_per_parameter=BF16_BYTES,
        dependencies=(activation,),
    )
    reduced = builder.add_collective(
        op_id="dense_all_reduce",
        name="Dense FFN TP all-reduce",
        category="communication",
        kind="all_reduce",
        logical_bytes=tokens * config.hidden_size * BF16_BYTES,
        logical_bytes_formula="tokens × hidden size × BF16 bytes",
        logical_bytes_substitution=(f"{tokens} × {config.hidden_size} × {BF16_BYTES}"),
        dependencies=(down,),
    )
    reduced_local = reduced
    prefix_tokens = tokens
    if work_ledger is not None and hardware.attention_dp_size > 1:
        prefix_tokens = work_ledger.critical_model_rows
        reduced_local = builder.add_roofline(
            op_id="dense_dp_scatter",
            name="Local DP row selection after dense FFN",
            category="communication",
            dependencies=(reduced,),
            hbm_bytes=(
                2 * prefix_tokens * config.hidden_size * BF16_BYTES
            ),
            hbm_formula=(
                "2 copy passes × critical DP rows × hidden size × BF16 bytes"
            ),
            hbm_substitution=(
                f"2 × {prefix_tokens} × {config.hidden_size} × {BF16_BYTES}"
            ),
            notes=("SGLang dp_scatter is a local slice/copy after global TP reduction.",),
        )
    return builder.add_roofline(
        op_id="dense_prefix_add",
        name="Dense FFN prefix/residual add",
        category="elementwise",
        dependencies=(reduced_local,),
        flops=prefix_tokens * config.hidden_size,
        hbm_bytes=3 * prefix_tokens * config.hidden_size * BF16_BYTES,
        flops_formula="tokens × hidden size",
        flops_substitution=f"{prefix_tokens} × {config.hidden_size}",
        hbm_formula="3 passes × tokens × hidden size × BF16 bytes",
        hbm_substitution=(
            f"3 × {prefix_tokens} × {config.hidden_size} × {BF16_BYTES}"
        ),
        notes=("KimiK3MLP adds the decoder prefix_sum after down projection.",),
    )


def _marlin_m_block_size(*, tokens: int, top_k: int, local_experts: int) -> int:
    """Returns SGLang Marlin's selected token-row block size."""

    pairs_per_local_expert = tokens * top_k / local_experts
    for block_size in (8, 16, 32, 48, 64):
        if pairs_per_local_expert / block_size < 0.9:
            return block_size
    return 64


def _blackwell_a2a_link_bytes(
    *,
    sent_pairs_by_dp: tuple[int, ...],
    attention_tp_size: int,
    ep_size: int,
    local_domain_size: int,
    routed_width: int,
) -> tuple[float, float]:
    """Return conditional critical directional bytes per local/remote fabric."""

    sent_pairs_by_rank = tuple(
        sent_pairs
        for sent_pairs in sent_pairs_by_dp
        for _ in range(attention_tp_size)
    )
    if len(sent_pairs_by_rank) != ep_size or ep_size % local_domain_size:
        raise AssertionError("K3 A2A rank layout must contain whole local domains.")

    total_sent_pairs = sum(sent_pairs_by_rank)
    critical_local_bytes = 0.0
    critical_remote_bytes = 0.0
    for rank, sent_pairs in enumerate(sent_pairs_by_rank):
        domain_start = rank // local_domain_size * local_domain_size
        domain_sent_pairs = sum(
            sent_pairs_by_rank[domain_start : domain_start + local_domain_size]
        )
        local_sent_pairs = sent_pairs * (local_domain_size - 1) / ep_size
        local_received_pairs = (domain_sent_pairs - sent_pairs) / ep_size
        remote_sent_pairs = sent_pairs * (ep_size - local_domain_size) / ep_size
        remote_received_pairs = (
            total_sent_pairs - domain_sent_pairs
        ) / ep_size

        local_outbound = (
            FP8_BYTES * local_sent_pairs
            + BF16_BYTES * local_received_pairs
        )
        local_inbound = (
            FP8_BYTES * local_received_pairs
            + BF16_BYTES * local_sent_pairs
        )
        remote_outbound = (
            FP8_BYTES * remote_sent_pairs
            + BF16_BYTES * remote_received_pairs
        )
        remote_inbound = (
            FP8_BYTES * remote_received_pairs
            + BF16_BYTES * remote_sent_pairs
        )
        critical_local_bytes = max(
            critical_local_bytes,
            routed_width * max(local_outbound, local_inbound),
        )
        critical_remote_bytes = max(
            critical_remote_bytes,
            routed_width * max(remote_outbound, remote_inbound),
        )
    return critical_local_bytes, critical_remote_bytes


def _blackwell_a2a_moe_ffn(
    builder: _LayerBuilder,
    *,
    root: str,
    config: KimiK3TextConfig,
    hardware: HardwareSpec,
    work_ledger: ParallelWorkLedger,
) -> str:
    """Account for K3's MegaMoE/DeepGEMM path without TP-MoE formulas."""

    source_rows = work_ledger.critical_source_rows_per_rank
    received_pairs = work_ledger.balanced_received_pairs_per_ep_rank
    routed_width = config.routed_expert_hidden_size
    expert_width = config.moe_intermediate_size
    shared_width = config.shared_intermediate_size

    router = builder.add_roofline(
        op_id="moe_ep_router",
        name="EP router projection",
        category="moe_front",
        dependencies=(root,),
        flops=2.0 * source_rows * config.hidden_size * config.num_experts,
        hbm_bytes=(
            config.hidden_size * config.num_experts * BF16_BYTES
            + source_rows * config.num_experts * FP32_BYTES
        ),
        flops_formula="2 × source rows × hidden size × experts",
        flops_substitution=(
            f"2 × {source_rows} × {config.hidden_size} × {config.num_experts}"
        ),
        hbm_formula=(
            "hidden size × experts × BF16 weight bytes + source rows × experts "
            "× FP32 router-intermediate bytes"
        ),
        hbm_substitution=(
            f"{config.hidden_size} × {config.num_experts} × {BF16_BYTES} + "
            f"{source_rows} × {config.num_experts} × {FP32_BYTES}"
        ),
        notes=(
            "The pinned overlap path can run router/TopK concurrently with latent-down.",
            "The source activation read is counted once on the latent-down branch; merged or cache-sharing implementations need not read it twice from HBM.",
        ),
    )
    latent_down = builder.add_gemm(
        op_id="moe_ep_latent_down",
        name="EP routed latent-down projection",
        category="moe_front",
        m=source_rows,
        k=config.hidden_size,
        n=routed_width,
        weight_bytes_per_parameter=BF16_BYTES,
        dependencies=(root,),
        notes=(
            "This branch may overlap router/TopK and joins it before pre-dispatch quantization.",
        ),
    )

    topk = builder.add_roofline(
        op_id="moe_topk",
        name="Top-16 routing",
        category="moe_routing",
        dependencies=(router,),
        hbm_bytes=(
            source_rows * config.num_experts * FP32_BYTES
            + config.num_experts * FP32_BYTES
            + source_rows
            * config.num_experts_per_token
            * (FP32_BYTES + 4)
        ),
        hbm_formula=(
            "source rows × experts × FP32 router-intermediate read + expert "
            "correction bias + source rows × top-k × "
            "(FP32 weight + int32 ID)"
        ),
        hbm_substitution=(
            f"{source_rows} × {config.num_experts} × {FP32_BYTES} + "
            f"{config.num_experts} × {FP32_BYTES} "
            f"+ {source_rows} × {config.num_experts_per_token} × "
            f"({FP32_BYTES} + 4)"
        ),
        notes=(
            "The public EP front completes TopK before issuing the shared-expert side stream.",
            "TopK instruction count is excluded because no source-backed FLOP lower bound is available.",
        ),
    )

    predispatch = builder.add_roofline(
        op_id="moe_predispatch_quant",
        name="MegaMoE FP8 pre-dispatch quantization",
        category="moe_routing",
        dependencies=(topk, latent_down),
        hbm_bytes=source_rows * routed_width * (BF16_BYTES + 1),
        hbm_formula=(
            "source rows × routed width × (BF16 input read + FP8 value write)"
        ),
        hbm_substitution=(
            f"{source_rows} × {routed_width} × ({BF16_BYTES} + 1)"
        ),
        notes=(
            "Pre-dispatch quantization is once per source row, not once per expert pair.",
            "Quantization FLOPs and FP8 scale traffic are excluded positive terms.",
        ),
    )

    shared_gate_up = builder.add_gemm(
        op_id="moe_shared_gate_up_tp1",
        name="Replicated TP1 shared-expert gate/up",
        category="moe_shared",
        m=source_rows,
        k=config.hidden_size,
        n=2 * shared_width,
        weight_bytes_per_parameter=BF16_BYTES,
        dependencies=(topk,),
        notes=(
            "K3 replicates the complete BF16 shared expert on every EP A2A rank.",
            "This branch is issued on a side stream while MegaMoE executes.",
        ),
    )
    shared_situ = builder.add_roofline(
        op_id="moe_shared_situ_tp1",
        name="Replicated TP1 shared-expert SiTU",
        category="moe_shared",
        dependencies=(shared_gate_up,),
        flops=8.0 * source_rows * shared_width,
        hbm_bytes=3 * source_rows * shared_width * BF16_BYTES,
        flops_formula="8 × source rows × full shared width",
        flops_substitution=f"8 × {source_rows} × {shared_width}",
        hbm_formula="3 passes × source rows × full shared width × BF16 bytes",
        hbm_substitution=(
            f"3 × {source_rows} × {shared_width} × {BF16_BYTES}"
        ),
    )
    shared_down = builder.add_gemm(
        op_id="moe_shared_down_tp1",
        name="Replicated TP1 shared-expert down projection",
        category="moe_shared",
        m=source_rows,
        k=shared_width,
        n=config.hidden_size,
        weight_bytes_per_parameter=BF16_BYTES,
        dependencies=(shared_situ,),
    )

    touched_experts = 1.0 if received_pairs else 0.0
    params_per_expert = 3 * routed_width * expert_width
    weight_bytes = touched_experts * (
        params_per_expert * config.mxfp4_weight_bytes_per_parameter
    )
    activation_bytes = (
        received_pairs * routed_width * FP8_BYTES
        + source_rows * routed_width * BF16_BYTES
    )
    mega_hbm = weight_bytes + activation_bytes
    mega_flops = (
        received_pairs * 6 * routed_width * expert_width
    )
    local_domain_size = (
        min(8, hardware.tp_size)
        if hardware.family == "b300"
        else hardware.tp_size
    )
    intra_link_bytes, inter_link_bytes = _blackwell_a2a_link_bytes(
        sent_pairs_by_dp=work_ledger.sent_pairs_per_attention_rank_by_dp,
        attention_tp_size=hardware.attention_tp_size,
        ep_size=hardware.ep_size,
        local_domain_size=local_domain_size,
        routed_width=routed_width,
    )
    payload_bytes = routed_width * max(
        FP8_BYTES * work_ledger.critical_sent_pairs_per_source_rank
        + BF16_BYTES * received_pairs,
        FP8_BYTES * received_pairs
        + BF16_BYTES * work_ledger.critical_sent_pairs_per_source_rank,
    )
    mega = builder.add_collective(
        op_id="moe_routed_experts",
        name="MegaMoE fused A2A dispatch, DeepGEMM experts, and combine",
        category="moe_experts",
        kind="all_to_all",
        logical_bytes=payload_bytes,
        logical_bytes_formula=(
            "routed width × max(critical-rank FP8 dispatch + BF16 combine sent, "
            "critical-rank FP8 dispatch + BF16 combine received)"
        ),
        logical_bytes_substitution=(
            f"{routed_width} × max({work_ledger.critical_sent_pairs_per_source_rank} "
            f"× {FP8_BYTES} + {received_pairs} × {BF16_BYTES}, "
            f"{received_pairs} × {FP8_BYTES} + "
            f"{work_ledger.critical_sent_pairs_per_source_rank} × {BF16_BYTES})"
        ),
        group_size=hardware.ep_size,
        local_domain_size=local_domain_size,
        intra_link_bytes_per_rank=intra_link_bytes,
        inter_link_bytes_per_rank=inter_link_bytes,
        dependencies=(predispatch,),
        flops=mega_flops,
        hbm_bytes=mega_hbm,
        compute_kind="k3_expert",
        flops_formula=(
            "balanced received pairs × 6 × routed width × expert width"
        ),
        flops_substitution=(
            f"{received_pairs} × 6 × {routed_width} × {expert_width}"
        ),
        hbm_formula=(
            "one deterministically touched expert × 3 × routed width × expert "
            "width × MXFP4 bytes + balanced received pairs × FP8 "
            "routed input + source rows × routed width × BF16 combined output"
        ),
        hbm_substitution=(
            f"{_display_number(touched_experts)} × 3 × {routed_width} × "
            f"{expert_width} × "
            f"{_display_number(config.mxfp4_weight_bytes_per_parameter)} + "
            f"{received_pairs} × {routed_width} × {FP8_BYTES} + {source_rows} × "
            f"{routed_width} × {BF16_BYTES}"
        ),
        notes=(
            "Certified routed FLOPs include only the three grouped GEMMs (6RI per received token-expert pair).",
            "Certified HBM includes one touched expert plus visible FP8 input and BF16 combined output; expected occupancy and internal intermediate traffic are excluded.",
            "The directional FP8-dispatch/BF16-combine payload is conditional on ideal-balanced routing.",
            "Dispatch, grouped GEMMs, and combine are one composite roofline operation because MegaMoE pipelines them.",
        ),
    )

    latent_norm = builder.add_roofline(
        op_id="moe_latent_norm",
        name="MegaMoE combined latent RMSNorm",
        category="normalization",
        dependencies=(mega,),
        flops=5.0 * source_rows * routed_width,
        hbm_bytes=(
            2 * source_rows * routed_width * BF16_BYTES
            + routed_width * BF16_BYTES
        ),
        flops_formula="5 × source rows × routed width",
        flops_substitution=f"5 × {source_rows} × {routed_width}",
        hbm_formula=(
            "2 passes × source rows × routed width × BF16 bytes + RMSNorm weight"
        ),
        hbm_substitution=(
            f"2 × {source_rows} × {routed_width} × {BF16_BYTES} + "
            f"{routed_width} × {BF16_BYTES}"
        ),
        notes=("MegaMoE combine already returns complete rows; no TP reduce follows.",),
    )
    latent_up = builder.add_gemm(
        op_id="moe_latent_up_replicated",
        name="Replicated latent-up projection",
        category="moe_tail",
        m=source_rows,
        k=routed_width,
        n=config.hidden_size,
        weight_bytes_per_parameter=BF16_BYTES,
        dependencies=(latent_norm,),
    )
    tail = builder.add_roofline(
        op_id="moe_tail_add",
        name="Routed, TP1 shared, and residual tail add",
        category="elementwise",
        dependencies=(latent_up, shared_down),
        flops=2.0 * source_rows * config.hidden_size,
        hbm_bytes=4 * source_rows * config.hidden_size * BF16_BYTES,
        flops_formula="2 adds × source rows × hidden size",
        flops_substitution=f"2 × {source_rows} × {config.hidden_size}",
        hbm_formula="4 tensor passes × source rows × hidden size × BF16 bytes",
        hbm_substitution=(
            f"4 × {source_rows} × {config.hidden_size} × {BF16_BYTES}"
        ),
    )
    return builder.add_collective(
        op_id="moe_sp_all_gather",
        name="SP-MoE row all-gather after MoE tail",
        category="communication",
        kind="all_gather",
        logical_bytes=(
            work_ledger.critical_model_rows * config.hidden_size * BF16_BYTES
        ),
        logical_bytes_formula="critical DP rows × hidden size × BF16 bytes",
        logical_bytes_substitution=(
            f"{work_ledger.critical_model_rows} × {config.hidden_size} × "
            f"{BF16_BYTES}"
        ),
        group_size=hardware.attention_tp_size,
        dependencies=(tail,),
        notes=(
            "RS + AG moves the same main BF16 payload as the former attention all-reduce.",
        ),
    )


def _moe_ffn(
    builder: _LayerBuilder,
    *,
    root: str,
    workload: Workload,
    config: KimiK3TextConfig,
    hardware: HardwareSpec,
    assumptions: EstimatorAssumptions,
) -> str:
    tokens = workload.token_count
    tp = hardware.tp_size
    shard = hardware.moe_shard_size
    marlin_w4a16 = "marlin" in hardware.moe_backend.lower()
    blackwell_mxfp8 = (
        not marlin_w4a16
        and hardware.family in ("b300", "gb300")
        and hardware.moe_sharding == "tp"
    )
    shared_local = config.shared_intermediate_size // tp
    front_n = 2 * shared_local + config.num_experts + config.routed_expert_hidden_size
    front = builder.add_gemm(
        op_id="moe_fused_front",
        name="Merged shared gate/up, router, and latent-down projection",
        category="moe_front",
        m=tokens,
        k=config.hidden_size,
        n=front_n,
        weight_bytes_per_parameter=BF16_BYTES,
        dependencies=(root,),
        output_bytes_per_element=BF16_BYTES,
        notes=(
            "Shared gate/up is TP-sharded; router and latent-down outputs are replicated.",
            "Assumes SGLang's post-load merged front is active for plain TP/EP without A2A.",
        ),
    )
    front_ready = front
    if not blackwell_mxfp8 and tokens > 1:
        front_ready = builder.add_roofline(
            op_id="moe_routed_input_contiguous",
            name="Marlin routed-latent contiguous materialization",
            category="moe_front",
            dependencies=(front,),
            hbm_bytes=(2 * tokens * config.routed_expert_hidden_size * BF16_BYTES),
            hbm_formula="2 copy passes × tokens × routed-latent width × BF16 bytes",
            hbm_substitution=(
                f"2 × {tokens} × {config.routed_expert_hidden_size} × {BF16_BYTES}"
            ),
            notes=(
                "H200 Marlin requires the strided fused-front latent slice to be contiguous for T>1.",
                "The current-stream copy completes before ordinary shared and routed work.",
            ),
    )
    if hardware.moe_sharding == "ep":
        local_experts = hardware.local_routed_experts
        local_expert_intermediate = config.moe_intermediate_size
        expert_bias_element_bytes = BF16_BYTES
        params_per_selected_expert = (
            3 * config.routed_expert_hidden_size * config.moe_intermediate_size
        )
        sharding_note = (
            "Ideal-balance lower bound for the critical active EP rank over "
            "EP-local full experts; no token all-to-all is present."
        )
    else:
        local_experts = config.num_experts
        local_expert_intermediate = (
            hardware.routed_expert_intermediate_size_per_partition
        )
        expert_bias_element_bytes = FP32_BYTES if blackwell_mxfp8 else BF16_BYTES
        params_per_selected_expert = (
            3 * config.routed_expert_hidden_size * local_expert_intermediate
        )
        sharding_note = "All expert IDs are present as TP-sharded expert slices."

    if hardware.moe_sharding == "ep":
        # The critical pair-count rank processes ceil(TK/EP) pairs. Since one
        # token cannot select the same expert twice, that rank must touch at
        # least ceil(K/EP) distinct local experts; every token can reuse them.
        unique = float(
            math.ceil(config.num_experts_per_token / hardware.ep_size)
        )
        unique_formula = "ceil(top-k / EP ranks)"
        unique_substitution = (
            f"ceil({config.num_experts_per_token} / {hardware.ep_size})"
        )
        occupancy_note = (
            "The critical active EP rank deterministically touches at least "
            f"{int(unique)} local expert weight set(s)."
        )
    else:
        # Every token selects top-k distinct IDs and every TP rank holds a slice
        # of every expert. Multiple tokens may reuse the same top-k set.
        unique = float(config.num_experts_per_token)
        unique_formula = "top-k distinct experts"
        unique_substitution = str(config.num_experts_per_token)
        occupancy_note = (
            "Every TP rank deterministically touches at least the same "
            f"{int(unique)} distinct expert weight slices."
        )

    expert_pairs = tokens * config.num_experts_per_token
    local_compute_instances = (
        (expert_pairs + hardware.ep_size - 1) // hardware.ep_size
        if hardware.moe_sharding == "ep"
        else expert_pairs
    )
    expert_backend_notes: tuple[str, ...] = ()
    if marlin_w4a16:
        marlin_m_block = _marlin_m_block_size(
            tokens=tokens,
            top_k=config.num_experts_per_token,
            local_experts=local_experts,
        )
        # For every routing realization, padded rows are at least both the
        # useful-row count and one M-block per deterministically required active
        # expert.
        padded_compute_instances = max(
            local_compute_instances, unique * marlin_m_block
        )
        expert_flops = (
            padded_compute_instances
            * 6
            * config.routed_expert_hidden_size
            * local_expert_intermediate
            + expert_pairs * 8 * local_expert_intermediate
        )
        expert_flops_formula = (
            "max(local useful pairs, minimum active local experts × Marlin M-block) "
            "× 6 × routed width × local expert intermediate + tokens × top-k × "
            "8 × local expert intermediate"
        )
        expert_flops_substitution = (
            f"max({_display_number(local_compute_instances)}, "
            f"{_display_number(unique)} × {marlin_m_block}) × 6 × "
            f"{config.routed_expert_hidden_size} × {local_expert_intermediate} + "
            f"{tokens} × {config.num_experts_per_token} × 8 × "
            f"{local_expert_intermediate}"
        )
        expert_backend_notes = (
            f"Marlin selects M-block {marlin_m_block}; the compute floor uses "
            f"{padded_compute_instances:.3f} padded row-equivalents per rank.",
            "The padded-row count uses only the deterministic minimum active "
            "experts, not expected routing occupancy.",
        )
    elif hardware.moe_sharding == "ep":
        expert_flops = local_compute_instances * (
            6 * config.routed_expert_hidden_size * config.moe_intermediate_size
            + 8 * config.moe_intermediate_size
        )
        expert_flops_formula = (
            "ceil(tokens × top-k / EP) × (6 × routed width × expert intermediate + "
            "8 × expert intermediate)"
        )
        expert_flops_substitution = (
            f"ceil({tokens} × {config.num_experts_per_token} / {shard}) × "
            f"(6 × {config.routed_expert_hidden_size} × "
            f"{config.moe_intermediate_size} "
            f"+ 8 × {config.moe_intermediate_size})"
        )
    else:
        expert_flops = (
            expert_pairs
            * (
                6 * config.routed_expert_hidden_size * local_expert_intermediate
                + 8 * local_expert_intermediate
            )
        )
        expert_flops_formula = (
            "tokens × top-k × (6 × routed width × TP-local padded intermediate + "
            "8 × TP-local padded intermediate)"
        )
        expert_flops_substitution = (
            f"{tokens} × {config.num_experts_per_token} × "
            f"(6 × {config.routed_expert_hidden_size} × {local_expert_intermediate} "
            f"+ 8 × {local_expert_intermediate})"
        )
    expert_weight_bytes = unique * (
        params_per_selected_expert * config.mxfp4_weight_bytes_per_parameter
        + (2 * local_expert_intermediate + config.routed_expert_hidden_size)
        * expert_bias_element_bytes
    )
    if hardware.moe_sharding == "ep" and marlin_w4a16:
        # Marlin's caches retain all global token/expert pairs on every EP rank.
        # Only GEMM rows that map to local experts scale down by EP.
        marlin_pair_bytes = (
            max(2 * local_expert_intermediate, config.routed_expert_hidden_size)
            + 3 * local_expert_intermediate
            + 2 * config.routed_expert_hidden_size
        ) * BF16_BYTES
        local_gemm_bytes = (
            local_compute_instances
            * (2 * config.routed_expert_hidden_size + 3 * local_expert_intermediate)
            * BF16_BYTES
        )
        expert_activation_bytes = (
            expert_pairs * marlin_pair_bytes
            + local_gemm_bytes
            + tokens * config.routed_expert_hidden_size * BF16_BYTES
        )
        expert_activation_formula = (
            "tokens × top-k × (max(2 × expert intermediate, routed width) + "
            "3 × expert intermediate + 2 × routed width) × BF16 bytes + "
            "ceil(tokens × top-k / EP) × (2 × routed width + "
            "3 × expert intermediate) "
            "× BF16 bytes + tokens × routed width × BF16 output bytes"
        )
        expert_activation_substitution = (
            f"{tokens} × {config.num_experts_per_token} × "
            f"(max(2 × {local_expert_intermediate}, "
            f"{config.routed_expert_hidden_size}) + 3 × "
            f"{local_expert_intermediate} + 2 × "
            f"{config.routed_expert_hidden_size}) × {BF16_BYTES} + "
            f"({_display_number(local_compute_instances)}) × "
            f"(2 × {config.routed_expert_hidden_size} + 3 × "
            f"{local_expert_intermediate}) × {BF16_BYTES} + {tokens} × "
            f"{config.routed_expert_hidden_size} × {BF16_BYTES}"
        )
    elif hardware.moe_sharding == "ep":
        expert_activation_bytes = (
            local_compute_instances
            * (2 * config.routed_expert_hidden_size + 3 * config.moe_intermediate_size)
            * BF16_BYTES
        )
        expert_activation_formula = (
            "ceil(tokens × top-k / EP) × (2 × routed width + "
            "3 × expert intermediate) "
            "× BF16 bytes"
        )
        expert_activation_substitution = (
            f"ceil({tokens} × {config.num_experts_per_token} / {shard}) × "
            f"(2 × {config.routed_expert_hidden_size} + 3 × "
            f"{config.moe_intermediate_size}) × {BF16_BYTES}"
        )
    elif blackwell_mxfp8:
        # Every selected token/expert pair is present on every TP rank. GEMM1
        # consumes group-32 MXFP8 values/scales; the routed output is BF16 and
        # only the expert intermediate is TP-sharded.
        mxfp8_input_bytes = config.routed_expert_hidden_size * (
            1 + 1 / config.mxfp4_group_size
        )
        expert_activation_bytes = (
            tokens
            * config.num_experts_per_token
            * (
                mxfp8_input_bytes
                + config.routed_expert_hidden_size * BF16_BYTES
                + 3 * local_expert_intermediate * BF16_BYTES
            )
        )
        expert_activation_formula = (
            "tokens × top-k × (routed width × (1 FP8 byte + 1/group scale byte) + "
            "routed width × BF16 bytes + 3 × TP-local padded intermediate × BF16 bytes)"
        )
        expert_activation_substitution = (
            f"{tokens} × {config.num_experts_per_token} × "
            f"({_display_number(mxfp8_input_bytes)} + "
            f"{config.routed_expert_hidden_size} × {BF16_BYTES} + 3 × "
            f"{local_expert_intermediate} × {BF16_BYTES})"
        )
    else:
        # TP-sharded Marlin zero-initializes a shared [pairs, max(2I, K)]
        # cache (K for K3), then materializes GEMM1 [pairs, 2I], SiTU
        # [pairs, I], GEMM2 [pairs, K], and the final top-k reduction output
        # [tokens, K]. Count every producer write and consumer read, matching
        # the logical HBM convention used by the decomposed dense/shared paths.
        expert_activation_bytes = (
            expert_pairs
            * (4 * config.routed_expert_hidden_size + 6 * local_expert_intermediate)
            * BF16_BYTES
            + tokens * config.routed_expert_hidden_size * BF16_BYTES
        )
        expert_activation_formula = (
            "tokens × top-k × (4 × routed width + 6 × TP-local padded intermediate) "
            "× BF16 bytes + tokens × routed width × BF16 output bytes"
        )
        expert_activation_substitution = (
            f"{tokens} × {config.num_experts_per_token} × "
            f"(4 × {config.routed_expert_hidden_size} + "
            f"6 × {local_expert_intermediate}) × {BF16_BYTES} + "
            f"{tokens} × {config.routed_expert_hidden_size} × {BF16_BYTES}"
        )
    metadata_bytes = tokens * config.num_experts_per_token * (FP32_BYTES + 4)
    expert_activation_bytes += metadata_bytes
    expert_activation_formula += " + tokens × top-k × (FP32 weight + int32 ID bytes)"
    expert_activation_substitution += (
        f" + {tokens} × {config.num_experts_per_token} × ({FP32_BYTES} + 4)"
    )

    plain_collective_path = (
        hardware.moe_sharding == "ep"
        or not hardware.k3_fused_all_reduce_capable
        or not assumptions.blackwell_k3_fused_all_reduce
    )

    def add_shared_branch(dependencies: Sequence[str]) -> str:
        shared_situ = builder.add_roofline(
            op_id="moe_shared_situ",
            name="Shared-expert SiTU activation",
            category="moe_shared",
            dependencies=dependencies,
            flops=8.0 * tokens * shared_local,
            hbm_bytes=3 * tokens * shared_local * BF16_BYTES,
            flops_formula="8 × tokens × TP-local shared intermediate",
            flops_substitution=f"8 × {tokens} × {shared_local}",
            hbm_formula=(
                "3 passes × tokens × TP-local shared intermediate × BF16 bytes"
            ),
            hbm_substitution=(f"3 × {tokens} × {shared_local} × {BF16_BYTES}"),
            notes=(
                (
                    "Runs before routed experts on the ordinary path."
                    if plain_collective_path
                    else "Runs on SGLang's alternate stream while routed experts execute."
                ),
            ),
        )
        return builder.add_gemm(
            op_id="moe_shared_down",
            name="Shared-expert down projection",
            category="moe_shared",
            m=tokens,
            k=shared_local,
            n=config.hidden_size,
            weight_bytes_per_parameter=BF16_BYTES,
            dependencies=(shared_situ,),
        )

    shared_down = add_shared_branch((front_ready,)) if plain_collective_path else None
    route_dependencies = (shared_down,) if shared_down is not None else (front_ready,)
    fused_route_quant = blackwell_mxfp8 and tokens <= 64
    route_flops = float(tokens * config.num_experts)
    route_hbm_bytes = (
        tokens * config.num_experts * BF16_BYTES
        + config.num_experts * FP32_BYTES
        + tokens * config.num_experts_per_token * (FP32_BYTES + 4)
    )
    route_flops_formula = "tokens × total experts compulsory correction-bias adds"
    route_flops_substitution = f"{tokens} × {config.num_experts}"
    route_hbm_formula = (
        "tokens × total experts × BF16 logit bytes + total experts × FP32 bias bytes + "
        "tokens × top-k × (FP32 weight + int32 ID bytes)"
    )
    route_hbm_substitution = (
        f"{tokens} × {config.num_experts} × {BF16_BYTES} + "
        f"{config.num_experts} × {FP32_BYTES} + {tokens} × "
        f"{config.num_experts_per_token} × ({FP32_BYTES} + 4)"
    )
    if fused_route_quant:
        route_hbm_bytes += tokens * (
            config.routed_expert_hidden_size
            * (BF16_BYTES + 1 + 1 / config.mxfp4_group_size)
            + config.num_experts_per_token * 4
        )
        route_hbm_formula += (
            " + tokens × (routed width × (BF16 input + FP8 output + 1/group scale) "
            "+ top-k × packed int32 metadata)"
        )
        route_hbm_substitution += (
            f" + {tokens} × ({config.routed_expert_hidden_size} × "
            f"({BF16_BYTES} + 1 + 1 / {config.mxfp4_group_size}) + "
            f"{config.num_experts_per_token} × 4)"
        )
    route = builder.add_roofline(
        op_id="moe_route_topk",
        name=(
            "Fused radix route, top-k pack, and MXFP8 quantization"
            if fused_route_quant
            else "Sigmoid router and exact top-16 selection"
        ),
        category="moe_routing",
        dependencies=route_dependencies,
        flops=route_flops,
        hbm_bytes=route_hbm_bytes,
        flops_formula=route_flops_formula,
        flops_substitution=route_flops_substitution,
        hbm_formula=route_hbm_formula,
        hbm_substitution=route_hbm_substitution,
        notes=(
            (
                "The merged front emits a BF16 router slice; sigmoid, correction "
                "bias, and ranking use FP32 arithmetic, while the fused radix "
                "kernel keeps full score rows register-resident."
            ),
            (
                "HBM counts BF16 logits, one ideal correction-bias read, and only "
                "the FP32 top-k weights/int32 IDs as output."
            ),
            (
                "At <=64 Blackwell tokens, the same launch packs top-k metadata and quantizes the BF16 routed latent to group-32 MXFP8."
                if fused_route_quant
                else (
                    "Blackwell uses the standalone routed-input quant/pack path above 64 tokens."
                    if blackwell_mxfp8
                    else "H200 Marlin consumes BF16 routed activations without MXFP8 quantization."
                )
            ),
            "The compute certificate counts only one correction-bias add per expert; sigmoid, comparison, and quantization arithmetic are excluded.",
        ),
    )
    routed_ready = route
    if blackwell_mxfp8 and not fused_route_quant:
        routed_ready = builder.add_roofline(
            op_id="moe_routed_input_quant_pack",
            name="MXFP8 routed-input quantization and top-k pack",
            category="moe_routing",
            dependencies=(route,),
            flops=0,
            hbm_bytes=(
                tokens
                * config.routed_expert_hidden_size
                * (BF16_BYTES + 1 + 1 / config.mxfp4_group_size)
                + tokens * config.num_experts_per_token * (FP32_BYTES + 4 + 4)
            ),
            flops_formula="0; quantization arithmetic is excluded",
            flops_substitution="0",
            hbm_formula=(
                "tokens × routed width × (BF16 input + FP8 output + 1/group scale) + "
                "tokens × top-k × (FP32 weight + int32 ID + packed int32 bytes)"
            ),
            hbm_substitution=(
                f"{tokens} × {config.routed_expert_hidden_size} × "
                f"({BF16_BYTES} + 1 + 1 / {config.mxfp4_group_size}) + "
                f"{tokens} × {config.num_experts_per_token} × "
                f"({FP32_BYTES} + 4 + 4)"
            ),
            notes=(
                "FlashInfer's standalone path writes FP8 values plus one UE8M0 scale per group of 32.",
                "Top-k pack reads FP32 weights/int32 IDs and writes packed int32 metadata.",
                "Quantization and packing arithmetic are excluded from the compute certificate.",
            ),
        )
    experts = builder.add_roofline(
        op_id="moe_routed_experts",
        name="Top-16 routed MXFP4 experts",
        category="moe_experts",
        dependencies=(routed_ready,),
        flops=expert_flops,
        hbm_bytes=expert_weight_bytes + expert_activation_bytes,
        compute_kind="k3_expert",
        flops_formula=expert_flops_formula,
        flops_substitution=expert_flops_substitution,
        hbm_formula=(
            f"[{unique_formula}] × (parameters per selected expert × MXFP4 bytes/parameter "
            "+ bias elements per expert × bias bytes) + activation/metadata bytes"
        ),
        hbm_substitution=(
            f"[{unique_substitution}] × ({_display_number(params_per_selected_expert)} × "
            f"{_display_number(config.mxfp4_weight_bytes_per_parameter)} + "
            f"(2 × {local_expert_intermediate} + {config.routed_expert_hidden_size}) × "
            f"{expert_bias_element_bytes}) + [{expert_activation_substitution}]"
        ),
        notes=(
            sharding_note,
            occupancy_note,
            "MXFP4 traffic includes one uint8 scale per group of 32 weights.",
            (
                "Blackwell routed GEMM1 activation reads are group-32 MXFP8; later routed intermediates/outputs remain BF16."
                if blackwell_mxfp8
                else "H200 Marlin routed activations remain BF16 (W4A16)."
            ),
            *expert_backend_notes,
            "Synthetic backend expert biases are included at their retained dtype.",
        ),
    )

    if shared_down is None:
        shared_down = add_shared_branch((front_ready,))

    if plain_collective_path:
        # The ordinary path runs shared then routed and performs one collective
        # over the flat pair. H200 always takes it; Blackwell takes it when K3's
        # CustomAllReduceV2 multicast fusion is unavailable/disabled.
        combined_ar = builder.add_collective(
            op_id="moe_combined_all_reduce",
            name="Ordinary combined routed-latent/shared all-reduce",
            category="communication",
            kind="all_reduce",
            logical_bytes=(
                tokens
                * (config.routed_expert_hidden_size + config.hidden_size)
                * BF16_BYTES
            ),
            logical_bytes_formula=(
                "tokens × (routed-latent width + hidden size) × BF16 bytes"
            ),
            logical_bytes_substitution=(
                f"{tokens} × ({config.routed_expert_hidden_size} + "
                f"{config.hidden_size}) × {BF16_BYTES}"
            ),
            dependencies=(experts,),
            notes=(
                "Reduces one concatenated [latent 3584 | shared 7168] tensor.",
                (
                    f"No token all-to-all: H200 EP{hardware.ep_size} evaluates "
                    "only local experts."
                    if hardware.moe_sharding == "ep"
                    else "Blackwell K3 fused all-reduce path is disabled."
                ),
            ),
        )
        shared_ready = combined_ar
        latent_ready = builder.add_roofline(
            op_id="moe_latent_norm",
            name="Latent MoE RMSNorm",
            category="normalization",
            dependencies=(combined_ar,),
            flops=5.0 * tokens * config.routed_expert_hidden_size,
            hbm_bytes=(
                2 * tokens * config.routed_expert_hidden_size * BF16_BYTES
                + config.routed_expert_hidden_size * BF16_BYTES
            ),
            flops_formula="5 × tokens × routed-latent width",
            flops_substitution=(f"5 × {tokens} × {config.routed_expert_hidden_size}"),
            hbm_formula=(
                "2 passes × tokens × routed-latent width × BF16 bytes + "
                "RMSNorm weight width × BF16 bytes"
            ),
            hbm_substitution=(
                f"2 × {tokens} × {config.routed_expert_hidden_size} × {BF16_BYTES} "
                f"+ {config.routed_expert_hidden_size} × {BF16_BYTES}"
            ),
        )
    else:
        shared_ar = builder.add_collective(
            op_id="moe_shared_all_reduce",
            name="Shared-expert TP all-reduce",
            category="communication",
            kind="all_reduce",
            logical_bytes=tokens * config.hidden_size * BF16_BYTES,
            logical_bytes_formula="tokens × hidden size × BF16 bytes",
            logical_bytes_substitution=(
                f"{tokens} × {config.hidden_size} × {BF16_BYTES}"
            ),
            dependencies=(shared_down,),
        )
        latent_bytes = tokens * config.routed_expert_hidden_size * BF16_BYTES
        latent_ready = builder.add_collective(
            op_id="moe_latent_all_reduce_norm",
            name="Fused routed-latent TP all-reduce + RMSNorm",
            category="communication",
            kind="all_reduce",
            logical_bytes=latent_bytes,
            logical_bytes_formula="tokens × routed-latent width × BF16 bytes",
            logical_bytes_substitution=(
                f"{tokens} × {config.routed_expert_hidden_size} × {BF16_BYTES}"
            ),
            dependencies=(experts, shared_ar),
            flops=5.0 * tokens * config.routed_expert_hidden_size,
            hbm_bytes=(
                2 * latent_bytes + config.routed_expert_hidden_size * BF16_BYTES
            ),
            compute_kind="bf16",
            flops_formula="5 × tokens × routed-latent width",
            flops_substitution=(f"5 × {tokens} × {config.routed_expert_hidden_size}"),
            hbm_formula=(
                "2 × logical collective bytes + RMSNorm weight width × BF16 bytes"
            ),
            hbm_substitution=(
                f"2 × {_display_number(latent_bytes)} + "
                f"{config.routed_expert_hidden_size} × {BF16_BYTES}"
            ),
            notes=(
                (
                    "SGLang serializes this after the shared reduction when fused "
                    "semaphores are reused."
                ),
                "RMSNorm is fused into the collective epilogue.",
            ),
        )
        shared_ready = shared_ar

    use_small_token_gemm_ag = (
        not plain_collective_path and hardware.tp_size == 8 and 0 < tokens <= 12
    )
    if use_small_token_gemm_ag:
        local_up_width = config.hidden_size // hardware.tp_size
        return builder.add_collective(
            op_id="moe_gemm_all_gather_tail",
            name="Fused TP8 latent-up GEMM, all-gather, and add3 tail",
            category="communication",
            kind="all_gather",
            logical_bytes=tokens * config.hidden_size * BF16_BYTES,
            logical_bytes_formula="tokens × hidden size × BF16 bytes",
            logical_bytes_substitution=(
                f"{tokens} × {config.hidden_size} × {BF16_BYTES}"
            ),
            dependencies=(latent_ready, shared_ready),
            flops=(
                2 * tokens * config.routed_expert_hidden_size * local_up_width
                + 2 * tokens * config.hidden_size
            ),
            hbm_bytes=(
                config.routed_expert_hidden_size * local_up_width * BF16_BYTES
                + tokens * config.routed_expert_hidden_size * BF16_BYTES
                + 4 * tokens * config.hidden_size * BF16_BYTES
            ),
            compute_kind="bf16",
            flops_formula=(
                "2 × tokens × routed-latent width × TP-local output width + "
                "2 × tokens × hidden size (add3)"
            ),
            flops_substitution=(
                f"2 × {tokens} × {config.routed_expert_hidden_size} × "
                f"{local_up_width} + 2 × {tokens} × {config.hidden_size}"
            ),
            hbm_formula=(
                "routed width × TP-local output width × BF16 weight bytes + "
                "tokens × routed width × BF16 bytes + "
                "4 passes × tokens × hidden size × BF16 bytes"
            ),
            hbm_substitution=(
                f"{config.routed_expert_hidden_size} × {local_up_width} × "
                f"{BF16_BYTES} + {tokens} × {config.routed_expert_hidden_size} × "
                f"{BF16_BYTES} + 4 × {tokens} × {config.hidden_size} × {BF16_BYTES}"
            ),
            notes=(
                "Composite kernel/pipeline; runtime coverage is 1-12 total tokens.",
                "Reads one eighth of the replicated latent-up weight per TP rank.",
            ),
        )

    up = builder.add_gemm(
        op_id="moe_latent_up_replicated",
        name="Replicated latent-up projection",
        category="moe_tail",
        m=tokens,
        k=config.routed_expert_hidden_size,
        n=config.hidden_size,
        weight_bytes_per_parameter=BF16_BYTES,
        dependencies=(latent_ready,),
        notes=(
            (
                "Replicated fallback; the fused TP8 GEMM/all-gather/add3 tail is "
                "used only with K3 AR fusion at 1-12 total tokens."
            ),
        ),
    )

    return builder.add_roofline(
        op_id="moe_tail_add",
        name="Routed, shared, and residual tail add",
        category="elementwise",
        dependencies=(up, shared_ready),
        flops=2.0 * tokens * config.hidden_size,
        hbm_bytes=4 * tokens * config.hidden_size * BF16_BYTES,
        flops_formula="2 adds × tokens × hidden size",
        flops_substitution=f"2 × {tokens} × {config.hidden_size}",
        hbm_formula="4 tensor passes × tokens × hidden size × BF16 bytes",
        hbm_substitution=(f"4 × {tokens} × {config.hidden_size} × {BF16_BYTES}"),
    )


def _decoder_layer(
    *,
    layer: DecoderLayerSpec,
    workload: Workload,
    config: KimiK3TextConfig,
    hardware: HardwareSpec,
    assumptions: EstimatorAssumptions,
    work_ledger: ParallelWorkLedger | None = None,
) -> LayerEstimate:
    builder = _LayerBuilder(
        name=f"decoder_{layer.number:02d}",
        number=layer.number,
        attention=layer.attention,
        ffn=layer.ffn,
        hardware=hardware,
        assumptions=assumptions,
    )
    model_rows = (
        work_ledger.critical_model_rows
        if work_ledger is not None
        else workload.token_count
    )
    reduce_scatter = bool(
        work_ledger is not None
        and layer.ffn == "moe"
    )
    post_attention_rows = (
        work_ledger.critical_source_rows_per_rank
        if reduce_scatter and work_ledger is not None
        else model_rows
    )
    agg1 = _residual_aggregate(
        builder,
        op_id="attn_residual_1",
        dependencies=(),
        token_count=model_rows,
        hidden_size=config.hidden_size,
        previous_blocks=layer.aggregation_1_previous_blocks,
        write_snapshot=layer.attention_residual_write,
    )
    has_pending_prefix = not layer.attention_residual_write
    fuse_attention_prefix = (
        has_pending_prefix
        and (
            (
                hardware.k3_fused_all_reduce_capable
                and assumptions.blackwell_k3_fused_all_reduce
            )
            or (
                reduce_scatter
                and hardware.family in ("b300", "gb300")
                and hardware.sp_moe
            )
        )
    )
    if layer.attention == "kda":
        attention_end = _kda_attention(
            builder,
            root=agg1,
            workload=workload,
            config=config,
            hardware=hardware,
            assumptions=assumptions,
            fuse_pending_prefix=fuse_attention_prefix,
            collective_token_count=model_rows,
            reduce_scatter=reduce_scatter,
        )
    else:
        attention_end = _mla_attention(
            builder,
            root=agg1,
            workload=workload,
            config=config,
            hardware=hardware,
            assumptions=assumptions,
            fuse_pending_prefix=fuse_attention_prefix,
            collective_token_count=model_rows,
            reduce_scatter=reduce_scatter,
        )
    if has_pending_prefix and not fuse_attention_prefix:
        attention_end = builder.add_roofline(
            op_id="attention_pending_prefix_add",
            name="Materialized pending attention-prefix add",
            category="elementwise",
            dependencies=(attention_end,),
            flops=post_attention_rows * config.hidden_size,
            hbm_bytes=(3 * post_attention_rows * config.hidden_size * BF16_BYTES),
            flops_formula="tokens × hidden size",
            flops_substitution=(f"{post_attention_rows} × {config.hidden_size}"),
            hbm_formula="3 passes × tokens × hidden size × BF16 bytes",
            hbm_substitution=(
                f"3 × {post_attention_rows} × {config.hidden_size} × {BF16_BYTES}"
            ),
            notes=(
                "The selected attention collective does not fuse agg1's pending prefix.",
            ),
        )
    agg2 = _residual_aggregate(
        builder,
        op_id="attn_residual_2",
        dependencies=(attention_end,),
        token_count=post_attention_rows,
        hidden_size=config.hidden_size,
        previous_blocks=layer.aggregation_2_previous_blocks,
        write_snapshot=False,
    )
    if layer.ffn == "dense":
        _dense_ffn(
            builder,
            root=agg2,
            workload=workload,
            config=config,
            hardware=hardware,
            work_ledger=work_ledger,
        )
    else:
        if work_ledger is not None:
            _blackwell_a2a_moe_ffn(
                builder,
                root=agg2,
                config=config,
                hardware=hardware,
                work_ledger=work_ledger,
            )
        else:
            _moe_ffn(
                builder,
                root=agg2,
                workload=workload,
                config=config,
                hardware=hardware,
                assumptions=assumptions,
            )
    return builder.finish()


def _embedding_layer(
    *,
    workload: Workload,
    config: KimiK3TextConfig,
    hardware: HardwareSpec,
    assumptions: EstimatorAssumptions,
    work_ledger: ParallelWorkLedger | None = None,
) -> LayerEstimate:
    builder = _LayerBuilder(
        name="embedding",
        number=None,
        attention=None,
        ffn=None,
        hardware=hardware,
        assumptions=assumptions,
    )
    tokens = (
        work_ledger.critical_model_rows
        if work_ledger is not None
        else workload.token_count
    )
    lookup = builder.add_roofline(
        op_id="embedding_lookup",
        name="Vocab-parallel BF16 embedding lookup",
        category="embedding",
        hbm_bytes=(
            tokens * config.hidden_size * BF16_BYTES / hardware.attention_tp_size
            + tokens * config.hidden_size * BF16_BYTES
        ),
        hbm_formula=(
            "tokens × hidden size × BF16 bytes / TP (local hits) + "
            "tokens × hidden size × BF16 bytes (local output)"
        ),
        hbm_substitution=(
            f"{tokens} × {config.hidden_size} × {BF16_BYTES} / "
            f"{hardware.attention_tp_size} "
            f"+ {tokens} × {config.hidden_size} × {BF16_BYTES}"
        ),
        notes=("Assumes token IDs are uniformly distributed across vocab shards.",),
    )
    builder.add_collective(
        op_id="embedding_all_reduce",
        name="Embedding TP all-reduce",
        category="communication",
        kind="all_reduce",
        logical_bytes=tokens * config.hidden_size * BF16_BYTES,
        logical_bytes_formula="tokens × hidden size × BF16 bytes",
        logical_bytes_substitution=(f"{tokens} × {config.hidden_size} × {BF16_BYTES}"),
        group_size=hardware.attention_tp_size,
        dependencies=(lookup,),
    )
    return builder.finish()


def _final_norm_layer(
    *,
    workload: Workload,
    config: KimiK3TextConfig,
    hardware: HardwareSpec,
    assumptions: EstimatorAssumptions,
    work_ledger: ParallelWorkLedger | None = None,
) -> LayerEstimate:
    builder = _LayerBuilder(
        name="final_attention_residual_norm",
        number=None,
        attention=None,
        ffn=None,
        hardware=hardware,
        assumptions=assumptions,
    )
    final_blocks = math.ceil(
        config.num_hidden_layers / config.attention_residual_block_size
    )
    _residual_aggregate(
        builder,
        op_id="final_attn_residual_norm",
        dependencies=(),
        token_count=(
            work_ledger.critical_model_rows
            if work_ledger is not None
            else workload.token_count
        ),
        hidden_size=config.hidden_size,
        previous_blocks=final_blocks,
        write_snapshot=False,
        final_output=True,
    )
    return builder.finish()


def _lm_head_layer(
    *,
    workload: Workload,
    config: KimiK3TextConfig,
    hardware: HardwareSpec,
    assumptions: EstimatorAssumptions,
    work_ledger: ParallelWorkLedger | None = None,
) -> LayerEstimate:
    builder = _LayerBuilder(
        name="lm_head",
        number=None,
        attention=None,
        ffn=None,
        hardware=hardware,
        assumptions=assumptions,
    )
    tokens = workload.logits_token_count
    local_vocab = config.vocab_size // hardware.lm_head_tp_size
    logits = builder.add_gemm(
        op_id="lm_head_gemm",
        name="Vocab-parallel BF16 LM head",
        category="lm_head",
        m=tokens,
        k=config.hidden_size,
        n=local_vocab,
        weight_bytes_per_parameter=BF16_BYTES,
        notes=(
            "Cold prefill computes one sampled-position logit row per request; input logprobs/full logits are out of scope.",
        ),
    )
    scaled_logits = builder.add_roofline(
        op_id="lm_head_logit_scale",
        name="In-place local logit scaling",
        category="lm_head",
        dependencies=(logits,),
        flops=tokens * local_vocab,
        hbm_bytes=2 * tokens * local_vocab * BF16_BYTES,
        flops_formula="logit rows × TP-local vocabulary",
        flops_substitution=f"{tokens} × {local_vocab}",
        hbm_formula=("2 passes × logit rows × TP-local vocabulary × BF16 bytes"),
        hbm_substitution=(f"2 × {tokens} × {local_vocab} × {BF16_BYTES}"),
        notes=(
            "The pinned config omits logit_scale, so Kimi-K3 supplies 1.0; "
            "SGLang still executes logits.mul_(1.0) before TP all-gather.",
        ),
    )
    builder.add_collective(
        op_id="logits_all_gather",
        name="Vocabulary logits TP all-gather",
        category="communication",
        kind="all_gather",
        logical_bytes=tokens * config.vocab_size * BF16_BYTES,
        logical_bytes_formula="logit rows × vocabulary size × BF16 bytes",
        logical_bytes_substitution=(f"{tokens} × {config.vocab_size} × {BF16_BYTES}"),
        group_size=hardware.lm_head_tp_size,
        dependencies=(scaled_logits,),
    )
    return builder.finish()


def _weight_memory(
    config: KimiK3TextConfig, hardware: HardwareSpec
) -> tuple[float, dict[str, float]]:
    h = config.hidden_size
    tp = hardware.tp_size
    attention_tp = hardware.attention_tp_size
    p = config.projection_size
    local_heads = hardware.local_attention_heads
    bf16 = float(BF16_BYTES)
    breakdown: dict[str, float] = {
        "embedding": config.vocab_size * h * bf16 / attention_tp,
        "lm_head": (
            0.0
            if config.tie_word_embeddings
            else config.vocab_size * h * bf16 / hardware.lm_head_tp_size
        ),
        "decoder_norms_and_attention_residual": 0.0,
        "kda_attention": 0.0,
        "mla_attention": 0.0,
        "mla_absorbed_weight_copies": 0.0,
        "dense_ffn": 0.0,
        "moe_router_latent_shared": 0.0,
        "routed_experts_mxfp4": 0.0,
        "routed_expert_synthetic_bias": 0.0,
        "final_norm_and_attention_residual": 3 * h * bf16,
    }
    for layer in config.layers():
        # input/post norms + two attention-residual norms and score vectors.
        breakdown["decoder_norms_and_attention_residual"] += 6 * h * bf16
        if layer.attention == "kda":
            kda_params_bf16 = (
                h * (4 * p // attention_tp)
                + h * local_heads
                + h * config.head_dim
                + config.head_dim * (p // attention_tp)
                + (p // attention_tp) * h
                + config.head_dim
            )
            kda_params_fp32 = (
                3 * p * config.short_conv_kernel_size // attention_tp
                + p // attention_tp
                + local_heads
            )
            breakdown["kda_attention"] += (
                kda_params_bf16 * bf16 + kda_params_fp32 * FP32_BYTES
            )
        else:
            mla_params = (
                h * (config.q_lora_rank + config.kv_lora_rank + config.qk_rope_head_dim)
                + config.q_lora_rank
                + config.kv_lora_rank
                + config.q_lora_rank * local_heads * config.mla_qk_head_dim
                + config.kv_lora_rank
                * local_heads
                * (config.qk_nope_head_dim + config.v_head_dim)
                + local_heads * config.v_head_dim * h
                + h * local_heads * config.v_head_dim
            )
            breakdown["mla_attention"] += mla_params * bf16
            # K3 keeps kv_b_proj and creates persistent contiguous absorbed
            # w_kc/w_vc copies during post-load processing.
            breakdown["mla_absorbed_weight_copies"] += (
                config.kv_lora_rank
                * local_heads
                * (config.qk_nope_head_dim + config.v_head_dim)
                * bf16
            )

        if layer.ffn == "dense":
            breakdown["dense_ffn"] += 3 * h * config.dense_intermediate_size * bf16 / tp
        else:
            dense_moe_bf16_params = (
                config.num_experts * h
                + 2 * h * config.routed_expert_hidden_size
                + 3
                * h
                * config.shared_intermediate_size
                / hardware.shared_expert_tp_size
                + config.routed_expert_hidden_size
            )
            breakdown["moe_router_latent_shared"] += (
                dense_moe_bf16_params * bf16 + config.num_experts * FP32_BYTES
            )
            local_experts = hardware.local_routed_experts
            local_intermediate = (
                config.moe_intermediate_size
                if hardware.moe_sharding == "ep"
                else hardware.routed_expert_intermediate_size_per_partition
            )
            routed_params = (
                3
                * local_experts
                * config.routed_expert_hidden_size
                * local_intermediate
            )
            breakdown["routed_experts_mxfp4"] += (
                routed_params * config.mxfp4_weight_bytes_per_parameter
            )
            synthetic_bias_bytes = (
                FP32_BYTES
                if hardware.family in ("b300", "gb300")
                and hardware.moe_sharding == "tp"
                else BF16_BYTES
            )
            breakdown["routed_expert_synthetic_bias"] += (
                local_experts
                * (2 * local_intermediate + config.routed_expert_hidden_size)
                * synthetic_bias_bytes
            )
    return sum(breakdown.values()), breakdown


def static_weight_bytes_per_rank(
    hardware: HardwareSpec,
    config: KimiK3TextConfig = KIMI_K3_TEXT_CONFIG,
) -> float:
    """Return workload-independent static model bytes resident on one rank."""

    weights, _ = _weight_memory(config, hardware)
    return weights


def _memory_estimate(
    *,
    workload: Workload,
    config: KimiK3TextConfig,
    hardware: HardwareSpec,
    work_ledger: ParallelWorkLedger | None = None,
) -> MemoryEstimate:
    weights, breakdown = _weight_memory(config, hardware)
    local_heads = hardware.local_attention_heads
    kda_state_per_request = (
        len(config.kda_layers)
        * local_heads
        * config.head_dim
        * config.v_head_dim
        * hardware.kda_state_bytes_per_element
    )
    kda_conv_per_request = (
        len(config.kda_layers)
        * 3
        * local_heads
        * config.head_dim
        * (config.short_conv_kernel_size - 1)
        * BF16_BYTES
    )
    kda_state = workload.batch_size * (kda_state_per_request + kda_conv_per_request)
    cache_tokens = (
        workload.token_count
        if workload.phase == "prefill"
        else workload.batch_size * (int(workload.context_length or 0) + 1)
    )
    mla_kv = (
        cache_tokens
        * len(config.full_attention_layers)
        * config.mla_latent_cache_dim
        * hardware.kv_cache_bytes_per_element
    )
    model_and_cache = weights + kda_state + mla_kv
    attention_residual_bank = (
        (
            work_ledger.critical_model_rows
            if work_ledger is not None
            else workload.token_count
        )
        * math.ceil(config.num_hidden_layers / config.attention_residual_block_size)
        * config.hidden_size
        * BF16_BYTES
    )
    total_accounted = model_and_cache + attention_residual_bank
    exceeds_capacity = total_accounted > hardware.nominal_hbm_capacity_bytes_per_gpu
    if exceeds_capacity:
        fits_capacity: bool | None = False
        capacity_status = "accounted_lower_bound_exceeds_nominal_hbm"
    elif hardware.uses_moe_a2a:
        fits_capacity = None
        capacity_status = "inconclusive_megamoe_workspace_excluded"
    else:
        fits_capacity = True
        capacity_status = "accounted_peak_fits_nominal_hbm"
    return MemoryEstimate(
        static_weight_bytes_per_rank=weights,
        kda_state_bytes_per_rank=kda_state,
        mla_kv_cache_bytes_per_rank=mla_kv,
        model_and_cache_bytes_per_rank=model_and_cache,
        attention_residual_bank_bytes_per_rank=attention_residual_bank,
        total_accounted_peak_bytes_per_rank=total_accounted,
        nominal_hbm_capacity_bytes_per_rank=(
            hardware.nominal_hbm_capacity_bytes_per_gpu
        ),
        fits_nominal_capacity=fits_capacity,
        capacity_status=capacity_status,
        weight_breakdown_bytes_per_rank=breakdown,
    )


def _balanced_request_counts(total: int, parts: int) -> tuple[int, ...]:
    base, remainder = divmod(total, parts)
    return tuple(base + (index < remainder) for index in range(parts))


def _captured_batch_size(batch_size: int, hardware: HardwareSpec) -> int:
    if batch_size == 0:
        return 0
    return next(
        bucket
        for bucket in hardware.decode_cuda_graph_batch_sizes
        if bucket >= batch_size
    )


def _blackwell_decode_model_rows(
    *,
    dp_requests: tuple[int, ...],
    hardware: HardwareSpec,
    decode_cuda_graph_replay: bool,
) -> tuple[tuple[int, ...], tuple[int, ...], str]:
    """Mirror SGLang's MLP-sync alignment and DP padding order."""

    aligned_rows = tuple(
        math.ceil(rows / hardware.attention_tp_size)
        * hardware.attention_tp_size
        if rows
        else 0
        for rows in dp_requests
    )
    maximum_rows = max(aligned_rows)
    if decode_cuda_graph_replay:
        captured_rows = _captured_batch_size(maximum_rows, hardware)
        return (
            aligned_rows,
            (captured_rows,) * hardware.attention_dp_size,
            "max_len_cuda_graph",
        )

    # DpPaddingMode.get_dp_padding_mode minimizes DP communication after the
    # mandatory attention-TP alignment. Decode selects MAX_LEN when its
    # communication volume is no larger than SUM_LEN, including equality.
    use_max_len = (
        sum(aligned_rows) * 2
        >= maximum_rows * hardware.attention_dp_size
    )
    if use_max_len:
        return (
            aligned_rows,
            (maximum_rows,) * hardware.attention_dp_size,
            "max_len",
        )
    return aligned_rows, aligned_rows, "sum_len"


def _parallel_work_ledger(
    *,
    workload: Workload,
    hardware: HardwareSpec,
    decode_cuda_graph_replay: bool,
) -> ParallelWorkLedger | None:
    if not hardware.uses_moe_a2a:
        return None

    dp_requests = _balanced_request_counts(
        workload.batch_size, hardware.attention_dp_size
    )
    if workload.phase == "prefill":
        assert workload.sequence_length is not None
        attention_rows = tuple(
            requests * workload.sequence_length for requests in dp_requests
        )
        # require_mlp_sync pads extend rows to an attention-TP multiple. K3
        # trims those rows from attention, but keeps them for residual/MoE work.
        mlp_aligned_rows = tuple(
            math.ceil(rows / hardware.attention_tp_size)
            * hardware.attention_tp_size
            if rows
            else 0
            for rows in attention_rows
        )
        model_rows = mlp_aligned_rows
        dp_padding_mode = "sum_len" if hardware.attention_dp_size > 1 else "max_len"
    else:
        attention_rows = dp_requests
        mlp_aligned_rows, model_rows, dp_padding_mode = (
            _blackwell_decode_model_rows(
                dp_requests=dp_requests,
                hardware=hardware,
                decode_cuda_graph_replay=decode_cuda_graph_replay,
            )
        )

    if any(rows % hardware.attention_tp_size for rows in model_rows):
        raise AssertionError("K3 MLP-sync rows must be attention-TP aligned.")
    source_rows = tuple(
        rows // hardware.attention_tp_size for rows in model_rows
    )
    global_model_rows = sum(model_rows)
    sent_pairs = tuple(
        rows * KIMI_K3_TEXT_CONFIG.num_experts_per_token for rows in source_rows
    )
    routed_pairs = global_model_rows * KIMI_K3_TEXT_CONFIG.num_experts_per_token
    if routed_pairs % hardware.ep_size:
        raise AssertionError("Balanced K3 routed pairs must divide the EP group.")
    balanced_received_pairs = routed_pairs // hardware.ep_size
    if hardware.family == "b300" and hardware.tp_size <= 8:
        topology_contract = (
            "All selected ranks stay inside one eight-GPU NVLink domain; no "
            "scale-out fabric is traversed."
        )
    elif hardware.family == "b300":
        topology_contract = (
            "Eight-GPU NVLink domains plus the configured per-GPU scale-out fabric."
        )
    else:
        topology_contract = (
            "All selected ranks must share one healthy NVL72 L1 NVLink domain."
        )
    return ParallelWorkLedger(
        attention_tp_size=hardware.attention_tp_size,
        attention_dp_size=hardware.attention_dp_size,
        dp_real_requests=dp_requests,
        dp_mlp_aligned_rows=mlp_aligned_rows,
        dp_model_rows=model_rows,
        dp_padding_mode=dp_padding_mode,
        source_rows_per_attention_rank=source_rows,
        critical_attention_rows=max(attention_rows),
        critical_model_rows=max(model_rows),
        global_model_rows=global_model_rows,
        routed_pair_instances=routed_pairs,
        sent_pairs_per_attention_rank_by_dp=sent_pairs,
        critical_sent_pairs_per_source_rank=max(sent_pairs),
        balanced_received_pairs_per_ep_rank=balanced_received_pairs,
        bound_condition_id="balanced_dp_fractional_uniform_ep_routing",
        bound_condition=(
            "Every derived Blackwell certificate and the latency result are "
            "conditional on balanced request assignment across attention-DP "
            "replicas and a fractional ideal-routing relaxation with uniform "
            "EP destinations. This is a scenario assumption, not a claim about "
            "realized per-step routes. The collective DAG assumes the public "
            "default SGLANG_K3_SP_ATTN_RES=0; cross-layer shard carry is outside "
            "this scenario."
        ),
        topology_contract=topology_contract,
        excluded_positive_term_ids=(
            "megamoe_alignment_padding",
            "topk_compute",
            "predispatch_quant_compute",
            "expert_activation_compute",
            "megamoe_internal_hbm_traffic",
            "fp8_scale_transport",
            "megamoe_control_metadata",
            "megamoe_symmetric_buffer_copies",
            "megamoe_transformed_weight_workspace",
            "fabric_contention",
            "collective_startup",
        ),
        notes=(
            "The request batch is global to the engine and is assigned to DP replicas as evenly as whole requests allow.",
            "SGLang MLP-sync first aligns every DP replica to attention TP8, then applies SUM_LEN or MAX_LEN DP padding; decode CUDA graphs use one common MAX_LEN capture shape.",
            "Mandatory MLP-sync alignment makes every modeled MoE forward use the recipe's SP reduce-scatter/all-gather path.",
            "The public default SGLANG_K3_SP_ATTN_RES=0 is fixed for this ledger; enabling cross-layer shard carry changes the DAG.",
            "Received pairs and fabric traffic use the explicit fractional uniform-destination routing scenario; routing locality or skew can move traffic between fabrics and is not classified as a positive excluded cost.",
        ),
    )


def _critical_dp_workload(
    workload: Workload, ledger: ParallelWorkLedger | None
) -> Workload:
    if ledger is None:
        return workload
    critical_requests = max(ledger.dp_real_requests)
    if workload.phase == "prefill":
        return Workload(
            phase="prefill",
            batch_size=critical_requests,
            sequence_length=workload.sequence_length,
        )
    return Workload(
        phase="decode",
        batch_size=critical_requests,
        context_length=workload.context_length,
        execution_batch_size=ledger.critical_model_rows,
    )


def _resolve_execution_workload(
    *,
    workload: Workload,
    hardware: HardwareSpec,
    assumptions: EstimatorAssumptions,
) -> tuple[Workload, bool, ParallelWorkLedger | None]:
    if workload.phase != "decode":
        return (
            workload,
            False,
            _parallel_work_ledger(
                workload=workload,
                hardware=hardware,
                decode_cuda_graph_replay=False,
            ),
        )

    critical_batch_size = math.ceil(
        workload.batch_size / hardware.attention_dp_size
    )
    graph_replay = (
        assumptions.decode_cuda_graph
        and critical_batch_size <= hardware.decode_cuda_graph_max_batch_size
    )
    work_ledger = _parallel_work_ledger(
        workload=workload,
        hardware=hardware,
        decode_cuda_graph_replay=graph_replay,
    )
    execution_batch_size = workload.batch_size
    if work_ledger is not None:
        execution_batch_size = work_ledger.global_model_rows
    elif graph_replay:
        execution_batch_size = _captured_batch_size(
            workload.batch_size, hardware
        )
    return (
        replace(workload, execution_batch_size=execution_batch_size),
        graph_replay,
        work_ledger,
    )


def estimate(
    *,
    hardware: str | HardwareSpec,
    workload: Workload,
    config: KimiK3TextConfig = KIMI_K3_TEXT_CONFIG,
    assumptions: EstimatorAssumptions = _DEFAULT_ESTIMATOR_ASSUMPTIONS,
) -> EstimateResult:
    workload.validate()
    assumptions.validate()
    config.validate()
    if (
        workload.phase == "prefill"
        and workload.sequence_length is not None
        and workload.sequence_length > config.max_position_embeddings
    ):
        raise ValueError(
            f"sequence_length exceeds Kimi-K3's {config.max_position_embeddings} "
            "position limit."
        )
    if (
        workload.phase == "decode"
        and workload.context_length is not None
        and workload.context_length >= config.max_position_embeddings
    ):
        raise ValueError(
            "context_length leaves no position for the next decode token under "
            f"Kimi-K3's {config.max_position_embeddings} position limit."
        )
    resolved_hardware = (
        HARDWARE_PRESETS[hardware] if isinstance(hardware, str) else hardware
    )
    resolved_hardware.validate()
    (
        execution_workload,
        decode_cuda_graph_replay,
        work_ledger,
    ) = _resolve_execution_workload(
        workload=workload,
        hardware=resolved_hardware,
        assumptions=assumptions,
    )
    execution_assumptions = replace(
        assumptions, decode_cuda_graph=decode_cuda_graph_replay
    )
    rank_workload = _critical_dp_workload(execution_workload, work_ledger)
    prefill_rows = (
        work_ledger.critical_attention_rows
        if work_ledger is not None
        else workload.token_count
    )
    if (
        workload.phase == "prefill"
        and prefill_rows > resolved_hardware.prefill_chunk_size
    ):
        raise ValueError(
            f"Cold-prefill critical DP replica has {prefill_rows} tokens, exceeding "
            f"{resolved_hardware.id}'s {resolved_hardware.prefill_chunk_size}-token "
            "single-forward chunk. Multi-chunk cached-prefix/extend MLA is not "
            "modeled yet."
        )

    layers: list[LayerEstimate] = [
        _embedding_layer(
            workload=rank_workload,
            config=config,
            hardware=resolved_hardware,
            assumptions=execution_assumptions,
            work_ledger=work_ledger,
        )
    ]
    layers.extend(
        _decoder_layer(
            layer=layer,
            workload=rank_workload,
            config=config,
            hardware=resolved_hardware,
            assumptions=execution_assumptions,
            work_ledger=work_ledger,
        )
        for layer in config.layers()
    )
    layers.append(
        _final_norm_layer(
            workload=rank_workload,
            config=config,
            hardware=resolved_hardware,
            assumptions=execution_assumptions,
            work_ledger=work_ledger,
        )
    )
    layers.append(
        _lm_head_layer(
            workload=rank_workload,
            config=config,
            hardware=resolved_hardware,
            assumptions=execution_assumptions,
            work_ledger=work_ledger,
        )
    )
    total = sum(layer.latency_seconds for layer in layers)
    memory = _memory_estimate(
        workload=rank_workload,
        config=config,
        hardware=resolved_hardware,
        work_ledger=work_ledger,
    )
    runtime_warnings: tuple[str, ...] = ()
    if resolved_hardware.k3_fused_all_reduce_capable:
        runtime_warnings += (
            (
                "Assumes Blackwell CustomAllReduceV2 multicast is available for "
                "K3 attention/MoE fusion; use the ordinary-fallback flag if it is not."
                if assumptions.blackwell_k3_fused_all_reduce
                else "Blackwell K3 fused all-reduce is disabled; ordinary combined "
                "collectives and materialized prefix adds are modeled."
            ),
            "At <=64 model tokens, the Blackwell path assumes SGLang's fused route+pack+MXFP8 JIT kernel is available; its fallback is the standalone chain.",
        )
    if execution_workload.phase == "decode":
        if decode_cuda_graph_replay:
            runtime_warnings += (
                (
                    "Decode CUDA-graph replay pads requested batch "
                    f"{execution_workload.batch_size} to model batch "
                    f"{execution_workload.model_batch_size}; useful throughput remains "
                    f"based on {execution_workload.batch_size} real output tokens."
                ),
            )
        elif assumptions.decode_cuda_graph:
            runtime_warnings += (
                (
                    "Requested decode batch exceeds the recipe's CUDA-graph maximum "
                    f"{resolved_hardware.decode_cuda_graph_max_batch_size}; eager fallback "
                    "and serialized projection/gate paths are modeled."
                ),
            )
        else:
            runtime_warnings += (
                "Decode CUDA-graph overlap is disabled; projections/gates are serialized.",
            )

    moe_accounting_warning = (
        "Blackwell routed-pair balance is conditional; its HBM certificate uses "
        "a deterministic one-expert floor and excludes expected occupancy."
        if resolved_hardware.uses_moe_a2a
        else "MoE expert-weight traffic uses only the deterministic minimum "
        "number of active experts; routing occupancy above that is excluded."
    )
    warnings = (
        "This is a conditional analytical lower bound, not a latency prediction or benchmark result.",
        f"SGLang recipe status: {resolved_hardware.recipe_status}.",
        "Default compute/HBM/collective efficiencies are 100% and collective startup latency is zero.",
        "No CUDA launch, scheduler, sampling, CPU, network-software, or straggler overhead is included.",
        moe_accounting_warning,
        "HBM-demand certificates assume counted logical reads and writes materialize through HBM; cache residency or fusion changes this scenario, while backend rereads are excluded.",
        "Prefill is constrained to one real SGLang chunk; multi-chunk cached-prefix extend is not modeled.",
        "Accounted peak memory includes persistent absorbed weights, synthetic expert biases, and the eight-row attention-residual bank, but excludes allocator reserve, CUDA-graph pools, workspaces, other live activations, and SGLang's extra KDA state slots.",
        *runtime_warnings,
        *config.known_conflicts,
        *resolved_hardware.warnings,
    )
    return EstimateResult(
        scope=(
            "Kimi-K3 text model only: cold prefill or one non-speculative decode "
            "step; no vision, DCP, PP, DSPARK, HiCache, PD, or serving overhead"
        ),
        model_revision=config.revision,
        hardware=resolved_hardware,
        workload=execution_workload,
        assumptions=assumptions,
        decode_cuda_graph_replay=decode_cuda_graph_replay,
        parallel_work_ledger=work_ledger,
        layers=tuple(layers),
        total_seconds=total,
        memory=memory,
        warnings=warnings,
    )
