"""Roofline and topology lower-bound estimator for the Kimi-K3 text model.

The estimator counts the operations selected by the scoped SGLang recipes and
then schedules their known dependencies.  It does not claim to predict real
latency: default efficiencies are 100%, collective startup is zero, HBM traffic
is a logical lower bound, and routing is uniform.  Those assumptions make the
result an optimistic bound that can later be calibrated without changing the
operation inventory.
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
CollectiveKind = Literal["all_reduce", "all_gather"]

BF16_BYTES = 2
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

    def to_dict(self) -> dict:
        result = asdict(self)
        result["calculations"] = {
            field: calculation.to_dict()
            for field, calculation in self.calculations.items()
        }
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
    limiting_floor: str

    @property
    def dominant_operation(self) -> OperationEstimate:
        return max(self.operations, key=lambda op: op.duration_seconds)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "number": self.number,
            "attention": self.attention,
            "ffn": self.ffn,
            "dependency_path_seconds": self.dependency_path_seconds,
            "compute_resource_seconds": self.compute_resource_seconds,
            "hbm_resource_seconds": self.hbm_resource_seconds,
            "communication_resource_seconds": self.communication_resource_seconds,
            "latency_seconds": self.latency_seconds,
            "limiting_floor": self.limiting_floor,
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
    fits_nominal_capacity: bool
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
    notes: tuple[str, ...]
    calculations: dict[str, CalculationProvenance]


_CALCULATION_LABELS = {
    "flops_per_rank": "FLOPs per rank",
    "hbm_bytes_per_rank": "HBM bytes per rank",
    "logical_collective_bytes": "Logical collective bytes",
    "link_bytes_per_rank": "Link bytes per rank",
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
        )
        communication_seconds = collective.seconds
        link_bytes = collective.link_bytes_per_rank
        peak = self._peak(compute_kind)
        compute_seconds = (
            flops / (peak * self.assumptions.compute_efficiency) if flops else 0.0
        )
        if hbm_bytes is None:
            # Logical device-memory floor: all-reduce reads and overwrites a
            # full local tensor; all-gather reads a local shard and writes the
            # full gathered tensor. Backend algorithms can move more.
            hbm_bytes = (
                2 * logical_bytes
                if kind == "all_reduce"
                else logical_bytes * (1 + 1 / self.hardware.tp_size)
            )
            hbm_formula = (
                "2 × logical collective bytes"
                if kind == "all_reduce"
                else "logical collective bytes × (1 + 1 / TP)"
            )
            hbm_substitution = (
                f"2 × {_display_number(logical_bytes)}"
                if kind == "all_reduce"
                else (
                    f"{_display_number(logical_bytes)} × "
                    f"(1 + 1 / {_display_number(self.hardware.tp_size)})"
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
                note="Full logical tensor size after the collective, before topology expansion.",
            ),
            "link_bytes_per_rank": _calculation(
                "link_bytes_per_rank",
                formula=collective.link_formula,
                substitution=collective.link_substitution,
                result=link_bytes,
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
        communication_resource = sum(op.communication_seconds for op in result)
        limiting_floor, latency = max(
            (
                ("dependency", dependency_path),
                ("compute_resource", compute_resource),
                ("hbm_resource", hbm_resource),
                ("communication_resource", communication_resource),
            ),
            key=lambda item: item[1],
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
            limiting_floor=limiting_floor,
        )


@dataclass(frozen=True)
class _CollectiveCost:
    seconds: float
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
) -> _CollectiveCost:
    """Return idealized per-rank collective time and link traffic.

    ``logical_bytes`` is the full tensor size after the operation, not a local
    shard size.  NVLink numbers are one-directional; vendor bidirectional
    marketing numbers were halved in the hardware specs.
    """

    p = hardware.tp_size
    efficiency = assumptions.collective_efficiency
    alpha = assumptions.collective_startup_seconds
    if p > hardware.nvlink_domain_size:
        # Explicit two-level lower bound: reduce-scatter/all-gather within each
        # NVLink domain, then exchange corresponding rank shards over the
        # per-GPU scale-out fabric. This is a model, not a claim about the exact
        # NCCL algorithm selected at runtime.
        local_p = hardware.nvlink_domain_size
        domains = p // local_p
        nv_bw = hardware.nvlink_bytes_per_s_per_direction * efficiency
        net_raw = hardware.scaleout_bytes_per_s_per_gpu_per_direction
        assert net_raw is not None
        net_bw = net_raw * efficiency
        if kind == "all_reduce":
            intra_bytes = 2 * (local_p - 1) / local_p * logical_bytes
            inter_bytes = 2 * (domains - 1) / domains * logical_bytes / local_p
            seconds = intra_bytes / nv_bw + inter_bytes / net_bw + alpha
            note = (
                f"Hierarchical {local_p}-GPU NVLink reduce-scatter/all-gather "
                f"plus a {domains}-domain scale-out shard all-reduce."
            )
            link_formula = (
                "2 × (local ranks − 1) / local ranks × logical bytes + "
                "2 × (domains − 1) / domains × logical bytes / local ranks"
            )
            link_substitution = (
                f"2 × ({local_p} − 1) / {local_p} × {_display_number(logical_bytes)} "
                f"+ 2 × ({domains} − 1) / {domains} × "
                f"{_display_number(logical_bytes)} / {local_p}"
            )
        else:
            # Gather corresponding rank shards across domains first, then
            # gather the larger shards within each local NVLink domain.
            inter_bytes = (domains - 1) / domains * logical_bytes / local_p
            intra_bytes = (local_p - 1) / local_p * logical_bytes
            seconds = intra_bytes / nv_bw + inter_bytes / net_bw + alpha
            note = (
                f"Scale-out all-gather across {domains} corresponding rank "
                f"shards followed by a {local_p}-GPU NVLink all-gather."
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
        return _CollectiveCost(
            seconds=seconds,
            link_bytes_per_rank=intra_bytes + inter_bytes,
            note=note,
            link_formula=link_formula,
            link_substitution=link_substitution,
            communication_formula=(
                "intra-domain bytes / (NVLink bytes/s × collective efficiency) + "
                "inter-domain bytes / (scale-out bytes/s × collective efficiency) + startup"
            ),
            communication_substitution=(
                f"{_display_number(intra_bytes)} / "
                f"({_display_number(hardware.nvlink_bytes_per_s_per_direction)} × "
                f"{_display_number(efficiency)}) + {_display_number(inter_bytes)} / "
                f"({_display_number(net_raw)} × {_display_number(efficiency)}) + "
                f"{_display_number(alpha)}"
            ),
        )

    nv_bw = hardware.nvlink_bytes_per_s_per_direction * efficiency
    if kind == "all_reduce":
        link_bytes = 2 * (p - 1) / p * logical_bytes
        note = "Ideal ring all-reduce inside one nonblocking NVLink domain."
        link_formula = "2 × (TP − 1) / TP × logical bytes"
        link_substitution = f"2 × ({p} − 1) / {p} × {_display_number(logical_bytes)}"
    else:
        link_bytes = (p - 1) / p * logical_bytes
        note = "Ideal ring all-gather inside one nonblocking NVLink domain."
        link_formula = "(TP − 1) / TP × logical bytes"
        link_substitution = f"({p} − 1) / {p} × {_display_number(logical_bytes)}"
    return _CollectiveCost(
        seconds=link_bytes / nv_bw + alpha,
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
    if previous_blocks == 0:
        flops = 5.0 * token_count * hidden_size
        hbm_bytes = token_count * hidden_size * 2 * BF16_BYTES
        flops_formula = "5 × tokens × hidden size"
        flops_substitution = f"5 × {token_count} × {hidden_size}"
        hbm_formula = "tokens × hidden size × 2 passes × BF16 bytes"
        hbm_substitution = f"{token_count} × {hidden_size} × 2 × {BF16_BYTES}"
    else:
        rows = previous_blocks + 1
        # Derived logical work: score RMSNorm+dot, short softmax, weighted
        # combination, and output RMSNorm.  This is intentionally not presented
        # as an exact instruction count for the fused TMA/Triton kernels.
        flops = token_count * (
            rows * 7 * hidden_size + rows * 5 + 2 * rows * hidden_size + 5 * hidden_size
        )
        flops_formula = (
            "tokens × (rows × 7 × hidden size + rows × 5 + "
            "2 × rows × hidden size + 5 × hidden size)"
        )
        flops_substitution = (
            f"{token_count} × ({rows} × 7 × {hidden_size} + {rows} × 5 + "
            f"2 × {rows} × {hidden_size} + 5 × {hidden_size})"
        )
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
            "FLOPs and HBM traffic remain derived logical counts, not measured transactions.",
        ),
    )


def _attention_all_reduce(
    builder: _LayerBuilder,
    *,
    dependency: str,
    token_count: int,
    hidden_size: int,
    fuse_pending_prefix: bool,
) -> str:
    logical_bytes = token_count * hidden_size * BF16_BYTES
    return builder.add_collective(
        op_id="attention_all_reduce",
        name=(
            "Attention output TP all-reduce + fused pending-prefix add"
            if fuse_pending_prefix
            else "Attention output TP all-reduce"
        ),
        category="communication",
        kind="all_reduce",
        logical_bytes=logical_bytes,
        logical_bytes_formula="tokens × hidden size × BF16 bytes",
        logical_bytes_substitution=(f"{token_count} × {hidden_size} × {BF16_BYTES}"),
        dependencies=(dependency,),
        flops=token_count * hidden_size if fuse_pending_prefix else 0,
        hbm_bytes=3 * logical_bytes if fuse_pending_prefix else None,
        compute_kind="bf16" if fuse_pending_prefix else "none",
        flops_formula=(
            "tokens × hidden size (fused prefix add)" if fuse_pending_prefix else None
        ),
        flops_substitution=(
            f"{token_count} × {hidden_size}" if fuse_pending_prefix else None
        ),
        hbm_formula=(
            "3 × logical collective bytes (output, prefix, and result)"
            if fuse_pending_prefix
            else None
        ),
        hbm_substitution=(
            f"3 × {_display_number(logical_bytes)}" if fuse_pending_prefix else None
        ),
        notes=(
            ("K3 fused AR consumes the pending BF16 prefix in its epilogue.",)
            if fuse_pending_prefix
            else ()
        ),
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
) -> str:
    tokens = workload.token_count
    tp = hardware.tp_size
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
    if workload.phase == "prefill":
        recurrence_flops = tokens * heads * (4 * k * v + 2 * k * k)
        recurrence_formula = "tokens × local heads × (4 × K × V + 2 × K²)"
        recurrence_substitution = (
            f"{tokens} × {heads} × (4 × {k} × {v} + 2 × {k} × {k})"
        )
        recurrence_note = (
            "KDA prefill FLOPs follow SGLang's benchmark approximation: "
            "4*K*V plus 2*K*K per token/head; inverse/scalar work is omitted."
        )
    else:
        recurrence_flops = tokens * heads * 7 * k * v
        recurrence_formula = "tokens × local heads × 7 × K × V"
        recurrence_substitution = f"{tokens} × {heads} × 7 × {k} × {v}"
        recurrence_note = (
            "KDA decode counts two state-vector products, state decay, and the "
            "rank-1 state update: approximately 7*K*V per token/head."
        )
    activation_bytes = tokens * heads * (3 * k + v) * BF16_BYTES
    gate_bytes = tokens * heads * k * FP32_BYTES
    beta_bytes = tokens * heads * FP32_BYTES
    state_bytes = 2 * state_sequences * heads * k * v * FP32_BYTES
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
                "State traffic assumes one FP32 KDA state read and write per sequence.",
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
                "State traffic assumes one FP32 KDA state read and write per sequence.",
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
        token_count=tokens,
        hidden_size=config.hidden_size,
        fuse_pending_prefix=fuse_pending_prefix,
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
) -> str:
    tokens = workload.token_count
    local_heads = hardware.local_attention_heads
    a_out = config.q_lora_rank + config.kv_lora_rank + config.qk_rope_head_dim
    overlap_limit = hardware.decode_overlap_token_limit
    overlap_output_gate = False

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
        cache_write = tokens * config.mla_latent_cache_dim * BF16_BYTES
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
                f"+ {tokens} × {config.mla_latent_cache_dim} × {BF16_BYTES}"
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
            * BF16_BYTES
            * assumptions.mla_kv_read_amplification
        )
        query_output_bytes = (
            tokens
            * local_heads
            * (config.mla_latent_cache_dim + config.kv_lora_rank)
            * BF16_BYTES
        )
        cache_write = tokens * config.mla_latent_cache_dim * BF16_BYTES
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
                "real batch × context × latent-cache width × BF16 bytes × read "
                "amplification + padded tokens × local heads × (latent-cache width + "
                "kv LoRA rank) × BF16 bytes + padded tokens × latent-cache width × BF16 bytes"
            ),
            hbm_substitution=(
                f"{workload.batch_size} × {workload.context_length} × "
                f"{config.mla_latent_cache_dim} × {BF16_BYTES} × "
                f"{_display_number(assumptions.mla_kv_read_amplification)} + "
                f"{tokens} × {local_heads} × ({config.mla_latent_cache_dim} + "
                f"{config.kv_lora_rank}) × {BF16_BYTES} + {tokens} × "
                f"{config.mla_latent_cache_dim} × {BF16_BYTES}"
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
        token_count=tokens,
        hidden_size=config.hidden_size,
        fuse_pending_prefix=fuse_pending_prefix,
    )


def _dense_ffn(
    builder: _LayerBuilder,
    *,
    root: str,
    workload: Workload,
    config: KimiK3TextConfig,
    hardware: HardwareSpec,
) -> str:
    tokens = workload.token_count
    local_intermediate = config.dense_intermediate_size // hardware.tp_size
    gate_up = builder.add_gemm(
        op_id="dense_gate_up",
        name="Dense SiTU gate/up projection",
        category="dense_ffn",
        m=tokens,
        k=config.hidden_size,
        n=2 * local_intermediate,
        weight_bytes_per_parameter=BF16_BYTES,
        dependencies=(root,),
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
    return builder.add_roofline(
        op_id="dense_prefix_add",
        name="Dense FFN prefix/residual add",
        category="elementwise",
        dependencies=(reduced,),
        flops=tokens * config.hidden_size,
        hbm_bytes=3 * tokens * config.hidden_size * BF16_BYTES,
        flops_formula="tokens × hidden size",
        flops_substitution=f"{tokens} × {config.hidden_size}",
        hbm_formula="3 passes × tokens × hidden size × BF16 bytes",
        hbm_substitution=(f"3 × {tokens} × {config.hidden_size} × {BF16_BYTES}"),
        notes=("KimiK3MLP adds the decoder prefix_sum after down projection.",),
    )


def _expected_unique_experts(
    *, total_experts: int, local_experts: int, tokens: int, top_k: int
) -> float:
    if tokens <= 0:
        return 0.0
    # Each token chooses top_k distinct experts. For any particular expert the
    # probability of omission by one uniform token is 1-top_k/total_experts.
    return local_experts * (1.0 - math.pow(1.0 - top_k / total_experts, tokens))


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
    blackwell_mxfp8 = (
        hardware.family in ("b300", "gb300") and hardware.moe_sharding == "tp"
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
        local_experts = config.num_experts // hardware.ep_size
        local_expert_intermediate = config.moe_intermediate_size
        expert_bias_element_bytes = BF16_BYTES
        params_per_selected_expert = (
            3 * config.routed_expert_hidden_size * config.moe_intermediate_size
        )
        sharding_note = "Uniform routing over EP-local full experts; no token all-to-all is present."
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

    unique = _expected_unique_experts(
        total_experts=config.num_experts,
        local_experts=local_experts,
        tokens=tokens,
        top_k=config.num_experts_per_token,
    )
    unique_formula = "local experts × (1 − (1 − top-k / total experts)^tokens)"
    unique_substitution = (
        f"{local_experts} × (1 − (1 − {config.num_experts_per_token} / "
        f"{config.num_experts})^{tokens})"
    )

    local_compute_instances = tokens * config.num_experts_per_token / shard
    if hardware.moe_sharding == "ep":
        expert_flops = local_compute_instances * (
            6 * config.routed_expert_hidden_size * config.moe_intermediate_size
            + 8 * config.moe_intermediate_size
        )
        expert_flops_formula = (
            "(tokens × top-k / EP) × (6 × routed width × expert intermediate + "
            "8 × expert intermediate)"
        )
        expert_flops_substitution = (
            f"({tokens} × {config.num_experts_per_token} / {shard}) × "
            f"(6 × {config.routed_expert_hidden_size} × {config.moe_intermediate_size} "
            f"+ 8 × {config.moe_intermediate_size})"
        )
    else:
        # Every token/expert pair runs on every TP rank. The selected routed
        # backend pads its local intermediate partition to a multiple of 128.
        expert_flops = (
            tokens
            * config.num_experts_per_token
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
    if hardware.moe_sharding == "ep":
        expert_activation_bytes = (
            local_compute_instances
            * (2 * config.routed_expert_hidden_size + 3 * config.moe_intermediate_size)
            * BF16_BYTES
        )
        expert_activation_formula = (
            "(tokens × top-k / EP) × (2 × routed width + 3 × expert intermediate) "
            "× BF16 bytes"
        )
        expert_activation_substitution = (
            f"({tokens} × {config.num_experts_per_token} / {shard}) × "
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
        marlin_pairs = tokens * config.num_experts_per_token
        expert_activation_bytes = (
            marlin_pairs
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
    route_flops = 6.0 * tokens * config.num_experts
    route_hbm_bytes = (
        tokens * config.num_experts * BF16_BYTES
        + config.num_experts * FP32_BYTES
        + tokens * config.num_experts_per_token * (FP32_BYTES + 4)
    )
    route_flops_formula = "6 × tokens × total experts"
    route_flops_substitution = f"6 × {tokens} × {config.num_experts}"
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
        route_flops += 3.0 * tokens * config.routed_expert_hidden_size
        route_hbm_bytes += tokens * (
            config.routed_expert_hidden_size
            * (BF16_BYTES + 1 + 1 / config.mxfp4_group_size)
            + config.num_experts_per_token * 4
        )
        route_flops_formula += " + 3 × tokens × routed-latent width"
        route_flops_substitution += (
            f" + 3 × {tokens} × {config.routed_expert_hidden_size}"
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
            "Quantization arithmetic is approximated as three operations per latent value.",
            "Comparison cost is only approximated.",
        ),
    )
    routed_ready = route
    if blackwell_mxfp8 and not fused_route_quant:
        routed_ready = builder.add_roofline(
            op_id="moe_routed_input_quant_pack",
            name="MXFP8 routed-input quantization and top-k pack",
            category="moe_routing",
            dependencies=(route,),
            flops=3.0 * tokens * config.routed_expert_hidden_size,
            hbm_bytes=(
                tokens
                * config.routed_expert_hidden_size
                * (BF16_BYTES + 1 + 1 / config.mxfp4_group_size)
                + tokens * config.num_experts_per_token * (FP32_BYTES + 4 + 4)
            ),
            flops_formula="3 × tokens × routed-latent width",
            flops_substitution=(f"3 × {tokens} × {config.routed_expert_hidden_size}"),
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
            f"Uniform-routing occupancy estimate reads {unique:.3f} unique local expert weights.",
            "MXFP4 traffic includes one uint8 scale per group of 32 weights.",
            (
                "Blackwell routed GEMM1 activation reads are group-32 MXFP8; later routed intermediates/outputs remain BF16."
                if blackwell_mxfp8
                else "H200 Marlin routed activations remain BF16 (W4A16)."
            ),
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
                    "No token all-to-all: H200 EP16 evaluates only local experts."
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
) -> LayerEstimate:
    builder = _LayerBuilder(
        name=f"decoder_{layer.number:02d}",
        number=layer.number,
        attention=layer.attention,
        ffn=layer.ffn,
        hardware=hardware,
        assumptions=assumptions,
    )
    agg1 = _residual_aggregate(
        builder,
        op_id="attn_residual_1",
        dependencies=(),
        token_count=workload.token_count,
        hidden_size=config.hidden_size,
        previous_blocks=layer.aggregation_1_previous_blocks,
        write_snapshot=layer.attention_residual_write,
    )
    has_pending_prefix = not layer.attention_residual_write
    fuse_attention_prefix = (
        has_pending_prefix
        and hardware.k3_fused_all_reduce_capable
        and assumptions.blackwell_k3_fused_all_reduce
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
        )
    if has_pending_prefix and not fuse_attention_prefix:
        attention_end = builder.add_roofline(
            op_id="attention_pending_prefix_add",
            name="Materialized pending attention-prefix add",
            category="elementwise",
            dependencies=(attention_end,),
            flops=workload.token_count * config.hidden_size,
            hbm_bytes=(3 * workload.token_count * config.hidden_size * BF16_BYTES),
            flops_formula="tokens × hidden size",
            flops_substitution=(f"{workload.token_count} × {config.hidden_size}"),
            hbm_formula="3 passes × tokens × hidden size × BF16 bytes",
            hbm_substitution=(
                f"3 × {workload.token_count} × {config.hidden_size} × {BF16_BYTES}"
            ),
            notes=(
                "Ordinary attention all-reduce cannot consume agg1's pending prefix.",
            ),
        )
    agg2 = _residual_aggregate(
        builder,
        op_id="attn_residual_2",
        dependencies=(attention_end,),
        token_count=workload.token_count,
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
) -> LayerEstimate:
    builder = _LayerBuilder(
        name="embedding",
        number=None,
        attention=None,
        ffn=None,
        hardware=hardware,
        assumptions=assumptions,
    )
    tokens = workload.token_count
    lookup = builder.add_roofline(
        op_id="embedding_lookup",
        name="Vocab-parallel BF16 embedding lookup",
        category="embedding",
        hbm_bytes=(
            tokens * config.hidden_size * BF16_BYTES / hardware.tp_size
            + tokens * config.hidden_size * BF16_BYTES
        ),
        hbm_formula=(
            "tokens × hidden size × BF16 bytes / TP (local hits) + "
            "tokens × hidden size × BF16 bytes (local output)"
        ),
        hbm_substitution=(
            f"{tokens} × {config.hidden_size} × {BF16_BYTES} / {hardware.tp_size} "
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
        dependencies=(lookup,),
    )
    return builder.finish()


def _final_norm_layer(
    *,
    workload: Workload,
    config: KimiK3TextConfig,
    hardware: HardwareSpec,
    assumptions: EstimatorAssumptions,
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
        token_count=workload.token_count,
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
    local_vocab = config.vocab_size // hardware.tp_size
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
        dependencies=(scaled_logits,),
    )
    return builder.finish()


def _weight_memory(
    config: KimiK3TextConfig, hardware: HardwareSpec
) -> tuple[float, dict[str, float]]:
    h = config.hidden_size
    tp = hardware.tp_size
    p = config.projection_size
    local_heads = hardware.local_attention_heads
    bf16 = float(BF16_BYTES)
    breakdown: dict[str, float] = {
        "embedding": config.vocab_size * h * bf16 / tp,
        "lm_head": (
            0.0 if config.tie_word_embeddings else config.vocab_size * h * bf16 / tp
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
                h * (4 * p // tp)
                + h * local_heads
                + h * config.head_dim
                + config.head_dim * (p // tp)
                + (p // tp) * h
                + config.head_dim
            )
            kda_params_fp32 = (
                3 * p * config.short_conv_kernel_size // tp + p // tp + local_heads
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
                + 3 * h * config.shared_intermediate_size / tp
                + config.routed_expert_hidden_size
            )
            breakdown["moe_router_latent_shared"] += (
                dense_moe_bf16_params * bf16 + config.num_experts * FP32_BYTES
            )
            local_experts = (
                config.num_experts // hardware.ep_size
                if hardware.moe_sharding == "ep"
                else config.num_experts
            )
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


def _memory_estimate(
    *,
    workload: Workload,
    config: KimiK3TextConfig,
    hardware: HardwareSpec,
) -> MemoryEstimate:
    weights, breakdown = _weight_memory(config, hardware)
    local_heads = hardware.local_attention_heads
    kda_state_per_request = (
        len(config.kda_layers)
        * local_heads
        * config.head_dim
        * config.v_head_dim
        * FP32_BYTES
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
        * BF16_BYTES
    )
    model_and_cache = weights + kda_state + mla_kv
    attention_residual_bank = (
        workload.token_count
        * math.ceil(config.num_hidden_layers / config.attention_residual_block_size)
        * config.hidden_size
        * BF16_BYTES
    )
    total_accounted = model_and_cache + attention_residual_bank
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
        fits_nominal_capacity=(
            total_accounted <= hardware.nominal_hbm_capacity_bytes_per_gpu
        ),
        weight_breakdown_bytes_per_rank=breakdown,
    )


def _resolve_execution_workload(
    *,
    workload: Workload,
    hardware: HardwareSpec,
    assumptions: EstimatorAssumptions,
) -> tuple[Workload, bool]:
    if workload.phase != "decode":
        return workload, False

    graph_replay = (
        assumptions.decode_cuda_graph
        and workload.batch_size <= hardware.decode_cuda_graph_max_batch_size
    )
    execution_batch_size = workload.batch_size
    if graph_replay:
        execution_batch_size = next(
            bucket
            for bucket in hardware.decode_cuda_graph_batch_sizes
            if bucket >= workload.batch_size
        )
    return replace(workload, execution_batch_size=execution_batch_size), graph_replay


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
    execution_workload, decode_cuda_graph_replay = _resolve_execution_workload(
        workload=workload,
        hardware=resolved_hardware,
        assumptions=assumptions,
    )
    execution_assumptions = replace(
        assumptions, decode_cuda_graph=decode_cuda_graph_replay
    )
    if (
        workload.phase == "prefill"
        and workload.token_count > resolved_hardware.prefill_chunk_size
    ):
        raise ValueError(
            f"Cold-prefill batch has {workload.token_count} tokens, exceeding "
            f"{resolved_hardware.id}'s {resolved_hardware.prefill_chunk_size}-token "
            "single-forward chunk. Multi-chunk cached-prefix/extend MLA is not "
            "modeled yet."
        )

    layers: list[LayerEstimate] = [
        _embedding_layer(
            workload=execution_workload,
            config=config,
            hardware=resolved_hardware,
            assumptions=execution_assumptions,
        )
    ]
    layers.extend(
        _decoder_layer(
            layer=layer,
            workload=execution_workload,
            config=config,
            hardware=resolved_hardware,
            assumptions=execution_assumptions,
        )
        for layer in config.layers()
    )
    layers.append(
        _final_norm_layer(
            workload=execution_workload,
            config=config,
            hardware=resolved_hardware,
            assumptions=execution_assumptions,
        )
    )
    layers.append(
        _lm_head_layer(
            workload=execution_workload,
            config=config,
            hardware=resolved_hardware,
            assumptions=execution_assumptions,
        )
    )
    total = sum(layer.latency_seconds for layer in layers)
    memory = _memory_estimate(
        workload=execution_workload, config=config, hardware=resolved_hardware
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

    warnings = (
        "This is an optimistic analytical lower bound, not a latency prediction or benchmark result.",
        f"SGLang recipe status: {resolved_hardware.recipe_status}.",
        "Default compute/HBM/collective efficiencies are 100% and collective startup latency is zero.",
        "No CUDA launch, scheduler, sampling, CPU, network-software, or straggler overhead is included.",
        "MoE routing is assumed uniform and perfectly balanced; expert-weight occupancy is an expectation.",
        "Attention HBM traffic is a logical minimum and may undercount backend-specific rereads.",
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
        layers=tuple(layers),
        total_seconds=total,
        memory=memory,
        warnings=warnings,
    )
