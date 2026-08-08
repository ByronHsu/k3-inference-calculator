"""Pinned model facts and hardware/execution assumptions for Kimi-K3.

The distinction between fact, derivation, and assumption is intentional:

* Model dimensions are facts copied from the exact Hugging Face revision used
  by SGLang's B300 CI.
* Dense hardware peaks derived from vendor system totals are marked as such.
* Collective algorithms, zero startup latency, perfect routing balance, and
  full peak utilization are analytical assumptions, not measured facts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal

AttentionKind = Literal["kda", "mla"]
FfnKind = Literal["dense", "moe"]
MoeSharding = Literal["tp", "ep"]
HardwareFamily = Literal["h200", "b300", "gb300"]


@dataclass(frozen=True)
class Source:
    title: str
    url: str
    note: str


@dataclass(frozen=True)
class DecoderLayerSpec:
    """One-based decoder-layer identity from the checkpoint configuration."""

    number: int
    attention: AttentionKind
    ffn: FfnKind
    attention_residual_write: bool
    aggregation_1_previous_blocks: int
    aggregation_2_previous_blocks: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class KimiK3TextConfig:
    revision: str
    hidden_size: int
    vocab_size: int
    tie_word_embeddings: bool
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int
    full_attention_layers: tuple[int, ...]
    kda_layers: tuple[int, ...]
    short_conv_kernel_size: int
    dense_intermediate_size: int
    first_k_dense_replace: int
    moe_intermediate_size: int
    routed_expert_hidden_size: int
    num_experts: int
    num_experts_per_token: int
    num_shared_experts: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    attention_residual_block_size: int
    max_position_embeddings: int
    mxfp4_group_size: int
    sources: tuple[Source, ...]
    known_conflicts: tuple[str, ...]

    @property
    def projection_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def mla_qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def mla_latent_cache_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def shared_intermediate_size(self) -> int:
        # This is how KimiK3MoE constructs the shared MLP.  The value is 6144
        # for the pinned checkpoint because num_shared_experts is 2.
        return self.moe_intermediate_size * self.num_shared_experts

    @property
    def mxfp4_weight_bytes_per_parameter(self) -> float:
        # Packed 4-bit value plus one uint8 scale per group of 32 values.
        return 0.5 + 1.0 / self.mxfp4_group_size

    def layers(self) -> tuple[DecoderLayerSpec, ...]:
        kda = set(self.kda_layers)
        full = set(self.full_attention_layers)
        layers = tuple(
            DecoderLayerSpec(
                number=number,
                attention="kda" if number in kda else "mla",
                ffn=("dense" if number - 1 < self.first_k_dense_replace else "moe"),
                attention_residual_write=(
                    (number - 1) % self.attention_residual_block_size == 0
                ),
                aggregation_1_previous_blocks=(
                    (number - 1 + self.attention_residual_block_size - 1)
                    // self.attention_residual_block_size
                ),
                aggregation_2_previous_blocks=(
                    (number + self.attention_residual_block_size - 1)
                    // self.attention_residual_block_size
                ),
            )
            for number in range(1, self.num_hidden_layers + 1)
        )
        if kda & full:
            raise ValueError("KDA and full-attention layer sets overlap.")
        if kda | full != set(range(1, self.num_hidden_layers + 1)):
            raise ValueError("KDA/full-attention sets do not partition all layers.")
        return layers

    def validate(self) -> None:
        layers = self.layers()
        if len(layers) != 93:
            raise ValueError(f"Expected 93 decoder layers, found {len(layers)}.")
        if sum(layer.attention == "kda" for layer in layers) != 69:
            raise ValueError("Expected 69 KDA layers.")
        if sum(layer.attention == "mla" for layer in layers) != 24:
            raise ValueError("Expected 24 MLA layers.")
        if sum(layer.ffn == "dense" for layer in layers) != 1:
            raise ValueError("Expected exactly one dense decoder layer.")
        if sum(layer.ffn == "moe" for layer in layers) != 92:
            raise ValueError("Expected 92 MoE decoder layers.")

    def to_dict(self, include_layers: bool = True) -> dict:
        result = asdict(self)
        result["projection_size"] = self.projection_size
        result["mla_qk_head_dim"] = self.mla_qk_head_dim
        result["mla_latent_cache_dim"] = self.mla_latent_cache_dim
        result["shared_intermediate_size"] = self.shared_intermediate_size
        result["mxfp4_weight_bytes_per_parameter"] = (
            self.mxfp4_weight_bytes_per_parameter
        )
        if include_layers:
            result["layers"] = [layer.to_dict() for layer in self.layers()]
        return result


_FULL_ATTENTION_LAYERS = tuple(range(4, 93, 4)) + (93,)
_KDA_LAYERS = tuple(
    number for number in range(1, 94) if number not in _FULL_ATTENTION_LAYERS
)

KIMI_K3_TEXT_CONFIG = KimiK3TextConfig(
    revision="9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
    hidden_size=7168,
    vocab_size=163840,
    tie_word_embeddings=False,
    num_hidden_layers=93,
    num_attention_heads=96,
    head_dim=128,
    full_attention_layers=_FULL_ATTENTION_LAYERS,
    kda_layers=_KDA_LAYERS,
    short_conv_kernel_size=4,
    dense_intermediate_size=33792,
    first_k_dense_replace=1,
    moe_intermediate_size=3072,
    routed_expert_hidden_size=3584,
    num_experts=896,
    num_experts_per_token=16,
    num_shared_experts=2,
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    attention_residual_block_size=12,
    max_position_embeddings=1_048_576,
    mxfp4_group_size=32,
    sources=(
        Source(
            title="MoonshotAI Kimi-K3 pinned config.json",
            url=(
                "https://huggingface.co/moonshotai/Kimi-K3/resolve/"
                "9f62e4e9fffbd0a83ddd60e1c209d828994b3569/config.json"
            ),
            note="Exact model revision pinned by SGLang's B300 CI.",
        ),
        Source(
            title="SGLang Kimi-K3 implementation",
            url=(
                "https://github.com/sgl-project/sglang/blob/"
                "ba7b810cc4be74623cf418c5467f29a8a39ac764/"
                "python/sglang/srt/models/kimi_k3.py"
            ),
            note="Defines actual layer construction, fusion, and sharding paths.",
        ),
    ),
    known_conflicts=(
        "The SGLang deployment snippet at commit ba7b810cc says '1 shared' "
        "expert, but the exact CI-pinned checkpoint config says "
        "num_shared_experts=2 and KimiK3MoE multiplies the shared intermediate "
        "width by this value. This analyzer follows the checkpoint config.",
    ),
)
KIMI_K3_TEXT_CONFIG.validate()


@dataclass(frozen=True)
class HardwareSpec:
    id: str
    label: str
    family: HardwareFamily
    gpu: str
    gpu_count: int
    node_count: int
    gpus_per_node: int
    tp_size: int
    ep_size: int
    moe_sharding: MoeSharding
    nominal_hbm_capacity_bytes_per_gpu: int
    hbm_bandwidth_bytes_per_s: float
    dense_bf16_flops_per_s: float
    k3_expert_flops_per_s: float
    nvlink_bytes_per_s_per_direction: float
    nvlink_domain_size: int
    scaleout_bytes_per_s_per_gpu_per_direction: float | None
    moe_backend: str
    mla_prefill_backend: str
    mla_decode_backend: str
    prefill_chunk_size: int
    decode_cuda_graph_max_batch_size: int
    decode_overlap_token_limit: int
    kda_fused_decode_capable: bool
    k3_fused_all_reduce_capable: bool
    recipe_status: str
    sources: tuple[Source, ...]
    derivations: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def local_attention_heads(self) -> int:
        if KIMI_K3_TEXT_CONFIG.num_attention_heads % self.tp_size:
            raise ValueError("Attention heads are not divisible by TP size.")
        return KIMI_K3_TEXT_CONFIG.num_attention_heads // self.tp_size

    @property
    def moe_shard_size(self) -> int:
        return self.ep_size if self.moe_sharding == "ep" else self.tp_size

    @property
    def local_routed_experts(self) -> int:
        if self.moe_sharding == "ep":
            return KIMI_K3_TEXT_CONFIG.num_experts // self.ep_size
        return KIMI_K3_TEXT_CONFIG.num_experts

    @property
    def routed_expert_intermediate_size_per_partition(self) -> int:
        if self.moe_sharding == "ep":
            return KIMI_K3_TEXT_CONFIG.moe_intermediate_size
        exact = KIMI_K3_TEXT_CONFIG.moe_intermediate_size // self.tp_size
        if (
            self.family in ("b300", "gb300")
            or "marlin" in self.moe_backend.lower()
        ):
            # Both FlashInfer's Blackwell path and Marlin W4A16 physically pad
            # each TP-local expert-intermediate partition to 128 elements.
            return ((exact + 127) // 128) * 128
        return exact

    @property
    def decode_cuda_graph_batch_sizes(self) -> tuple[int, ...]:
        """Default non-speculative capture buckets from SGLang ServerArgs."""

        max_bs = self.decode_cuda_graph_max_batch_size
        values = (
            [1, 2, 4, 8, 12]
            + list(range(16, 257, 8))
            + list(range(272, 512, 16))
            + list(range(512, max_bs + 1, 32))
        )
        buckets = [value for value in values if value <= max_bs]
        if max_bs not in buckets:
            buckets.append(max_bs)
        return tuple(sorted(set(buckets)))

    def validate(self) -> None:
        if self.gpu_count != self.node_count * self.gpus_per_node:
            raise ValueError(f"{self.id}: node/GPU topology is inconsistent.")
        if self.tp_size != self.gpu_count:
            raise ValueError(f"{self.id}: scoped presets require TP == GPU count.")
        if KIMI_K3_TEXT_CONFIG.num_attention_heads % self.tp_size:
            raise ValueError(f"{self.id}: TP does not divide 96 attention heads.")
        if (
            self.moe_sharding == "tp"
            and KIMI_K3_TEXT_CONFIG.moe_intermediate_size % self.tp_size
        ):
            raise ValueError(
                f"{self.id}: TP does not divide the routed intermediate width."
            )
        if self.nvlink_domain_size <= 0:
            raise ValueError(f"{self.id}: NVLink domain size must be positive.")
        if self.decode_cuda_graph_max_batch_size <= 0:
            raise ValueError(f"{self.id}: CUDA-graph max batch must be positive.")
        if (
            self.tp_size > self.nvlink_domain_size
            and self.scaleout_bytes_per_s_per_gpu_per_direction is None
        ):
            raise ValueError(
                f"{self.id}: TP crosses NVLink domains without a scale-out fabric."
            )
        if self.moe_sharding == "ep" and KIMI_K3_TEXT_CONFIG.num_experts % self.ep_size:
            raise ValueError(f"{self.id}: EP does not divide 896 routed experts.")

    def to_dict(self) -> dict:
        result = asdict(self)
        result["local_attention_heads"] = self.local_attention_heads
        result["moe_shard_size"] = self.moe_shard_size
        result["local_routed_experts"] = self.local_routed_experts
        result["routed_expert_intermediate_size_per_partition"] = (
            self.routed_expert_intermediate_size_per_partition
        )
        result["decode_cuda_graph_batch_sizes"] = self.decode_cuda_graph_batch_sizes
        return result


_H200_SOURCE = Source(
    title="NVIDIA H200 specifications",
    url="https://www.nvidia.com/en-gb/data-center/h200/",
    note="141 GB HBM3e, 4.8 TB/s, and sparse-marked tensor-core peaks.",
)
_DGX_H200_NETWORK_SOURCE = Source(
    title="NVIDIA DGX H100/H200 hardware overview",
    url=("https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html"),
    note="Eight 400 Gb/s ConnectX-7 cluster cards mapped one-to-one to GPUs.",
)
_B300_SOURCE = Source(
    title="NVIDIA HGX B300 specifications",
    url="https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html",
    note="Eight-GPU HGX topology, 288 GB/GPU, 8 TB/s/GPU, NVLink facts.",
)
_HGX_PEAK_SOURCE = Source(
    title="NVIDIA HGX B300 precision specifications",
    url="https://www.nvidia.com/en-us/data-center/hgx/",
    note="HGX precision totals and sparse/dense footnotes used for per-GPU peaks.",
)
_GB300_SOURCE = Source(
    title="NVIDIA GB300 NVL72 specifications",
    url="https://www.nvidia.com/en-us/data-center/gb300-nvl72/",
    note="72-GPU rack totals used to derive per-GPU dense peaks.",
)
_GB300_TOPOLOGY_SOURCE = Source(
    title="NVIDIA GB300 NVL72 topology",
    url="https://docs.nvidia.com/enterprise-reference-architectures/nvl72-ai-factory/latest/components.html",
    note="Four GPUs per compute tray and one nonblocking 72-GPU NVLink domain.",
)
_CUTLASS_SOURCE = Source(
    title="NVIDIA CUTLASS Blackwell functionality",
    url="https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html",
    note="Mixed MXF8/MXF6/MXF4 MMA throughput class used for W4A8 derivation.",
)


HARDWARE_PRESETS: dict[str, HardwareSpec] = {
    "h200-tpep16": HardwareSpec(
        id="h200-tpep16",
        label="H200 2x8 TP16+EP16",
        family="h200",
        gpu="NVIDIA H200 SXM",
        gpu_count=16,
        node_count=2,
        gpus_per_node=8,
        tp_size=16,
        ep_size=16,
        moe_sharding="ep",
        # Vendor-nameplate capacity, not an exact NVML/CUDA allocatable byte count.
        nominal_hbm_capacity_bytes_per_gpu=141_000_000_000,
        hbm_bandwidth_bytes_per_s=4.8e12,
        # NVIDIA's 1,979 TF/s BF16 figure is explicitly with sparsity.
        dense_bf16_flops_per_s=989.5e12,
        # Marlin W4A16 uses the BF16/FP16 tensor-core compute class.
        k3_expert_flops_per_s=989.5e12,
        # Vendor 900 GB/s NVLink figure is bidirectional.
        nvlink_bytes_per_s_per_direction=450e9,
        nvlink_domain_size=8,
        # Explicit reference-system assumption: one NDR400 HCA per GPU.
        scaleout_bytes_per_s_per_gpu_per_direction=50e9,
        moe_backend="marlin W4A16",
        mla_prefill_backend="FA3/MHA path (cold prefill)",
        mla_decode_backend="FlashMLA absorbed MLA",
        prefill_chunk_size=8192,
        decode_cuda_graph_max_batch_size=512,
        decode_overlap_token_limit=64,
        kda_fused_decode_capable=False,
        k3_fused_all_reduce_capable=False,
        recipe_status="configured; SGLang marks final verification in progress",
        sources=(_H200_SOURCE, _DGX_H200_NETWORK_SOURCE),
        derivations=(
            "Dense BF16 is one half of NVIDIA's sparsity-marked 1,979 TF/s.",
            "Marlin W4A16 is bounded by the dense BF16/FP16 compute class.",
            "The communication model assumes a DGX/HGX-style NDR400 HCA per GPU.",
            "SGLang's H200 memory heuristic selects an 8192-token prefill chunk.",
            "SGLang's default non-speculative decode CUDA-graph maximum is 512 requests.",
        ),
        warnings=(
            "TP16 spans two separate eight-GPU NVLink domains and therefore crosses NDR400.",
            "The branch recipe forces NCCL_MNNVL_ENABLE=1, which requires runtime validation on ordinary two-node H200 systems.",
            "No MoE A2A backend is selected: tokens are replicated, experts are local, and latent/shared outputs are reduced.",
            "The 141 GB HBM value is a vendor-nameplate capacity; runtime-reported and allocatable bytes can differ.",
        ),
    ),
    "b300-tp8": HardwareSpec(
        id="b300-tp8",
        label="HGX B300 1x8 TP8",
        family="b300",
        gpu="NVIDIA B300 SXM",
        gpu_count=8,
        node_count=1,
        gpus_per_node=8,
        tp_size=8,
        ep_size=1,
        moe_sharding="tp",
        # Vendor-nameplate capacity, not an exact NVML/CUDA allocatable byte count.
        nominal_hbm_capacity_bytes_per_gpu=288_000_000_000,
        hbm_bandwidth_bytes_per_s=8e12,
        # HGX total sparse BF16 is 36 PF/s: /8 GPUs /2 sparsity.
        dense_bf16_flops_per_s=2.25e15,
        # W4A8 maps to the dense FP8-class rate, not the W4A4 FP4 peak.
        k3_expert_flops_per_s=4.5e15,
        # Vendor 1.8 TB/s NVLink figure is bidirectional.
        nvlink_bytes_per_s_per_direction=900e9,
        nvlink_domain_size=8,
        scaleout_bytes_per_s_per_gpu_per_direction=None,
        moe_backend="FlashInfer MXFP4 W4A8",
        mla_prefill_backend="TRT-LLM MHA path (cold prefill)",
        mla_decode_backend="TRT-LLM absorbed MLA",
        prefill_chunk_size=16384,
        decode_cuda_graph_max_batch_size=512,
        decode_overlap_token_limit=128,
        kda_fused_decode_capable=True,
        k3_fused_all_reduce_capable=True,
        recipe_status="configured; SGLang marks final verification in progress",
        sources=(_B300_SOURCE, _HGX_PEAK_SOURCE, _CUTLASS_SOURCE),
        derivations=(
            "Dense BF16/FP8 are half of sparsity-marked HGX system totals, divided by eight GPUs.",
            "K3 W4A8 uses the dense FP8-class ceiling; the 13.5 PF/s dense FP4 number is W4A4 and is not used.",
            "SGLang's >=160 GiB memory heuristic selects a 16384-token prefill chunk.",
            "SGLang's default non-speculative decode CUDA-graph maximum is 512 requests.",
        ),
        warnings=(
            "Published peak and bandwidth values are ceilings, not measured K3 kernel throughput.",
            "The 288 GB HBM value is a vendor-nameplate capacity; runtime-reported and allocatable bytes can differ.",
        ),
    ),
    "gb300-tp8": HardwareSpec(
        id="gb300-tp8",
        label="GB300 NVL72 2x4 TP8",
        family="gb300",
        gpu="NVIDIA Blackwell Ultra GPU in GB300 NVL72",
        gpu_count=8,
        node_count=2,
        gpus_per_node=4,
        tp_size=8,
        ep_size=1,
        moe_sharding="tp",
        # Vendor-nameplate capacity, not an exact NVML/CUDA allocatable byte count.
        nominal_hbm_capacity_bytes_per_gpu=288_000_000_000,
        hbm_bandwidth_bytes_per_s=8e12,
        # NVL72 total sparse BF16 is 360 PF/s: /72 GPUs /2 sparsity.
        dense_bf16_flops_per_s=2.5e15,
        # NVL72 total sparse FP8 is 720 PF/s: /72 /2 sparsity.
        k3_expert_flops_per_s=5e15,
        nvlink_bytes_per_s_per_direction=900e9,
        nvlink_domain_size=72,
        scaleout_bytes_per_s_per_gpu_per_direction=None,
        moe_backend="FlashInfer MXFP4 W4A8",
        mla_prefill_backend="TRT-LLM MHA path (cold prefill)",
        mla_decode_backend="TRT-LLM absorbed MLA",
        prefill_chunk_size=16384,
        decode_cuda_graph_max_batch_size=512,
        decode_overlap_token_limit=128,
        kda_fused_decode_capable=True,
        k3_fused_all_reduce_capable=True,
        recipe_status="configured; SGLang marks final verification in progress",
        sources=(_GB300_SOURCE, _GB300_TOPOLOGY_SOURCE, _CUTLASS_SOURCE),
        derivations=(
            "Dense BF16/FP8 are half of sparsity-marked NVL72 totals, divided by 72 GPUs.",
            "The two four-GPU compute trays are assumed to share one healthy NVL72 L1 NVLink domain.",
            "K3 W4A8 uses the dense FP8-class ceiling; the 15 PF/s dense FP4 number is W4A4 and is not used.",
            "SGLang's >=160 GiB memory heuristic selects a 16384-token prefill chunk.",
            "SGLang's default non-speculative decode CUDA-graph maximum is 512 requests.",
        ),
        warnings=(
            "If the two trays are not in the same NVL72 L1 domain, this preset's communication model is invalid.",
            "Published peak and bandwidth values are ceilings, not measured K3 kernel throughput.",
            "The 288 GB HBM value is a vendor-nameplate capacity; runtime-reported and allocatable bytes can differ.",
        ),
    ),
}

HARDWARE_PRESETS["h200-tpep32"] = replace(
    HARDWARE_PRESETS["h200-tpep16"],
    id="h200-tpep32",
    label="H200 4x8 TP32+EP32",
    gpu_count=32,
    node_count=4,
    tp_size=32,
    ep_size=32,
    recipe_status=(
        "configured from SGLang's H200 high-throughput TP32+EP32 recipe; "
        "final verification in progress"
    ),
    derivations=(
        *HARDWARE_PRESETS["h200-tpep16"].derivations,
        "SGLang's H200 high-throughput recipe widens to TP32+EP32 over four nodes.",
    ),
    warnings=(
        "TP32 spans four separate eight-GPU NVLink domains and therefore "
        "crosses NDR400.",
        "The branch recipe forces NCCL_MNNVL_ENABLE=1, which requires runtime "
        "validation on ordinary four-node H200 systems.",
        "No MoE A2A backend is selected: tokens are replicated, experts are "
        "local, and latent/shared outputs are reduced.",
        "The 141 GB HBM value is a vendor-nameplate capacity; runtime-reported "
        "and allocatable bytes can differ.",
    ),
)

for _hardware in HARDWARE_PRESETS.values():
    _hardware.validate()


CALCULATOR_HARDWARE_FAMILIES = ("h200", "b300", "gb300")
CALCULATOR_TP_SIZES = (8, 16, 32, 64)


def make_tp_hardware(family: str, tp_size: int) -> HardwareSpec:
    """Build one calculator-only, TP-only hardware topology.

    These variants intentionally do not claim to be deployment recipes.  H200
    and B300 groups larger than one eight-GPU NVLink domain use the per-GPU,
    one-direction scale-out bandwidths requested by the calculator: 50 GB/s
    and 100 GB/s respectively.  GB300 groups are placed wholly inside one
    healthy NVL72 L1 domain.
    """

    if family not in CALCULATOR_HARDWARE_FAMILIES:
        valid = ", ".join(CALCULATOR_HARDWARE_FAMILIES)
        raise ValueError(f"Unknown hardware family {family!r}; choose from {valid}.")
    if tp_size not in CALCULATOR_TP_SIZES:
        valid = ", ".join(str(size) for size in CALCULATOR_TP_SIZES)
        raise ValueError(f"TP size must be one of: {valid}.")
    if KIMI_K3_TEXT_CONFIG.num_attention_heads % tp_size:
        raise ValueError(
            f"TP{tp_size} is not supported by TP-only Kimi-K3: its 96 KDA "
            "attention heads must divide evenly across the attention-TP group."
        )

    template_id = {
        "h200": "h200-tpep16",
        "b300": "b300-tp8",
        "gb300": "gb300-tp8",
    }[family]
    template = HARDWARE_PRESETS[template_id]
    gpus_per_node = 4 if family == "gb300" else 8
    nodes = tp_size // gpus_per_node
    label_prefix = {
        "h200": "H200",
        "b300": "B300",
        "gb300": "GB300 NVL72",
    }[family]
    scaleout = {"h200": 50e9, "b300": 100e9, "gb300": None}[family]
    if family == "gb300":
        topology_derivation = (
            f"TP{tp_size} is assumed to occupy {nodes} four-GPU compute trays "
            "inside one healthy, nonblocking 72-GPU NVLink domain."
        )
        topology_warnings = (
            "This lower bound is invalid if all selected GPUs are not in the same NVL72 L1 NVLink domain.",
        )
    else:
        ib_gbps = 50 if family == "h200" else 100
        topology_derivation = (
            f"TP{tp_size} uses {nodes} eight-GPU NVLink domain(s); groups larger "
            f"than eight assume {ib_gbps} GB/s per GPU, one direction, for scale-out IB."
        )
        topology_warnings = (
            f"The {ib_gbps} GB/s per-GPU, one-direction IB value is a calculator scenario assumption, not a measured K3 collective result.",
        )

    spec = replace(
        template,
        id=f"{family}-tp{tp_size}",
        label=f"{label_prefix} TP{tp_size}",
        gpu_count=tp_size,
        node_count=nodes,
        gpus_per_node=gpus_per_node,
        tp_size=tp_size,
        ep_size=1,
        moe_sharding="tp",
        scaleout_bytes_per_s_per_gpu_per_direction=scaleout,
        kda_fused_decode_capable=(tp_size == 8),
        # The current K3 multicast/custom-AR and GEMM-AR kernels are TP8-only.
        k3_fused_all_reduce_capable=(family in ("b300", "gb300") and tp_size == 8),
        recipe_status="calculator TP-only topology; not a verified SGLang deployment recipe",
        derivations=(*template.derivations, topology_derivation),
        warnings=(
            "Published peak and bandwidth values are ceilings, not measured K3 kernel throughput.",
            *topology_warnings,
            "TP-only modeling uses no expert parallelism or token all-to-all.",
            "Runtime support and kernel selection for this exact TP size have not been benchmark-verified.",
        ),
    )
    spec.validate()
    return spec


def make_calculator_hardware(family: str, tp_size: int) -> HardwareSpec:
    """Returns the recipe-backed topology selected by the public calculator."""
    h200_ep_recipes = {
        16: "h200-tpep16",
        32: "h200-tpep32",
    }
    preset_id = h200_ep_recipes.get(tp_size) if family == "h200" else None
    if preset_id is not None:
        return HARDWARE_PRESETS[preset_id]
    return make_tp_hardware(family, tp_size)
