"""Auditable lower-bound model for Kimi-K3 text inference.

The implementation is deliberately standard-library-only. Use the repository
launcher ``benchmark/kimi_k3_theoretical.py`` to avoid importing SGLang's
top-level package and its serving dependencies.
"""

from .estimator import (
    CalculationProvenance,
    EstimateResult,
    EstimatorAssumptions,
    Workload,
    estimate,
)
from .specs import (
    CALCULATOR_HARDWARE_FAMILIES,
    CALCULATOR_TP_SIZES,
    HARDWARE_PRESETS,
    KIMI_K3_TEXT_CONFIG,
    HardwareSpec,
    KimiK3TextConfig,
    make_tp_hardware,
)

__all__ = [
    "CALCULATOR_HARDWARE_FAMILIES",
    "CALCULATOR_TP_SIZES",
    "HARDWARE_PRESETS",
    "KIMI_K3_TEXT_CONFIG",
    "CalculationProvenance",
    "EstimateResult",
    "EstimatorAssumptions",
    "HardwareSpec",
    "KimiK3TextConfig",
    "Workload",
    "estimate",
    "make_tp_hardware",
]
