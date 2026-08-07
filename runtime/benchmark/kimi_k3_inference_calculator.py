#!/usr/bin/env python3
"""Local web UI and JSON API for the Kimi-K3 inference calculator.

The launcher is standard-library-only and deliberately loads the theoretical
analyzer without importing SGLang's top-level runtime dependency chain.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import math
import mimetypes
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence
from urllib.parse import unquote, urlsplit

_PRIVATE_PACKAGE_NAME = "_sglang_kimi_k3_calculator_analyzer"
_ANALYTICAL_STATUS = "optimistic lower bound; not measured"
_MAX_REQUEST_BYTES = 1 << 20
_MAX_BATCH_SIZE = 4096
STATIC_DIR = Path(__file__).resolve().with_suffix("")

_REQUEST_FIELDS = frozenset(
    {
        "phase",
        "hardware",
        "tp_size",
        "sequence_length",
        "batch_size",
        "context_length",
        "compute_efficiency",
        "hbm_efficiency",
        "collective_efficiency",
        "collective_startup_us",
        "mla_kv_read_amplification",
        "decode_cuda_graph",
        "blackwell_k3_fused_all_reduce",
    }
)


class ApiError(ValueError):
    """A client-visible request error."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message

    def to_dict(self) -> dict[str, dict[str, str]]:
        return {"error": {"type": self.error_type, "message": self.message}}


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"Non-standard JSON constant: {token}")


def _load_analyzer() -> ModuleType:
    existing = sys.modules.get(_PRIVATE_PACKAGE_NAME)
    if existing is not None:
        return existing

    package_dir = (
        Path(__file__).resolve().parents[1]
        / "python"
        / "sglang"
        / "benchmark"
        / "kimi_k3_theoretical"
    )
    spec = importlib.util.spec_from_file_location(
        _PRIVATE_PACKAGE_NAME,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load analyzer package from {package_dir}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PRIVATE_PACKAGE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _require_object(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ApiError("invalid_request", "Request JSON must be an object.")
    unknown = sorted(set(value) - _REQUEST_FIELDS)
    if unknown:
        raise ApiError(
            "invalid_request",
            "Unknown request field(s): " + ", ".join(unknown) + ".",
        )
    return value


def _positive_integer(
    payload: dict[str, Any], name: str, *, maximum: int | None = None
) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ApiError("invalid_request", f"{name} must be a positive integer.")
    if maximum is not None and value > maximum:
        raise ApiError(
            "invalid_request", f"{name} must be less than or equal to {maximum}."
        )
    return value


def _require_finite_numbers(value: object) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ApiError(
                "invalid_request",
                "Inputs produced a non-finite estimate; reduce the workload or assumptions.",
            )
        return
    if isinstance(value, dict):
        for item in value.values():
            _require_finite_numbers(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _require_finite_numbers(item)


def _number(
    payload: dict[str, Any],
    name: str,
    default: float,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    value = payload.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ApiError("invalid_request", f"{name} must be a number.")
    try:
        result = float(value)
    except OverflowError as error:
        raise ApiError("invalid_request", f"{name} must be finite.") from error
    if not math.isfinite(result):
        raise ApiError("invalid_request", f"{name} must be finite.")
    if result < minimum or (maximum is not None and result > maximum):
        interval = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ApiError("invalid_request", f"{name} must be {interval}.")
    return result


def _efficiency(payload: dict[str, Any], name: str) -> float:
    value = _number(payload, name, 1.0, minimum=0.0, maximum=1.0)
    if value == 0:
        raise ApiError("invalid_request", f"{name} must be greater than 0.")
    return value


def _boolean(payload: dict[str, Any], name: str, default: bool) -> bool:
    value = payload.get(name, default)
    if not isinstance(value, bool):
        raise ApiError("invalid_request", f"{name} must be a boolean.")
    return value


def _tp_size(payload: dict[str, Any], analyzer: ModuleType) -> int:
    value = payload.get("tp_size", 8)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ApiError("invalid_request", "tp_size must be an integer.")
    if value not in analyzer.CALCULATOR_TP_SIZES:
        valid = ", ".join(str(size) for size in analyzer.CALCULATOR_TP_SIZES)
        raise ApiError("invalid_request", f"tp_size must be one of: {valid}.")
    return value


def _hardware_specs(
    payload: dict[str, Any], analyzer: ModuleType
) -> tuple[object, ...]:
    value = payload.get("hardware", ["all"])
    if not isinstance(value, list) or not value:
        raise ApiError(
            "invalid_request",
            "hardware must be a non-empty list of hardware families, or ['all'].",
        )
    if any(not isinstance(item, str) for item in value):
        raise ApiError("invalid_request", "Every hardware entry must be a string.")
    if "all" in value:
        if value != ["all"]:
            raise ApiError(
                "invalid_request", "'all' cannot be combined with hardware preset IDs."
            )
        value = list(analyzer.CALCULATOR_HARDWARE_FAMILIES)

    requested = list(dict.fromkeys(value))
    families = set(analyzer.CALCULATOR_HARDWARE_FAMILIES)
    legacy = set(analyzer.HARDWARE_PRESETS)
    if all(item in families for item in requested):
        tp_size = _tp_size(payload, analyzer)
        try:
            return tuple(
                analyzer.make_tp_hardware(family, tp_size) for family in requested
            )
        except ValueError as error:
            raise ApiError("invalid_request", str(error)) from error

    # Keep the original fixed-preset request form available to scripts that
    # predate the calculator's independent hardware/TP controls.
    if all(item in legacy for item in requested):
        if "tp_size" in payload:
            raise ApiError(
                "invalid_request",
                "tp_size cannot be combined with legacy fixed hardware preset IDs.",
            )
        return tuple(analyzer.HARDWARE_PRESETS[item] for item in requested)

    valid = ", ".join(analyzer.CALCULATOR_HARDWARE_FAMILIES)
    raise ApiError(
        "invalid_request",
        f"Unknown hardware preset or mixed family selection; choose families from: {valid}.",
    )


def _normalized_request(
    value: object, analyzer: ModuleType
) -> tuple[object, object, tuple[str, ...]]:
    payload = _require_object(value)
    phase = payload.get("phase")
    if phase not in ("prefill", "decode"):
        raise ApiError("invalid_request", "phase must be 'prefill' or 'decode'.")

    max_positions = analyzer.KIMI_K3_TEXT_CONFIG.max_position_embeddings
    batch_size = _positive_integer(payload, "batch_size", maximum=_MAX_BATCH_SIZE)
    if payload.get("sequence_length") is not None:
        _positive_integer(payload, "sequence_length", maximum=max_positions)
    if payload.get("context_length") is not None:
        _positive_integer(payload, "context_length", maximum=max_positions - 1)
    if phase == "prefill":
        workload = analyzer.Workload(
            phase="prefill",
            batch_size=batch_size,
            sequence_length=_positive_integer(
                payload, "sequence_length", maximum=max_positions
            ),
        )
    else:
        workload = analyzer.Workload(
            phase="decode",
            batch_size=batch_size,
            context_length=_positive_integer(
                payload, "context_length", maximum=max_positions - 1
            ),
        )

    assumptions = analyzer.EstimatorAssumptions(
        compute_efficiency=_efficiency(payload, "compute_efficiency"),
        hbm_efficiency=_efficiency(payload, "hbm_efficiency"),
        collective_efficiency=_efficiency(payload, "collective_efficiency"),
        collective_startup_seconds=(
            _number(
                payload,
                "collective_startup_us",
                0.0,
                minimum=0.0,
            )
            * 1e-6
        ),
        mla_kv_read_amplification=_number(
            payload,
            "mla_kv_read_amplification",
            1.0,
            minimum=1.0,
        ),
        decode_cuda_graph=_boolean(payload, "decode_cuda_graph", True),
        blackwell_k3_fused_all_reduce=_boolean(
            payload, "blackwell_k3_fused_all_reduce", True
        ),
    )
    hardware_specs = _hardware_specs(payload, analyzer)
    return workload, assumptions, hardware_specs


def _calculator_model(analyzer: ModuleType) -> dict[str, Any]:
    """Return calculator model facts without the intentionally hidden conflict."""

    model = analyzer.KIMI_K3_TEXT_CONFIG.to_dict(include_layers=True)
    model.pop("known_conflicts", None)
    return model


def _calculator_result(result: object, analyzer: ModuleType) -> dict[str, Any]:
    payload = result.to_dict()
    hidden = set(analyzer.KIMI_K3_TEXT_CONFIG.known_conflicts)
    payload["warnings"] = [
        warning for warning in payload.get("warnings", ()) if warning not in hidden
    ]
    return payload


def manifest_payload(analyzer: ModuleType | None = None) -> dict[str, Any]:
    """Return model facts, supported presets, and the UI request defaults."""

    analyzer = analyzer or _load_analyzer()
    return {
        "schema_version": 1,
        "analytical_status": _ANALYTICAL_STATUS,
        "model": _calculator_model(analyzer),
        "hardware_presets": {
            key: value.to_dict() for key, value in analyzer.HARDWARE_PRESETS.items()
        },
        "hardware_families": {
            family: {
                "id": family,
                "label": {
                    "h200": "H200",
                    "b300": "B300",
                    "gb300": "GB300",
                }[family],
                "gpu": analyzer.make_tp_hardware(family, 8).gpu,
                "gpus_per_node": analyzer.make_tp_hardware(family, 8).gpus_per_node,
                "nvlink_domain_size": analyzer.make_tp_hardware(
                    family, 8
                ).nvlink_domain_size,
                "nvlink_bytes_per_s_per_direction": analyzer.make_tp_hardware(
                    family, 8
                ).nvlink_bytes_per_s_per_direction,
                "scaleout_bytes_per_s_per_gpu_per_direction": analyzer.make_tp_hardware(
                    family, 8
                ).scaleout_bytes_per_s_per_gpu_per_direction,
                "prefill_chunk_size": analyzer.make_tp_hardware(
                    family, 8
                ).prefill_chunk_size,
                "decode_cuda_graph_max_batch_size": analyzer.make_tp_hardware(
                    family, 8
                ).decode_cuda_graph_max_batch_size,
            }
            for family in analyzer.CALCULATOR_HARDWARE_FAMILIES
        },
        "tp_sizes": list(analyzer.CALCULATOR_TP_SIZES),
        "invalid_combinations": [
            {
                "hardware": list(analyzer.CALCULATOR_HARDWARE_FAMILIES),
                "tp_size": 64,
                "reason": (
                    "TP64 is not supported by TP-only Kimi-K3 because 96 KDA "
                    "attention heads do not divide evenly across 64 ranks."
                ),
            }
        ],
        "defaults": {
            "phase": "prefill",
            "hardware": ["all"],
            "tp_size": 8,
            "sequence_length": 4096,
            "batch_size": 1,
            "context_length": 4096,
            "compute_efficiency": 1.0,
            "hbm_efficiency": 1.0,
            "collective_efficiency": 1.0,
            "collective_startup_us": 0.0,
            "mla_kv_read_amplification": 1.0,
            "decode_cuda_graph": True,
            "blackwell_k3_fused_all_reduce": True,
        },
    }


def calculate_payload(
    value: object, analyzer: ModuleType | None = None
) -> dict[str, Any]:
    """Validate one calculator request and run the audited estimator."""

    analyzer = analyzer or _load_analyzer()
    workload, assumptions, hardware_specs = _normalized_request(value, analyzer)
    workload.validate()
    assumptions.validate()
    if workload.phase == "prefill":
        for hardware in hardware_specs:
            if workload.token_count > hardware.prefill_chunk_size:
                raise ApiError(
                    "invalid_request",
                    f"Cold-prefill batch has {workload.token_count} tokens, exceeding "
                    f"{hardware.id}'s {hardware.prefill_chunk_size}-token "
                    "single-forward chunk. Multi-chunk cached-prefix/extend MLA is "
                    "not modeled yet.",
                )
    results = [
        _calculator_result(
            analyzer.estimate(
                hardware=hardware,
                workload=workload,
                assumptions=assumptions,
            ),
            analyzer,
        )
        for hardware in hardware_specs
    ]
    response = {
        "schema_version": 1,
        "analytical_status": _ANALYTICAL_STATUS,
        "model": _calculator_model(analyzer),
        "results": results,
    }
    _require_finite_numbers(response)
    return response


def _static_file(path: str, static_root: Path) -> Path | None:
    """Resolve a URL path below static_root without permitting traversal."""

    decoded = unquote(path)
    if "\x00" in decoded or "\\" in decoded:
        return None
    relative = decoded.lstrip("/") or "index.html"
    candidate = (static_root / relative).resolve()
    root = static_root.resolve()
    if not candidate.is_relative_to(root):
        return None
    if candidate.is_dir():
        candidate = (candidate / "index.html").resolve()
        if not candidate.is_relative_to(root):
            return None
    return candidate if candidate.is_file() else None


class KimiK3CalculatorHandler(BaseHTTPRequestHandler):
    """Serve the calculator's static UI and standard-library JSON API."""

    static_root = STATIC_DIR
    analyzer: ModuleType | None = None

    def _json_response(
        self, status: int, payload: object, *, head: bool = False
    ) -> None:
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
        accepts_gzip = "gzip" in self.headers.get("Accept-Encoding", "").lower()
        compressed = accepts_gzip and len(body) >= 1024
        if compressed:
            body = gzip.compress(body, compresslevel=1)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Vary", "Accept-Encoding")
        if compressed:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        if not head:
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def _error(
        self, status: int, error_type: str, message: str, *, head: bool = False
    ) -> None:
        self._json_response(status, ApiError(error_type, message).to_dict(), head=head)

    def _serve_static(self, *, head: bool = False) -> None:
        path = _static_file(urlsplit(self.path).path, self.static_root)
        if path is None:
            self._error(
                HTTPStatus.NOT_FOUND,
                "not_found",
                "Resource not found.",
                head=head,
            )
            return
        body = path.read_bytes()
        content_type, _ = mimetypes.guess_type(path.name)
        self.send_response(HTTPStatus.OK)
        charset = "; charset=utf-8" if (content_type or "").startswith("text/") else ""
        self.send_header(
            "Content-Type", (content_type or "application/octet-stream") + charset
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if not head:
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/api/manifest":
            self._json_response(HTTPStatus.OK, manifest_payload(self.analyzer))
            return
        if path.startswith("/api/"):
            self._error(
                HTTPStatus.NOT_FOUND,
                "not_found",
                "API route not found.",
            )
            return
        self._serve_static()

    def do_HEAD(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/api/manifest":
            self._json_response(
                HTTPStatus.OK, manifest_payload(self.analyzer), head=True
            )
            return
        if path.startswith("/api/"):
            self._error(
                HTTPStatus.NOT_FOUND,
                "not_found",
                "API route not found.",
                head=True,
            )
            return
        self._serve_static(head=True)

    def do_POST(self) -> None:  # noqa: N802
        if urlsplit(self.path).path != "/api/calculate":
            self._error(HTTPStatus.NOT_FOUND, "not_found", "API route not found.")
            return
        content_type = self.headers.get("Content-Type", "")
        if content_type.split(";", 1)[0].strip().lower() != "application/json":
            self._error(
                HTTPStatus.BAD_REQUEST,
                "invalid_json",
                "Content-Type must be application/json.",
            )
            return
        try:
            content_length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "invalid_json",
                "A valid Content-Length header is required.",
            )
            return
        if not 0 < content_length <= _MAX_REQUEST_BYTES:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "invalid_json",
                f"JSON body must be between 1 and {_MAX_REQUEST_BYTES} bytes.",
            )
            return
        try:
            value = json.loads(
                self.rfile.read(content_length).decode("utf-8"),
                parse_constant=_reject_json_constant,
            )
        except (RecursionError, UnicodeDecodeError, ValueError):
            self._error(
                HTTPStatus.BAD_REQUEST,
                "invalid_json",
                "Request body is not valid JSON.",
            )
            return
        try:
            response = calculate_payload(value, self.analyzer)
        except ApiError as error:
            self._json_response(HTTPStatus.BAD_REQUEST, error.to_dict())
            return
        except Exception as error:  # pragma: no cover - defensive HTTP boundary
            self.log_error("calculator failure: %s", error)
            self._error(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "internal_error",
                "The calculator failed while evaluating this request.",
            )
            return
        self._json_response(HTTPStatus.OK, response)


def make_handler(
    *, static_root: Path = STATIC_DIR, analyzer: ModuleType | None = None
) -> type[KimiK3CalculatorHandler]:
    """Create an isolated handler class for a server or test instance."""

    class ConfiguredHandler(KimiK3CalculatorHandler):
        pass

    ConfiguredHandler.static_root = static_root
    ConfiguredHandler.analyzer = analyzer
    return ConfiguredHandler


def create_server(
    host: str = "127.0.0.1", port: int = 8000, *, static_root: Path = STATIC_DIR
) -> ThreadingHTTPServer:
    if not static_root.is_dir():
        raise FileNotFoundError(f"Calculator assets not found at {static_root}.")
    # Load once before accepting concurrent requests; sys.modules exposes a
    # module before exec_module completes, so lazy first-loads can otherwise race.
    analyzer = _load_analyzer()
    server = ThreadingHTTPServer(
        (host, port), make_handler(static_root=static_root, analyzer=analyzer)
    )
    server.daemon_threads = True
    return server


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve the local Kimi-K3 inference calculator UI."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not 0 <= args.port <= 65535:
        raise SystemExit("--port must be between 0 and 65535")
    try:
        server = create_server(args.host, args.port)
    except (FileNotFoundError, OSError) as error:
        raise SystemExit(str(error)) from error

    bound_host, bound_port = server.server_address[:2]
    print(f"K3 Inference Calculator: http://{bound_host}:{bound_port}/", flush=True)
    print("Press Ctrl-C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
