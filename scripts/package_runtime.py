from __future__ import annotations

import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = REPO_ROOT / "runtime"
RUNTIME_FILES = (
    Path("benchmark/kimi_k3_inference_calculator.py"),
    Path("python/sglang/benchmark/kimi_k3_theoretical/__init__.py"),
    Path("python/sglang/benchmark/kimi_k3_theoretical/specs.py"),
    Path("python/sglang/benchmark/kimi_k3_theoretical/estimator.py"),
)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: package_runtime.py OUTPUT.zip")
    output = Path(sys.argv[1]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in RUNTIME_FILES:
            source = RUNTIME_ROOT / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            archive.write(source, relative.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
