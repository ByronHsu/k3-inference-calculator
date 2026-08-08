# K3 Inference Calculator

An interactive, per-layer roofline calculator for theoretical Kimi-K3 inference
on NVIDIA H200, B300, and GB300 systems at TP8, TP16, and TP32. H200 TP16 and
TP32 use the matching SGLang EP16 and EP32 recipes.

**Live calculator:** <https://byronhsu.github.io/k3-inference-calculator/>

The report covers cold prefill and decode, all 96 sequential model stages,
per-operator FLOPs and HBM traffic, communication floors, formula substitutions,
and per-rank memory accounting. Its interactive log-log roofline can be panned
and zoomed to compare compute-, memory-bandwidth-, and communication-bound
operators for the current Prefill or Decode workload. Results are optimistic analytical lower
bounds, not measured latency predictions or OOM guarantees.

## How the public site works

GitHub Pages is static, so the audited standard-library Python estimator runs
locally in the browser through a pinned Pyodide Web Worker. Calculations do not
depend on a backend server and no workload inputs leave the browser. The first
visit downloads the WebAssembly Python runtime; subsequent calculations reuse
the loaded worker.

The estimator source lives in `runtime/` and the UI lives in `site/`. The Pages
workflow tests the complete supported hardware/phase/TP matrix, builds
`runtime.zip`, and deploys the static artifact.

## Local development

```bash
python3 -m unittest discover -s tests
bash scripts/build_site.sh
python3 -m http.server 8000 --directory _site
```

Open <http://127.0.0.1:8000/>. Opening `index.html` directly will not work
because browsers block worker assets loaded from `file://` URLs.

## Scope

- Kimi-K3 text model: 93 decoder layers, with 69 KDA and 24 MLA layers
- Hardware families: H200, B300, GB300
- Tensor parallel sizes: 8, 16, 32
- H200 recipe mappings: TP16/EP16 and TP32/EP32
- H200 and B300 scale-out assumptions: 50 GB/s and 100 GB/s per GPU per direction
- Default efficiencies: 100% compute, HBM, and collective utilization
- Serving, scheduler, tokenizer, CPU, launch, and network-request overhead excluded

The model facts and runtime behavior are pinned in `runtime/python/sglang/benchmark/kimi_k3_theoretical/`.

## License

Apache-2.0. See `LICENSE`.
