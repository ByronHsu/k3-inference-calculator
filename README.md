# K3 Inference Calculator

An interactive, per-layer roofline calculator for theoretical Kimi-K3 inference
on NVIDIA H200, B300, and GB300 systems. The UI exposes six explicit sharding
scenarios: TP8, TP16, TP32, TP8+EP8, TP16+EP16, and TP32+EP32.

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
workflow tests the complete modeled hardware/phase/sharding matrix, builds
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
- Explicit sharding scenarios: TP8, TP16, TP32, TP8+EP8, TP16+EP16, TP32+EP32
- TP*N*+EP*N* uses the same *N* ranks, not *N*² GPUs; its execution recipe is
  hardware-specific rather than a generic expert-placement toggle
- H200 capacity: TP8 and TP8+EP8 are modeled, but their static weights exceed
  nominal HBM
- H200 TP+EP: TP16+EP16 and TP32+EP32 use the corresponding SGLang recipes
- B300/GB300 TP+EP: global TP/EP 8, 16, or 32 with attention TP8 and DP 1,
  2, or 4; the independent ledger models SP-MoE, MegaMoE A2A, DeepGEMM, and
  the overlapping replicated TP1 shared-expert branch from the
  [public SGLang Kimi-K3 recipe](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3)
- Blackwell TP8+EP8 is available from the public SGLang live generator but is
  omitted from the rendered 16-64 GPU table; it remains an unverified
  analytical scenario. The ordinary B300/GB300 8-GPU default is TP8 (or
  TP8+DCP8), not TP8+EP8
- The Blackwell ledger mirrors mandatory MLP-sync row alignment to attention
  TP8 before eager SUM_LEN/MAX_LEN padding or CUDA-graph MAX_LEN capture
- The Blackwell collective DAG fixes the public default
  `SGLANG_K3_SP_ATTN_RES=0`; cross-layer shard carry is outside this scenario
- Blackwell TP+EP numbers are conditional analytical lower bounds under balanced
  attention-DP request assignment and a fractional uniform-destination EP
  routing scenario. Routing locality or skew may redistribute demand between
  fabrics; it is a scenario change, not a positive excluded cost. MegaMoE control
  traffic, alignment, symmetric buffers, workspace, efficiency losses, and
  collective startup remain explicit excluded costs;
  therefore HBM fit is inconclusive unless the accounted lower bound already
  exceeds nominal capacity
- Every HBM-demand certificate is conditional on the counted logical tensor
  reads and writes materializing through HBM; cache residency or additional
  fusion changes the scenario, while backend rereads remain excluded
- H200 and B300 scale-out assumptions: 50 GB/s and 100 GB/s per GPU per direction
- Lower-bound certificate mode fixes compute, HBM, and collective efficiency at
  100%, collective startup at zero, and MLA KV read amplification at one
- Serving, scheduler, tokenizer, CPU, launch, and network-request overhead excluded

The model facts and runtime behavior are pinned in `runtime/python/sglang/benchmark/kimi_k3_theoretical/`.

## License

Apache-2.0. See `LICENSE`.
