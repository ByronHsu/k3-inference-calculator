"use strict";

const calculatorFetch = globalThis.k3CalculatorFetch || globalThis.fetch.bind(globalThis);

const HARDWARE_META = {
  h200: { short: "H200", color: "#ffad5c" },
  b300: { short: "B300", color: "#b8f455" },
  gb300: { short: "GB300", color: "#a88cff" },
};

const STAGE_GROUPS = [
  { id: "embedding", label: "Embedding", color: "#6aa6ff" },
  { id: "dense", label: "KDA + dense layer 1", color: "#ffad5c" },
  { id: "kda", label: "KDA + MoE", color: "#b8f455" },
  { id: "mla", label: "MLA + MoE", color: "#a88cff" },
  { id: "final", label: "Final residual norm", color: "#5ed6d2" },
  { id: "head", label: "LM head", color: "#ff7373" },
];

const FLOOR_COLORS = {
  critical_path: "#6aa6ff",
  compute: "#a88cff",
  hbm: "#5ed6d2",
  communication: "#ffad5c",
  unknown: "#8f969f",
};

const MEMORY_COLORS = ["#b8f455", "#a88cff", "#5ed6d2", "#ffad5c", "#6aa6ff"];
const AUTO_CALCULATE_DELAY_MS = 300;

const OPERATION_CALCULATION_FIELDS = [
  { key: "flops_per_rank", label: "FLOPs / rank" },
  { key: "hbm_bytes_per_rank", label: "HBM bytes / rank" },
  { key: "logical_collective_bytes", label: "Logical collective payload" },
  { key: "link_bytes_per_rank", label: "Fabric-byte diagnostic" },
  { key: "compute_seconds", label: "Compute floor" },
  { key: "hbm_seconds", label: "HBM floor" },
  { key: "communication_seconds", label: "Communication floor" },
  { key: "duration_seconds", label: "Operator roofline" },
];

const state = {
  response: null,
  manifest: null,
  results: [],
  expandedLayers: new Set(),
  expandedOperations: new Set(),
  chartHits: [],
  rooflineHits: [],
  rooflineViews: new Map(),
  rooflineLayout: null,
  rooflineDrag: null,
  abortController: null,
  autoCalculateTimer: null,
  ready: false,
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function finite(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function formatNumber(value, maximumFractionDigits = 0) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits }).format(finite(value));
}

function formatDuration(seconds, dashZero = false) {
  const value = finite(seconds);
  if (dashZero && value === 0) return "—";
  if (value < 1e-6) return `${formatNumber(value * 1e9, 3)} ns`;
  if (value < 1e-3) return `${formatNumber(value * 1e6, 3)} µs`;
  if (value < 1) return `${formatNumber(value * 1e3, 3)} ms`;
  return `${formatNumber(value, 6)} s`;
}

function durationParts(seconds) {
  const formatted = formatDuration(seconds);
  const split = formatted.lastIndexOf(" ");
  return { value: formatted.slice(0, split), unit: formatted.slice(split + 1) };
}

function formatBytes(bytes, dashZero = false) {
  let value = finite(bytes);
  if (dashZero && value === 0) return "—";
  const units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
  let index = 0;
  while (Math.abs(value) >= 1024 && index < units.length - 1) {
    value /= 1024;
    index += 1;
  }
  const digits = value >= 100 ? 1 : value >= 10 ? 2 : 3;
  return `${formatNumber(value, digits)} ${units[index]}`;
}

function formatFlops(flops, dashZero = false) {
  let value = finite(flops);
  if (dashZero && value === 0) return "—";
  const units = ["F", "KF", "MF", "GF", "TF", "PF"];
  let index = 0;
  while (Math.abs(value) >= 1000 && index < units.length - 1) {
    value /= 1000;
    index += 1;
  }
  return `${formatNumber(value, value >= 100 ? 1 : 2)} ${units[index]}`;
}

function formatFlopsPerSecond(value) {
  return `${formatFlops(value)}/s`;
}

function formatIntensity(value) {
  const intensity = finite(value);
  if (intensity >= 1000) return `${formatNumber(intensity / 1000, 2)}K FLOP/B`;
  if (intensity >= 1) return `${formatNumber(intensity, 2)} FLOP/B`;
  return `${formatNumber(intensity, 4)} FLOP/B`;
}

function formatRate(value) {
  const number = finite(value);
  if (number >= 1e9) return `${formatNumber(number / 1e9, 2)}G tok/s`;
  if (number >= 1e6) return `${formatNumber(number / 1e6, 2)}M tok/s`;
  if (number >= 1e3) return `${formatNumber(number / 1e3, 2)}K tok/s`;
  return `${formatNumber(number, 2)} tok/s`;
}

function formatPercent(ratio, digits = 1) {
  return `${formatNumber(finite(ratio) * 100, digits)}%`;
}

function humanize(value) {
  return String(value ?? "")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function hardwareMeta(resultOrId) {
  const hardware = typeof resultOrId === "string" ? null : resultOrId?.hardware;
  const id = String(typeof resultOrId === "string" ? resultOrId : hardware?.id || "").toLowerCase();
  const declaredFamily = String(hardware?.family || "").toLowerCase();
  const family = HARDWARE_META[declaredFamily]
    ? declaredFamily
    : Object.keys(HARDWARE_META).find(
        (name) => id === name || id.startsWith(`${name}-`),
      );
  return HARDWARE_META[family] ?? { short: id || "Hardware", color: "#b8f455" };
}

function resultById(id) {
  return state.results.find((result) => result.hardware.id === id) ?? state.results[0];
}

function normalizeFloor(value) {
  const aliases = {
    critical_path: "critical_path",
    dependency: "critical_path",
    compute: "compute",
    compute_resource: "compute",
    hbm: "hbm",
    hbm_resource: "hbm",
    communication: "communication",
    communication_resource: "communication",
  };
  return aliases[String(value || "").toLowerCase()] || "unknown";
}

const FLOOR_LABELS = {
  compute: "Compute",
  hbm: "Memory bandwidth",
  communication: "Communication",
  critical_path: "Critical path",
  unknown: "Unknown",
};

const CERTIFICATE_LABELS = {
  critical_path: "Critical-path lower bound",
  compute: "Compute-demand lower bound",
  hbm: "HBM-demand lower bound",
  communication: "Communication-demand lower bound",
  unknown: "Unknown lower-bound certificate",
};

const PADDING_MODE_LABELS = {
  max_len: "MAX_LEN",
  sum_len: "SUM_LEN",
  max_len_cuda_graph: "CUDA graph MAX_LEN",
};

const EXCLUDED_TERM_LABELS = {
  megamoe_alignment_padding: "MegaMoE alignment/padding",
  topk_compute: "TopK compute",
  predispatch_quant_compute: "Pre-dispatch quantization compute",
  expert_activation_compute: "Expert activation compute",
  megamoe_internal_hbm_traffic: "MegaMoE internal HBM traffic",
  fp8_scale_transport: "FP8 scale transport",
  megamoe_control_metadata: "MegaMoE control metadata",
  megamoe_symmetric_buffer_copies: "MegaMoE symmetric-buffer copies",
  megamoe_transformed_weight_workspace: "MegaMoE transformed-weight workspace",
  fabric_contention: "Fabric contention",
  collective_startup: "Collective startup",
};

function floorLabel(value) {
  return FLOOR_LABELS[normalizeFloor(value)];
}

function boundLabel(value) {
  const floor = normalizeFloor(value);
  return floor === "unknown" ? "Unknown bound" : `${floorLabel(floor)}-bound`;
}

function certificateLabel(value) {
  return CERTIFICATE_LABELS[normalizeFloor(value)];
}

function layerCertificates(layer) {
  const values = Array.isArray(layer?.limiting_certificates)
    ? layer.limiting_certificates
    : [];
  return [...new Set(values.map(normalizeFloor))];
}

function certificateChips(layer) {
  return layerCertificates(layer)
    .map(
      (certificate) =>
        `<span class="resource-chip ${certificate}">${escapeHtml(certificateLabel(certificate))}</span>`,
    )
    .join("");
}

function parallelismLabel(hardware) {
  if (hardware.moe_sharding === "ep") {
    return `TP${hardware.tp_size}+EP${hardware.ep_size}`;
  }
  return `TP${hardware.tp_size}`;
}

function expertParallelismLabel(hardware) {
  return hardware.moe_sharding === "ep" ? `EP${hardware.ep_size}` : "EP off";
}

function setView(view) {
  const loading = $("#loading-state");
  const error = $("#error-state");
  const results = $("#results");
  loading.hidden = view !== "loading";
  error.hidden = view !== "error";
  results.hidden = view !== "results";
  $("#report").setAttribute("aria-busy", view === "loading" ? "true" : "false");
  $("#calculation-status").textContent =
    {
      loading: "Updating calculation.",
      error: "Calculation failed.",
      results: "Calculation updated.",
    }[view] || "";
}

function initTheme() {
  const stored = (() => {
    try {
      return localStorage.getItem("k3-calculator-theme");
    } catch {
      return null;
    }
  })();
  const preferred = window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  setTheme(stored === "light" || stored === "dark" ? stored : preferred);
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const toggle = $("#theme-toggle");
  toggle.setAttribute("aria-label", `Switch to ${theme === "dark" ? "light" : "dark"} theme`);
  toggle.title = `Switch to ${theme === "dark" ? "light" : "dark"} theme`;
  try {
    localStorage.setItem("k3-calculator-theme", theme);
  } catch {
    // Theme persistence is optional in privacy-restricted contexts.
  }
  if (state.results.length) {
    requestAnimationFrame(renderLayerChart);
    requestAnimationFrame(renderRooflineChart);
  }
}

function currentPhase() {
  return $("input[name='phase']:checked").value;
}

function updatePhaseFields() {
  const decode = currentPhase() === "decode";
  $("#prefill-fields").hidden = decode;
  $("#decode-fields").hidden = !decode;
  $("#sequence-length").required = !decode;
  $("#batch-size").required = decode;
  $("#context-length").required = decode;
}

function selectedHardware() {
  return $$("input[name='hardware']:checked").map((input) => input.value);
}

function selectedSharding() {
  const input = $("input[name='sharding']:checked");
  if (!input) return null;
  const tpSize = Number(input.dataset.tpSize);
  const moeSharding = input.dataset.moeSharding;
  return {
    id: input.value,
    tpSize,
    moeSharding,
    label: moeSharding === "ep" ? `TP${tpSize}+EP${tpSize}` : `TP${tpSize}`,
  };
}

function updateHardwareState() {
  updateShardingCompatibility();
  updateSequenceLimit();
}

function manifestShardingOption(id) {
  const options = state.manifest?.sharding_options;
  return Array.isArray(options) ? options.find((option) => option.id === id) : null;
}

function shardingSupport(sharding, family) {
  const declared = manifestShardingOption(sharding.id)?.families?.[family];
  if (declared && typeof declared === "object" && typeof declared.status === "string") {
    return declared;
  }
  return { status: "support_not_declared", reason: "Support is not declared by the manifest." };
}

function unsupportedSelectedHardware(sharding = selectedSharding()) {
  if (!sharding) return [];
  return selectedHardware()
    .map((family) => ({ family, support: shardingSupport(sharding, family) }))
    .filter(({ support }) => support.status !== "modeled");
}

function updateShardingCompatibility() {
  const selected = selectedHardware();
  const allFamilies = $$("input[name='hardware']").map((input) => input.value);
  const hasSupportMatrix = Array.isArray(state.manifest?.sharding_options);
  for (const input of $$("input[name='sharding']")) {
    const sharding = {
      id: input.value,
      tpSize: Number(input.dataset.tpSize),
      moeSharding: input.dataset.moeSharding,
    };
    const option = input.closest(".choice-option");
    const description = $(`#${input.getAttribute("aria-describedby")}`, option);
    if (!hasSupportMatrix) {
      option.classList.remove("is-unavailable", "is-partial");
      option.title = "Loading sharding support.";
      if (description) description.textContent = "Loading sharding support.";
      continue;
    }
    const unavailable = selected.filter(
      (family) => shardingSupport(sharding, family).status !== "modeled",
    );
    option.classList.toggle(
      "is-unavailable",
      selected.length > 0 && unavailable.length === selected.length,
    );
    option.classList.toggle(
      "is-partial",
      unavailable.length > 0 && unavailable.length < selected.length,
    );
    const reasons = unavailable.map((family) => {
      const reason = shardingSupport(sharding, family).reason || "Model unavailable.";
      return `${family.toUpperCase()}: ${reason}`;
    });
    const modeledFamilies = allFamilies.filter(
      (family) => shardingSupport(sharding, family).status === "modeled",
    );
    const unmodeledFamilies = allFamilies.filter(
      (family) => shardingSupport(sharding, family).status !== "modeled",
    );
    const capacityLimited = modeledFamilies.filter(
      (family) =>
        shardingSupport(sharding, family).capacity_hint ===
        "static_weights_exceed_nominal_hbm",
    );
    const supportText = modeledFamilies.length
      ? `Modeled for ${modeledFamilies.map((family) => family.toUpperCase()).join(", ")}.`
      : "Not modeled for any hardware family.";
    const unavailableText = unmodeledFamilies.length
      ? ` Not modeled for ${unmodeledFamilies
          .map((family) => family.toUpperCase())
          .join(", ")}.`
      : "";
    const capacityText = capacityLimited.length
      ? ` Static weights exceed nominal HBM on ${capacityLimited
          .map((family) => family.toUpperCase())
          .join(", ")}.`
      : "";
    const descriptionText = `${supportText}${unavailableText}${capacityText}`;
    option.title = [descriptionText, ...reasons].join(" ");
    if (description) description.textContent = descriptionText;
  }

  const status = $("#sharding-support");
  const sharding = selectedSharding();
  if (!hasSupportMatrix) {
    status.dataset.state = "loading";
    status.textContent = "Loading sharding support…";
    return;
  }
  const unsupported = unsupportedSelectedHardware(sharding);
  const capacityLimited = sharding
    ? selected.filter(
        (family) =>
          shardingSupport(sharding, family).capacity_hint ===
          "static_weights_exceed_nominal_hbm",
      )
    : [];
  if (!selected.length) {
    status.dataset.state = "error";
    status.textContent = "Select at least one hardware family.";
  } else if (!sharding) {
    status.dataset.state = "error";
    status.textContent = "Select one sharding scenario.";
  } else if (unsupported.length) {
    const names = unsupported.map(({ family }) => family.toUpperCase()).join(", ");
    status.dataset.state = "error";
    status.textContent = `${sharding.label} is recognized but its execution model is unavailable for ${names}; no estimate will run.`;
  } else if (capacityLimited.length) {
    const names = capacityLimited.map((family) => family.toUpperCase()).join(", ");
    status.dataset.state = "warning";
    status.textContent = `${names} ${sharding.label} is modeled, but its static weights exceed nominal HBM.`;
  } else {
    status.dataset.state = "ok";
    status.textContent = sharding.moeSharding === "ep"
      ? `All selected families support ${sharding.label}; Blackwell uses attention TP8, DP=N/8, SP-MoE, and MegaMoE/DeepGEMM.`
      : `All selected hardware families support ${sharding.label}.`;
  }
}

function manifestHardwarePresets() {
  const raw = state.manifest?.hardware_families || state.manifest?.hardware_presets;
  if (Array.isArray(raw)) return raw;
  if (raw && typeof raw === "object") return Object.values(raw);
  return [];
}

function updateSequenceLimit() {
  const ids = selectedHardware();
  const presets = manifestHardwarePresets().filter((item) =>
    ids.some((id) => String(item.id || item.family || "").toLowerCase() === id),
  );
  const manifestLimits = presets
    .map((item) => finite(item.prefill_chunk_size, Infinity))
    .filter(Number.isFinite);
  const fallbackLimit = ids.includes("h200") ? 8192 : 16384;
  const limit = manifestLimits.length ? Math.min(...manifestLimits) : fallbackLimit;
  if (Number.isFinite(limit)) {
    $("#sequence-length").max = String(limit);
    $("#sequence-length").title = `Maximum one-chunk sequence for this selection: ${formatNumber(limit)} tokens`;
  }
}

function validateForm() {
  const phase = currentPhase();
  const hardware = selectedHardware();
  if (!hardware.length) return "Select at least one hardware family.";
  const sharding = selectedSharding();
  if (
    !sharding ||
    ![8, 16, 32].includes(sharding.tpSize) ||
    !["tp", "ep"].includes(sharding.moeSharding)
  ) {
    return "Select one of the six supported sharding scenarios.";
  }
  const unsupported = unsupportedSelectedHardware(sharding);
  if (unsupported.length) {
    const names = unsupported.map(({ family }) => family.toUpperCase()).join(", ");
    return `${sharding.label} is recognized but not modeled for ${names}. Deselect those hardware families or choose a TP-only scenario.`;
  }
  if (phase === "prefill") {
    const sequenceLength = Number($("#sequence-length").value);
    const limit = Number($("#sequence-length").max);
    if (!Number.isInteger(sequenceLength) || sequenceLength <= 0) {
      return "Sequence length must be a positive whole number.";
    }
    if (Number.isFinite(limit) && sequenceLength > limit) {
      return `This selection supports one cold-prefill chunk up to ${formatNumber(limit)} tokens.`;
    }
  } else {
    const batchSize = Number($("#batch-size").value);
    const batchLimit = Number($("#batch-size").max);
    const contextLength = Number($("#context-length").value);
    const contextLimit = Number($("#context-length").max);
    if (!Number.isInteger(batchSize) || batchSize <= 0) {
      return "Decode batch size must be a positive whole number.";
    }
    if (Number.isFinite(batchLimit) && batchSize > batchLimit) {
      return `Decode batch size must not exceed ${formatNumber(batchLimit)}.`;
    }
    if (!Number.isInteger(contextLength) || contextLength <= 0) {
      return "Decode context length must be a positive whole number.";
    }
    if (Number.isFinite(contextLimit) && contextLength > contextLimit) {
      return `Decode context must not exceed ${formatNumber(contextLimit)} tokens.`;
    }
  }
  return null;
}

function requestPayload() {
  const phase = currentPhase();
  const sharding = selectedSharding();
  const payload = {
    phase,
    hardware: selectedHardware(),
    tp_size: sharding.tpSize,
    moe_sharding: sharding.moeSharding,
    batch_size: phase === "prefill" ? 1 : Number($("#batch-size").value),
  };
  if (phase === "prefill") payload.sequence_length = Number($("#sequence-length").value);
  else payload.context_length = Number($("#context-length").value);
  return payload;
}

function validateCalculatorResponse(payload, body) {
  if (body?.schema_version !== 2) {
    throw new Error("The calculator returned an unsupported response schema; no result was accepted.");
  }
  const results = Array.isArray(body.results) ? body.results : [];
  if (!results.length) throw new Error("The calculator returned no hardware results.");
  if (results.some((result) => !Array.isArray(result.layers) || !result.hardware)) {
    throw new Error("The calculator response is missing its layer or hardware inventory.");
  }

  const requestedFamilies = [...new Set(payload.hardware.map((family) => String(family).toLowerCase()))];
  if (requestedFamilies.length !== payload.hardware.length) {
    throw new Error("The calculator request contains duplicate hardware families.");
  }
  const requestedSet = new Set(requestedFamilies);
  const returnedFamilies = new Set();
  const expectedTpSize = payload.tp_size;
  const expectedMoeSharding = payload.moe_sharding;
  const expectedEpSize = expectedMoeSharding === "ep" ? expectedTpSize : 1;

  for (const result of results) {
    const hardware = result.hardware;
    const family = String(hardware.family || "").toLowerCase();
    if (!requestedSet.has(family) || returnedFamilies.has(family)) {
      throw new Error(
        "The calculator returned an unexpected or duplicate hardware family; no result was accepted.",
      );
    }
    returnedFamilies.add(family);
    const certificateFields = [
      ["critical_path", "critical_path_lower_bound_seconds"],
      ["compute", "compute_resource_seconds"],
      ["hbm", "hbm_resource_seconds"],
      ["communication", "communication_resource_seconds"],
    ];
    for (const layer of result.layers) {
      const values = certificateFields.map(([name, field]) => [name, layer?.[field]]);
      if (
        values.some(([, value]) => !Number.isFinite(value) || value < 0) ||
        !Number.isFinite(layer?.latency_seconds) ||
        layer.latency_seconds < 0 ||
        !Array.isArray(layer?.limiting_certificates) ||
        !layer.limiting_certificates.length
      ) {
        throw new Error(
          `The calculator returned ${family.toUpperCase()} without the canonical lower-bound certificate fields; no result was accepted.`,
        );
      }
      const expectedLatency = Math.max(...values.map(([, value]) => value));
      const expectedCertificates = values
        .filter(([, value]) => value === expectedLatency)
        .map(([name]) => name);
      if (
        layer.latency_seconds !== expectedLatency ||
        layer.limiting_certificates.length !== expectedCertificates.length ||
        layer.limiting_certificates.some(
          (name, index) => name !== expectedCertificates[index],
        )
      ) {
        throw new Error(
          `The calculator returned ${family.toUpperCase()} with inconsistent lower-bound certificates; no result was accepted.`,
        );
      }
    }
    if (
      hardware.gpu_count !== expectedTpSize ||
      hardware.tp_size !== expectedTpSize ||
      hardware.moe_sharding !== expectedMoeSharding ||
      hardware.ep_size !== expectedEpSize
    ) {
      throw new Error(
        `The calculator returned ${family.toUpperCase()} with a sharding topology that does not match the request; no result was accepted.`,
      );
    }
    const blackwellEp =
      expectedMoeSharding === "ep" && ["b300", "gb300"].includes(family);
    const ledger = result.parallel_work_ledger;
    if (
      blackwellEp &&
      (hardware.execution_contract_id !== "kimi_k3_blackwell_megamoe_sp_v1" ||
        ledger?.contract_id !== "kimi_k3_blackwell_parallel_ledger_v1" ||
        ledger?.bound_condition_id !== "balanced_dp_fractional_uniform_ep_routing" ||
        typeof ledger.bound_condition !== "string" ||
        !ledger.bound_condition ||
        typeof ledger.topology_contract !== "string" ||
        !ledger.topology_contract ||
        !Array.isArray(ledger.dp_mlp_aligned_rows) ||
        ledger.dp_mlp_aligned_rows.length !== expectedTpSize / 8 ||
        !Array.isArray(ledger.dp_model_rows) ||
        ledger.dp_model_rows.length !== expectedTpSize / 8 ||
        !Array.isArray(ledger.excluded_positive_term_ids) ||
        !ledger.excluded_positive_term_ids.includes("megamoe_alignment_padding") ||
        !ledger.excluded_positive_term_ids.includes("collective_startup"))
    ) {
      throw new Error(
        `The calculator returned ${family.toUpperCase()} with a Blackwell EP execution recipe that does not match the request; no result was accepted.`,
      );
    }
    const returnedWorkload = result.workload;
    const workloadMatches =
      returnedWorkload?.phase === payload.phase &&
      returnedWorkload.batch_size === payload.batch_size &&
      (payload.phase === "prefill"
        ? returnedWorkload.sequence_length === payload.sequence_length
        : returnedWorkload.context_length === payload.context_length);
    if (!workloadMatches) {
      throw new Error(
        `The calculator returned ${family.toUpperCase()} with a workload that does not match the request; no result was accepted.`,
      );
    }
  }
  if (returnedFamilies.size !== requestedFamilies.length) {
    throw new Error("The calculator returned a partial hardware comparison; no result was accepted.");
  }
  return results;
}

function scheduleCalculate({ immediate = false } = {}) {
  window.clearTimeout(state.autoCalculateTimer);
  state.autoCalculateTimer = null;
  if (!state.ready) return;
  state.abortController?.abort();
  if (immediate) {
    void calculate();
    return;
  }
  state.autoCalculateTimer = window.setTimeout(() => {
    state.autoCalculateTimer = null;
    void calculate();
  }, AUTO_CALCULATE_DELAY_MS);
}

async function calculate() {
  const formError = $("#form-error");
  const validation = validateForm();
  if (validation) {
    formError.textContent = validation;
    formError.hidden = false;
    $("#error-message").textContent = validation;
    setView("error");
    return;
  }
  formError.hidden = true;
  state.abortController?.abort();
  const controller = new AbortController();
  state.abortController = controller;
  setView("loading");
  const payload = requestPayload();

  try {
    const response = await calculatorFetch("./api/calculate", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    let body;
    try {
      body = await response.json();
    } catch {
      throw new Error(`The calculator returned an unreadable response (${response.status}).`);
    }
    if (!response.ok || body?.error) {
      throw new Error(body?.error?.message || `Calculation failed (${response.status}).`);
    }
    if (state.abortController !== controller || controller.signal.aborted) return;
    const results = validateCalculatorResponse(payload, body);
    state.response = body;
    state.results = results;
    state.manifest = { ...(state.manifest || {}), ...body, results: undefined };
    state.expandedLayers.clear();
    state.expandedOperations.clear();
    state.rooflineViews.clear();
    state.rooflineLayout = null;
    state.rooflineDrag = null;
    renderReport();
    setView("results");
  } catch (error) {
    if (error.name === "AbortError") return;
    $("#error-message").textContent = error.message || "Unknown calculation error.";
    setView("error");
  } finally {
    if (state.abortController === controller) state.abortController = null;
  }
}

async function loadManifest() {
  try {
    const response = await calculatorFetch("./api/manifest", {
      credentials: "same-origin",
      headers: { Accept: "application/json" },
    });
    if (!response.ok) return false;
    const body = await response.json();
    if (body && typeof body === "object") {
      state.manifest = body;
      updateShardingCompatibility();
      updateSequenceLimit();
      return true;
    }
  } catch (error) {
    const message = error?.message || "The browser calculation engine failed to load.";
    console.error(message);
    $("#error-message").textContent = message;
    setView("error");
  }
  return false;
}

function populateHardwareSelect(select, preferredId) {
  const previous = preferredId || select.value;
  select.innerHTML = state.results
    .map((result) => {
      const meta = hardwareMeta(result);
      return `<option value="${escapeHtml(result.hardware.id)}">${escapeHtml(meta.short)} · ${escapeHtml(parallelismLabel(result.hardware))}</option>`;
    })
    .join("");
  if (state.results.some((result) => result.hardware.id === previous)) select.value = previous;
}

function renderReport() {
  const first = state.results[0];
  const phase = first.workload.phase;
  $("#report-phase").textContent = `${phase === "prefill" ? "Prefill" : "Decode"} · ${parallelismLabel(first.hardware)}`;
  $("#report-workload").textContent =
    phase === "prefill"
      ? `${formatNumber(first.workload.sequence_length)} tokens · batch 1`
      : `batch ${formatNumber(first.workload.batch_size)}${first.workload.model_batch_size !== first.workload.batch_size ? ` → ${formatNumber(first.workload.model_batch_size)} execution rows` : ""} · ${formatNumber(first.workload.context_length)} context · ${first.decode_cuda_graph_replay ? "CUDA graph replay" : "eager"}`;
  const hasConditionalEpScenario = state.results.some(
    (result) => result.parallel_work_ledger?.bound_condition_id,
  );
  const status = hasConditionalEpScenario
    ? "conditional analytical lower bound under logical HBM materialization, balanced DP assignment, and a fractional uniform-destination EP routing scenario; not measured"
    : state.response?.analytical_status || state.manifest?.analytical_status || "optimistic lower bound; not measured";
  $("#report-status").textContent = `${status.charAt(0).toUpperCase()}${status.slice(1)}.`;
  const time = new Date();
  $("#generated-time").dateTime = time.toISOString();
  $("#generated-time").textContent = time.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  renderComparison();
  for (const id of [
    "breakdown-hardware",
    "roofline-hardware",
    "layer-hardware",
    "memory-hardware",
  ]) {
    populateHardwareSelect($(`#${id}`));
  }
  renderBreakdown();
  renderRoofline();
  renderLayerAnalysis();
  renderMemory();
  renderManifest();
}

function renderComparison() {
  const cards = $("#comparison-cards");
  const fittingResults = state.results.filter(
    (result) => result.memory.fits_nominal_capacity === true,
  );
  const fastest = fittingResults.length
    ? Math.min(...fittingResults.map((result) => finite(result.total_seconds, Infinity)))
    : null;
  cards.style.setProperty("--comparison-columns", String(Math.min(state.results.length, 3)));
  cards.innerHTML = state.results
    .map((result) => {
      const meta = hardwareMeta(result);
      const parts = durationParts(result.total_seconds);
      const fits = result.memory.fits_nominal_capacity;
      const capacityUnknown = fits == null;
      const isFastest = fits === true && fastest != null && Math.abs(result.total_seconds - fastest) <= fastest * 1e-9;
      const ratio = fastest == null ? null : result.total_seconds / fastest;
      const memoryRatio = result.memory.total_accounted_peak_bytes_per_rank / result.memory.nominal_hbm_capacity_bytes_per_rank;
      const conditionalRoute = result.parallel_work_ledger
        ? "Conditional DP assignment and EP routing"
        : null;
      const capacityNote = fits === false
        ? "Capacity-infeasible analytical result"
        : capacityUnknown
          ? "MegaMoE workspace excluded · fit is inconclusive"
          : state.results.length === 1
            ? "Selected configuration"
            : isFastest
              ? "Lowest lower bound among accounted-fit results"
              : `${formatNumber((ratio - 1) * 100, 1)}% above the lowest accounted-fit lower bound`;
      return `
        <article class="comparison-card ${isFastest ? "fastest" : ""} ${fits === false ? "does-not-fit" : ""} ${capacityUnknown ? "capacity-unknown" : ""}" style="--card-color:${meta.color}">
          <div class="card-system-row">
            <span class="system-name"><i class="system-dot"></i><strong>${escapeHtml(meta.short)} · ${escapeHtml(parallelismLabel(result.hardware))}</strong></span>
            ${fits === false ? '<span class="capacity-warning-badge">Does not fit</span>' : capacityUnknown ? '<span class="capacity-unknown-badge">Capacity unknown</span>' : isFastest ? '<span class="winner-badge">Lowest accounted-fit bound</span>' : ""}
          </div>
          <div>
            <div class="card-latency">${escapeHtml(parts.value)} <small>${escapeHtml(parts.unit)}</small></div>
            <div class="card-subline">${escapeHtml([conditionalRoute, capacityNote].filter(Boolean).join(" · "))}</div>
          </div>
          <dl class="card-metrics">
            <div><dt>Ideal throughput</dt><dd>${escapeHtml(formatRate(result.ideal_tokens_per_second))}</dd></div>
            <div><dt>HBM accounted</dt><dd>${escapeHtml(formatPercent(memoryRatio))}</dd></div>
          </dl>
        </article>`;
    })
    .join("");

  $("#comparison-table-body").innerHTML = state.results
    .map((result) => {
      const meta = hardwareMeta(result);
      const executionPath =
        result.workload.phase === "prefill"
          ? "Cold prefill"
          : result.decode_cuda_graph_replay
            ? `CUDA graph · B${formatNumber(result.workload.model_batch_size)}`
            : "Eager decode";
      return `
        <tr>
          <td><span class="comparison-system"><i class="system-dot" style="--card-color:${meta.color}"></i>${escapeHtml(meta.short)} ${escapeHtml(parallelismLabel(result.hardware))}</span></td>
          <td class="mono">${escapeHtml(parallelismLabel(result.hardware))}</td>
          <td class="numeric">${escapeHtml(formatDuration(result.total_seconds))}</td>
          <td class="numeric">${escapeHtml(formatRate(result.ideal_tokens_per_second))}</td>
          <td class="numeric">${escapeHtml(formatBytes(result.memory.static_weight_bytes_per_rank))}</td>
          <td class="numeric">${escapeHtml(formatBytes(result.memory.total_accounted_peak_bytes_per_rank))}</td>
          <td><span class="fit-badge ${result.memory.fits_nominal_capacity === true ? "fits" : result.memory.fits_nominal_capacity === false ? "over" : "unknown"}">${result.memory.fits_nominal_capacity === true ? "Fits accounted peak" : result.memory.fits_nominal_capacity === false ? "Accounted lower bound exceeds HBM" : "Fit inconclusive · workspace excluded"}</span></td>
          <td><span class="path-chip other">${escapeHtml(executionPath)}</span></td>
        </tr>`;
    })
    .join("");
}

function stageGroupForLayer(layer) {
  if (layer.name === "embedding") return "embedding";
  if (layer.name === "lm_head") return "head";
  if (layer.number === 1 || layer.ffn === "dense") return "dense";
  if (layer.attention === "kda") return "kda";
  if (layer.attention === "mla") return "mla";
  return "final";
}

function renderBreakdown() {
  const result = resultById($("#breakdown-hardware").value);
  if (!result) return;
  const totals = Object.fromEntries(STAGE_GROUPS.map((group) => [group.id, 0]));
  for (const layer of result.layers) totals[stageGroupForLayer(layer)] += finite(layer.latency_seconds);
  const sum = Object.values(totals).reduce((accumulator, value) => accumulator + value, 0);
  const additiveTolerance = Math.max(1e-12, Math.abs(result.total_seconds) * 1e-8);
  console.assert(
    Math.abs(sum - result.total_seconds) <= additiveTolerance,
    "Exact stage groups must sum to total_seconds",
    { stageSum: sum, reportedTotal: result.total_seconds },
  );
  $("#stage-total").textContent = formatDuration(sum);
  const stackedBar = $("#stage-stacked-bar");
  stackedBar.setAttribute(
    "aria-label",
    STAGE_GROUPS.map((group) => `${group.label} ${formatPercent(totals[group.id] / sum)}`).join(", "),
  );
  stackedBar.innerHTML = STAGE_GROUPS.map((group) => {
    const ratio = sum ? totals[group.id] / sum : 0;
    return `<span class="stacked-bar-segment" style="width:${ratio * 100}%;--segment-color:${group.color}" title="${escapeHtml(group.label)}: ${escapeHtml(formatDuration(totals[group.id]))} (${escapeHtml(formatPercent(ratio))})"></span>`;
  }).join("");
  $("#stage-legend").innerHTML = STAGE_GROUPS.map((group) => {
    const ratio = sum ? totals[group.id] / sum : 0;
    return `<div class="legend-item" style="--item-color:${group.color}"><i></i><span>${escapeHtml(group.label)}</span><strong>${escapeHtml(formatDuration(totals[group.id]))} · ${escapeHtml(formatPercent(ratio))}</strong></div>`;
  }).join("");

  const resources = {
    critical_path: result.layers.reduce((sumValue, layer) => sumValue + finite(layer.critical_path_lower_bound_seconds), 0),
    compute: result.layers.reduce((sumValue, layer) => sumValue + finite(layer.compute_resource_seconds), 0),
    hbm: result.layers.reduce((sumValue, layer) => sumValue + finite(layer.hbm_resource_seconds), 0),
    communication: result.layers.reduce((sumValue, layer) => sumValue + finite(layer.communication_resource_seconds), 0),
  };
  $("#resource-bars").setAttribute(
    "aria-label",
    Object.entries(resources)
      .map(([resource, value]) => `${certificateLabel(resource)} ${formatDuration(value)}`)
      .join(", "),
  );
  $("#resource-bars").innerHTML = Object.entries(resources)
    .map(([resource, value]) => {
      const ratio = result.total_seconds ? Math.min(value / result.total_seconds, 1) : 0;
      return `
        <div class="resource-column">
          <div class="resource-column-head"><span>${escapeHtml(certificateLabel(resource))}</span><strong>${escapeHtml(formatDuration(value))}</strong></div>
          <div class="resource-track" title="Independent, overlapping certificate: ${escapeHtml(formatPercent(ratio))} of the latency lower bound"><i style="--bar-width:${ratio * 100}%;--bar-color:${FLOOR_COLORS[resource]}"></i></div>
        </div>`;
    })
    .join("");
  $("#certificate-latency-total").textContent = formatDuration(result.total_seconds);
}

function collectRooflineData(result) {
  const pointGroups = new Map();
  const communicationGroups = new Map();
  const boundCounts = { compute: 0, hbm: 0, communication: 0 };

  for (const layer of result.layers) {
    const stageId = stageGroupForLayer(layer);
    const stageLabel =
      STAGE_GROUPS.find((group) => group.id === stageId)?.label || humanize(stageId);
    for (const operation of layer.operations) {
      const bottleneck = normalizeFloor(operation.bottleneck);
      if (Object.hasOwn(boundCounts, bottleneck)) boundCounts[bottleneck] += 1;

      const operationIntensity = finite(
        operation.arithmetic_intensity_flops_per_hbm_byte,
        Number.NaN,
      );
      const operationPerformance = finite(
        operation.roofline_flops_per_second,
        Number.NaN,
      );
      if (
        operationIntensity > 0 &&
        operationPerformance > 0 &&
        Number.isFinite(operationIntensity) &&
        Number.isFinite(operationPerformance)
      ) {
        const key = [
          stageId,
          operation.id,
          operation.compute_kind,
          bottleneck,
        ].join(":");
        const group = pointGroups.get(key) || {
          id: operation.id,
          name: operation.name,
          stage: stageLabel,
          computeKind: operation.compute_kind,
          bottleneck,
          count: 0,
          flops: 0,
          hbmBytes: 0,
          duration: 0,
          computeSeconds: 0,
          hbmSeconds: 0,
          communicationSeconds: 0,
        };
        group.count += 1;
        group.flops += finite(operation.flops_per_rank);
        group.hbmBytes += finite(operation.hbm_bytes_per_rank);
        group.duration += finite(operation.duration_seconds);
        group.computeSeconds += finite(operation.compute_seconds);
        group.hbmSeconds += finite(operation.hbm_seconds);
        group.communicationSeconds += finite(operation.communication_seconds);
        pointGroups.set(key, group);
      }

      if (
        finite(operation.communication_seconds) > 0 &&
        finite(operation.flops_per_rank) === 0
      ) {
        const key = `${stageId}:${operation.id}`;
        const group = communicationGroups.get(key) || {
          id: operation.id,
          name: operation.name,
          stage: stageLabel,
          count: 0,
          seconds: 0,
          logicalBytes: 0,
        };
        group.count += 1;
        group.seconds += finite(operation.communication_seconds);
        group.logicalBytes += finite(operation.logical_collective_bytes);
        communicationGroups.set(key, group);
      }
    }
  }

  const points = Array.from(pointGroups.values())
    .map((group) => ({
      ...group,
      intensity: group.flops / group.hbmBytes,
      performance: group.flops / group.duration,
    }))
    .sort((left, right) => left.intensity - right.intensity);
  const communicationOnly = Array.from(communicationGroups.values()).sort(
    (left, right) => right.seconds - left.seconds,
  );
  return { points, communicationOnly, boundCounts };
}

function automaticRooflineView(result, points) {
  const hardware = result.hardware;
  const bandwidth = finite(hardware.hbm_bandwidth_bytes_per_s, 1);
  const peaks = [
    finite(hardware.dense_bf16_flops_per_s, 1),
    finite(hardware.k3_expert_flops_per_s, 1),
  ].filter((value) => value > 0);
  const intensities = points.map((point) => point.intensity).filter((value) => value > 0);
  intensities.push(...peaks.map((peak) => peak / bandwidth));
  const performances = points
    .map((point) => point.performance)
    .filter((value) => value > 0);
  performances.push(...peaks);

  const xLogs = intensities.map((value) => Math.log10(value));
  const yLogs = performances.map((value) => Math.log10(value));
  let xMin = Math.floor(Math.min(...xLogs)) - 0.25;
  let xMax = Math.ceil(Math.max(...xLogs)) + 0.25;
  let yMin = Math.floor(Math.min(...yLogs)) - 0.25;
  let yMax = Math.ceil(Math.max(...yLogs)) + 0.25;
  if (xMax - xMin < 2) {
    const center = (xMin + xMax) / 2;
    xMin = center - 1;
    xMax = center + 1;
  }
  if (yMax - yMin < 2) {
    const center = (yMin + yMax) / 2;
    yMin = center - 1;
    yMax = center + 1;
  }
  return { xMin, xMax, yMin, yMax };
}

function rooflineViewKey(result) {
  return `${result.hardware.id}:${result.workload.phase}`;
}

function logarithmicTicks(minimum, maximum, maximumTicks = 7) {
  const first = Math.ceil(minimum);
  const last = Math.floor(maximum);
  const span = Math.max(0, last - first);
  const step = Math.max(1, Math.ceil((span + 1) / maximumTicks));
  const ticks = [];
  for (let value = first; value <= last; value += step) ticks.push(value);
  return ticks;
}

function standardRooflinePerformance(
  intensity,
  bandwidthBytesPerSecond,
  peakFlopsPerSecond,
) {
  return Math.min(bandwidthBytesPerSecond * intensity, peakFlopsPerSecond);
}

function drawRooflineMark(context, point, x, y, radius, color) {
  context.fillStyle = color;
  context.strokeStyle = color;
  context.lineWidth = 1.5;
  context.beginPath();
  if (point.bottleneck === "hbm") {
    context.rect(x - radius, y - radius, radius * 2, radius * 2);
  } else if (point.bottleneck === "communication") {
    context.moveTo(x, y - radius - 1);
    context.lineTo(x + radius + 1, y + radius);
    context.lineTo(x - radius - 1, y + radius);
    context.closePath();
  } else {
    context.arc(x, y, radius, 0, Math.PI * 2);
  }
  context.globalAlpha = 0.82;
  context.fill();
  context.globalAlpha = 1;
  context.stroke();
}

function renderRooflineChart() {
  const result = resultById($("#roofline-hardware")?.value);
  const canvas = $("#roofline-chart");
  if (!result || !canvas || canvas.hidden) return;
  const rect = canvas.getBoundingClientRect();
  if (rect.width < 20) return;

  const data = collectRooflineData(result);
  if (!data.points.length) return;
  const key = rooflineViewKey(result);
  if (!state.rooflineViews.has(key)) {
    state.rooflineViews.set(key, automaticRooflineView(result, data.points));
  }
  const view = state.rooflineViews.get(key);
  const width = rect.width;
  const height = width < 640 ? 360 : 430;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.style.height = `${height}px`;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const context = canvas.getContext("2d");
  context.scale(dpr, dpr);

  const css = getComputedStyle(document.documentElement);
  const colors = {
    grid: css.getPropertyValue("--line-soft").trim(),
    frame: css.getPropertyValue("--line").trim(),
    text: css.getPropertyValue("--text-dim").trim(),
    strongText: css.getPropertyValue("--text-muted").trim(),
    compute: css.getPropertyValue("--purple").trim(),
    hbm: css.getPropertyValue("--cyan").trim(),
    communication: css.getPropertyValue("--orange").trim(),
    expert: css.getPropertyValue("--accent").trim(),
  };
  const padding = {
    top: 30,
    right: 20,
    bottom: 58,
    left: width < 640 ? 60 : 72,
  };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const xRange = view.xMax - view.xMin;
  const yRange = view.yMax - view.yMin;
  const xScale = (logValue) =>
    padding.left + ((logValue - view.xMin) / xRange) * plotWidth;
  const yScale = (logValue) =>
    padding.top + ((view.yMax - logValue) / yRange) * plotHeight;

  context.clearRect(0, 0, width, height);
  context.font = "9px SFMono-Regular, Consolas, monospace";
  context.textBaseline = "middle";
  context.lineWidth = 1;

  for (const tick of logarithmicTicks(view.xMin, view.xMax)) {
    const x = xScale(tick) + 0.5;
    context.strokeStyle = colors.grid;
    context.beginPath();
    context.moveTo(x, padding.top);
    context.lineTo(x, padding.top + plotHeight);
    context.stroke();
    context.fillStyle = colors.text;
    context.textAlign = "center";
    context.fillText(formatIntensity(10 ** tick).replace(" FLOP/B", ""), x, height - 37);
  }
  for (const tick of logarithmicTicks(view.yMin, view.yMax)) {
    const y = yScale(tick) + 0.5;
    context.strokeStyle = colors.grid;
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(padding.left + plotWidth, y);
    context.stroke();
    context.fillStyle = colors.text;
    context.textAlign = "right";
    context.fillText(formatFlopsPerSecond(10 ** tick), padding.left - 8, y);
  }
  context.strokeStyle = colors.frame;
  context.strokeRect(padding.left + 0.5, padding.top + 0.5, plotWidth, plotHeight);

  context.fillStyle = colors.strongText;
  context.textAlign = "center";
  context.fillText("Arithmetic intensity · FLOP / HBM byte", padding.left + plotWidth / 2, height - 13);
  context.save();
  context.translate(13, padding.top + plotHeight / 2);
  context.rotate(-Math.PI / 2);
  context.fillText("Effective per-rank throughput · FLOP/s", 0, 0);
  context.restore();

  const bandwidth = finite(result.hardware.hbm_bandwidth_bytes_per_s);
  const densePeak = finite(result.hardware.dense_bf16_flops_per_s);
  const expertPeak = finite(result.hardware.k3_expert_flops_per_s);
  context.save();
  context.beginPath();
  context.rect(padding.left, padding.top, plotWidth, plotHeight);
  context.clip();

  const distinctExpertRoof =
    Math.abs(expertPeak - densePeak) / Math.max(densePeak, 1) > 0.01;
  const rooflines = [
    ...(distinctExpertRoof
      ? [{ label: "Expert compute peak", value: expertPeak, color: colors.expert, dashed: true }]
      : []),
    {
      label: distinctExpertRoof ? "BF16 compute peak" : "Compute peak",
      value: densePeak,
      color: colors.compute,
    },
  ];
  for (const roof of rooflines) {
    const ridgeLogX = Math.log10(roof.value / bandwidth);
    const logXValues = [view.xMin];
    if (ridgeLogX > view.xMin && ridgeLogX < view.xMax) {
      logXValues.push(ridgeLogX);
    }
    logXValues.push(view.xMax);
    context.strokeStyle = roof.color;
    context.lineWidth = 1.8;
    context.setLineDash(roof.dashed ? [6, 4] : []);
    context.beginPath();
    for (const [index, logX] of logXValues.entries()) {
      const performance = standardRooflinePerformance(
        10 ** logX,
        bandwidth,
        roof.value,
      );
      const x = xScale(logX);
      const y = yScale(Math.log10(performance));
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
    context.stroke();
  }
  context.setLineDash([]);

  state.rooflineHits = [];
  const maxCount = Math.max(...data.points.map((point) => point.count), 1);
  for (const point of data.points) {
    const logX = Math.log10(point.intensity);
    const logY = Math.log10(point.performance);
    if (
      logX < view.xMin ||
      logX > view.xMax ||
      logY < view.yMin ||
      logY > view.yMax
    ) {
      continue;
    }
    const x = xScale(logX);
    const y = yScale(logY);
    const radius = 3.5 + 3.5 * Math.sqrt(point.count / maxCount);
    drawRooflineMark(
      context,
      point,
      x,
      y,
      radius,
      colors[point.bottleneck] || colors.strongText,
    );
    state.rooflineHits.push({ x, y, radius: Math.max(radius, 7), point });
  }
  context.restore();

  const lowestRidgeLogX = Math.log10(
    Math.min(...rooflines.map((roof) => roof.value)) / bandwidth,
  );
  const memoryLabelXLog = Math.min(
    view.xMin + xRange * 0.12,
    lowestRidgeLogX - 0.12,
  );
  const memoryLabelYLog = Math.log10(
    standardRooflinePerformance(
      10 ** memoryLabelXLog,
      bandwidth,
      Math.min(...rooflines.map((roof) => roof.value)),
    ),
  );
  if (memoryLabelYLog > view.yMin && memoryLabelYLog < view.yMax) {
    context.fillStyle = colors.strongText;
    context.textAlign = "left";
    context.fillText(
      `Memory-bandwidth roof · ${formatBytes(bandwidth)}/s`,
      xScale(memoryLabelXLog) + 5,
      yScale(memoryLabelYLog) - 9,
    );
  }
  for (const roof of rooflines) {
    const logY = Math.log10(roof.value);
    if (logY <= view.yMin || logY >= view.yMax) continue;
    context.fillStyle = roof.color;
    context.textAlign = "right";
    const label = `${roof.label} · ${formatFlopsPerSecond(roof.value)}`;
    context.fillText(label, padding.left + plotWidth - 7, yScale(logY) - 9);
  }

  state.rooflineLayout = {
    key,
    result,
    data,
    padding,
    plotWidth,
    plotHeight,
    view: { ...view },
  };
  canvas.setAttribute(
    "aria-label",
    `${hardwareMeta(result).short} ${result.workload.phase} roofline with ${data.points.length} grouped arithmetic operators. Drag to pan and use the wheel to zoom.`,
  );
}

function renderRoofline() {
  const result = resultById($("#roofline-hardware")?.value);
  if (!result) return;
  const data = collectRooflineData(result);
  const phase = result.workload.phase === "prefill" ? "Prefill" : "Decode";
  const hardware = result.hardware;
  const expertParallelism = expertParallelismLabel(hardware);
  $("#roofline-summary").textContent =
    `${phase} · global TP${hardware.tp_size} · attention TP${hardware.attention_tp_size} · ${expertParallelism} · ${formatNumber(data.points.length)} grouped points`;

  const topologyRows = [
    ["Global tensor parallel", `TP${hardware.tp_size}`],
    ["Attention tensor parallel", `TP${hardware.attention_tp_size}`],
    ["Attention data parallel", `DP${hardware.attention_dp_size}`],
    ["Expert parallel", expertParallelism],
    [
      "MoE sharding",
      hardware.moe_sharding === "ep" ? "Expert parallel" : "Tensor parallel",
    ],
    ["Routed experts / rank", formatNumber(hardware.local_routed_experts)],
    ["MoE exchange", hardware.moe_a2a_backend || "No token A2A"],
    ["Recipe status", hardware.recipe_status],
  ];
  const ledger = result.parallel_work_ledger;
  if (ledger) {
    topologyRows.push(
      ["DP real requests", ledger.dp_real_requests.join(", ")],
      ["DP MLP-aligned rows", ledger.dp_mlp_aligned_rows.join(", ")],
      ["DP execution rows", ledger.dp_model_rows.join(", ")],
      [
        "DP padding mode",
        PADDING_MODE_LABELS[ledger.dp_padding_mode] || humanize(ledger.dp_padding_mode),
      ],
      ["Global MoE source rows", formatNumber(ledger.global_model_rows)],
      ["Routed pair instances", formatNumber(ledger.routed_pair_instances)],
      ["Critical sent pairs / source rank", formatNumber(ledger.critical_sent_pairs_per_source_rank)],
      ["Balanced received pairs / EP rank", formatNumber(ledger.balanced_received_pairs_per_ep_rank)],
      ["Scenario assumptions", ledger.bound_condition],
      ["Topology contract", ledger.topology_contract],
      [
        "Excluded positive terms",
        ledger.excluded_positive_term_ids
          .map((term) => EXCLUDED_TERM_LABELS[term] || humanize(term))
          .join(" · "),
      ],
    );
  }
  const sourceLinks = (hardware.sources || [])
    .filter((source) => /^https?:\/\//.test(source.url || ""))
    .map(
      (source) => `<a href="${escapeHtml(source.url)}" target="_blank" rel="noreferrer">${escapeHtml(source.title)} ↗</a>`,
    )
    .join(" · ");
  $("#roofline-topology").innerHTML = topologyRows
    .map(
      ([label, value]) => `
        <div class="roofline-topology-row">
          <span>${escapeHtml(label)}</span>
          <strong>${escapeHtml(value)}</strong>
        </div>`,
    )
    .join("") + (sourceLinks ? `
      <div class="roofline-topology-row">
        <span>Recipe evidence</span>
        <strong>${sourceLinks}</strong>
      </div>` : "");

  $("#roofline-boundary-summary").innerHTML = ["compute", "hbm", "communication"]
    .map(
      (bound) => `
        <div class="roofline-bound-row ${bound}">
          <span><i></i>${escapeHtml(boundLabel(bound))}</span>
          <strong>${formatNumber(data.boundCounts[bound])}</strong>
        </div>`,
    )
    .join("");

  const maximumCommunication = Math.max(
    ...data.communicationOnly.map((operation) => operation.seconds),
    1e-12,
  );
  $("#roofline-communication-list").innerHTML = data.communicationOnly.length
    ? data.communicationOnly
        .slice(0, 6)
        .map(
          (operation) => `
            <div class="roofline-communication-row">
              <span title="${escapeHtml(operation.name)}">${escapeHtml(operation.name)}</span>
              <strong>${escapeHtml(formatDuration(operation.seconds))}</strong>
              <span class="micro-bar"><i style="--bar-width:${(operation.seconds / maximumCommunication) * 100}%;--bar-color:${FLOOR_COLORS.communication}"></i></span>
              <small>${formatNumber(operation.count)}× · ${escapeHtml(formatBytes(operation.logicalBytes))} logical</small>
            </div>`,
        )
        .join("")
    : '<p class="roofline-empty">No communication-only operators.</p>';
  requestAnimationFrame(renderRooflineChart);
}

function showRooflineTooltip(event) {
  const canvas = $("#roofline-chart");
  const tooltip = $("#roofline-tooltip");
  if (!canvas || !tooltip || state.rooflineDrag) return;
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  let nearest = null;
  let nearestDistance = Infinity;
  for (const hit of state.rooflineHits) {
    const distance = Math.hypot(x - hit.x, y - hit.y);
    if (distance <= hit.radius + 7 && distance < nearestDistance) {
      nearest = hit;
      nearestDistance = distance;
    }
  }
  if (!nearest) {
    tooltip.hidden = true;
    return;
  }
  const point = nearest.point;
  tooltip.innerHTML = `
    <strong>${escapeHtml(point.name)}</strong>
    <span>${escapeHtml(point.stage)} · ${formatNumber(point.count)} occurrence${point.count === 1 ? "" : "s"}</span>
    <span>${escapeHtml(formatIntensity(point.intensity))} · ${escapeHtml(formatFlopsPerSecond(point.performance))}</span>
    <span>${escapeHtml(boundLabel(point.bottleneck))} · C ${escapeHtml(formatDuration(point.computeSeconds, true))} · HBM ${escapeHtml(formatDuration(point.hbmSeconds, true))} · Comm ${escapeHtml(formatDuration(point.communicationSeconds, true))}</span>`;
  tooltip.style.left = `${Math.max(105, Math.min(rect.width - 105, nearest.x))}px`;
  tooltip.style.top = `${Math.max(88, nearest.y)}px`;
  tooltip.hidden = false;
}

function resetRooflineView() {
  const result = resultById($("#roofline-hardware")?.value);
  if (!result) return;
  state.rooflineViews.delete(rooflineViewKey(result));
  state.rooflineDrag = null;
  $("#roofline-chart")?.classList.remove("is-dragging");
  $("#roofline-tooltip").hidden = true;
  renderRooflineChart();
}

function startRooflinePan(event) {
  const layout = state.rooflineLayout;
  const canvas = $("#roofline-chart");
  if (!layout || !canvas || event.button !== 0) return;
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const { padding, plotWidth, plotHeight } = layout;
  if (
    x < padding.left ||
    x > padding.left + plotWidth ||
    y < padding.top ||
    y > padding.top + plotHeight
  ) {
    return;
  }
  canvas.setPointerCapture(event.pointerId);
  canvas.classList.add("is-dragging");
  $("#roofline-tooltip").hidden = true;
  state.rooflineDrag = {
    pointerId: event.pointerId,
    startX: event.clientX,
    startY: event.clientY,
    view: { ...layout.view },
    plotWidth,
    plotHeight,
    key: layout.key,
  };
}

function moveRooflinePointer(event) {
  const drag = state.rooflineDrag;
  if (!drag || drag.pointerId !== event.pointerId) {
    showRooflineTooltip(event);
    return;
  }
  const xRange = drag.view.xMax - drag.view.xMin;
  const yRange = drag.view.yMax - drag.view.yMin;
  const xShift = -((event.clientX - drag.startX) / drag.plotWidth) * xRange;
  const yShift = ((event.clientY - drag.startY) / drag.plotHeight) * yRange;
  state.rooflineViews.set(drag.key, {
    xMin: drag.view.xMin + xShift,
    xMax: drag.view.xMax + xShift,
    yMin: drag.view.yMin + yShift,
    yMax: drag.view.yMax + yShift,
  });
  renderRooflineChart();
}

function endRooflinePan(event) {
  const drag = state.rooflineDrag;
  if (!drag || drag.pointerId !== event.pointerId) return;
  const canvas = $("#roofline-chart");
  if (canvas?.hasPointerCapture(event.pointerId)) {
    canvas.releasePointerCapture(event.pointerId);
  }
  canvas?.classList.remove("is-dragging");
  state.rooflineDrag = null;
}

function zoomRoofline(event) {
  const layout = state.rooflineLayout;
  const canvas = $("#roofline-chart");
  if (!layout || !canvas) return;
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const { padding, plotWidth, plotHeight, view } = layout;
  if (
    x < padding.left ||
    x > padding.left + plotWidth ||
    y < padding.top ||
    y > padding.top + plotHeight
  ) {
    return;
  }
  event.preventDefault();
  const factor = Math.exp(Math.max(-400, Math.min(400, event.deltaY)) * 0.0015);
  const oldXRange = view.xMax - view.xMin;
  const oldYRange = view.yMax - view.yMin;
  const newXRange = Math.max(0.6, Math.min(10, oldXRange * factor));
  const newYRange = Math.max(0.6, Math.min(10, oldYRange * factor));
  const xRatio = (x - padding.left) / plotWidth;
  const yRatio = (padding.top + plotHeight - y) / plotHeight;
  const anchorX = view.xMin + xRatio * oldXRange;
  const anchorY = view.yMin + yRatio * oldYRange;
  state.rooflineViews.set(layout.key, {
    xMin: anchorX - xRatio * newXRange,
    xMax: anchorX + (1 - xRatio) * newXRange,
    yMin: anchorY - yRatio * newYRange,
    yMax: anchorY + (1 - yRatio) * newYRange,
  });
  $("#roofline-tooltip").hidden = true;
  renderRooflineChart();
}

function renderLayerAnalysis() {
  state.expandedLayers.clear();
  state.expandedOperations.clear();
  renderLayerTable();
  requestAnimationFrame(renderLayerChart);
}

function layerKey(result, index) {
  return `${result.hardware.id}:${index}`;
}

function layerPathLabel(layer) {
  const pieces = [];
  if (layer.attention) pieces.push(layer.attention.toUpperCase());
  if (layer.ffn) pieces.push(layer.ffn === "moe" ? "MoE" : "Dense FFN");
  return pieces.length ? pieces.join(" · ") : humanize(layer.name);
}

function operationDisplayValue(operation, key) {
  if (key === "flops_per_rank") return formatFlops(operation[key], true);
  if (key.endsWith("bytes_per_rank") || key === "logical_collective_bytes") {
    return formatBytes(operation[key], true);
  }
  return formatDuration(operation[key], true);
}

function calculationText(value, fallback = "Not provided by analyzer") {
  if (value === null || value === undefined || value === "") return fallback;
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function calculationTrace(operation, layerIndex, operationIndex, expanded = false) {
  const calculations =
    operation.calculations && typeof operation.calculations === "object"
      ? operation.calculations
      : {};
  const items = OPERATION_CALCULATION_FIELDS.map(({ key, label }) => {
    const calculation =
      calculations[key] && typeof calculations[key] === "object"
        ? calculations[key]
        : {};
    const result = calculationText(calculation.result, calculationText(operation[key], "—"));
    const unit = calculationText(calculation.unit ?? calculation.units, "");
    const note = calculationText(calculation.note, "");
    return `
      <article class="calculation-card">
        <header>
          <div>
            <strong>${escapeHtml(calculationText(calculation.label, label))}</strong>
            <code>${escapeHtml(key)}</code>
          </div>
          <span class="calculation-shown" title="Value shown in the operator table">${escapeHtml(operationDisplayValue(operation, key))}</span>
        </header>
        <dl class="calculation-steps">
          <div>
            <dt>Formula</dt>
            <dd><code>${escapeHtml(calculationText(calculation.formula))}</code></dd>
          </div>
          <div>
            <dt>Substitute</dt>
            <dd><code>${escapeHtml(calculationText(calculation.substitution))}</code></dd>
          </div>
          <div class="calculation-result">
            <dt>Result</dt>
            <dd><code>= ${escapeHtml(result)}${unit ? ` ${escapeHtml(unit)}` : ""}</code></dd>
          </div>
        </dl>
        ${note ? `<p>${escapeHtml(note)}</p>` : ""}
      </article>`;
  }).join("");
  return `
    <tr class="operation-calculation-row" id="operator-calculation-${layerIndex}-${operationIndex}"${expanded ? "" : " hidden"}>
      <td colspan="11">
        <section class="calculation-trace" aria-label="${escapeHtml(operation.name)} calculation trace">
          <div class="calculation-trace-head">
            <div>
              <strong>Calculation trace</strong>
              <span>Logical payload is separate from per-fabric traffic; communication uses the maximum of topology-resolved independent fabric floors.</span>
            </div>
            <span class="calculation-count">${formatNumber(OPERATION_CALCULATION_FIELDS.length)} fields</span>
          </div>
          <div class="calculation-grid">${items}</div>
        </section>
      </td>
    </tr>`;
}

function renderLayerTable() {
  const result = resultById($("#layer-hardware").value);
  if (!result) return;
  const filter = $("#layer-filter").value.trim().toLowerCase();
  const maxLatency = Math.max(...result.layers.map((layer) => finite(layer.latency_seconds)), 1e-12);
  const rows = [];
  result.layers.forEach((layer, index) => {
    const dominant = layer.operations.find((operation) => operation.id === layer.dominant_operation);
    const haystack = [
      layer.name,
      layer.number,
      layer.attention,
      layer.ffn,
      dominant?.name,
      ...layerCertificates(layer),
    ]
      .join(" ")
      .toLowerCase();
    if (filter && !haystack.includes(filter)) return;
    const key = layerKey(result, index);
    const expanded = state.expandedLayers.has(key);
    const pathClass = layer.attention || "other";
    const share = result.total_seconds ? layer.latency_seconds / result.total_seconds : 0;
    const barWidth = (layer.latency_seconds / maxLatency) * 100;
    rows.push(`
      <tr class="layer-row" data-layer-index="${index}" data-layer-key="${escapeHtml(key)}" aria-expanded="${expanded}">
        <td class="expand-cell"><button class="expand-button" type="button" aria-expanded="${expanded}" aria-controls="ops-${index}" aria-label="${expanded ? "Collapse" : "Expand"} ${escapeHtml(layer.name)} operators">${expanded ? "×" : "+"}</button></td>
        <td class="stage-name-cell">${escapeHtml(layer.number == null ? humanize(layer.name) : `Layer ${layer.number}`)}<small>${escapeHtml(layer.name)}</small></td>
        <td><span class="path-chip ${pathClass}">${escapeHtml(layerPathLabel(layer))}</span></td>
        <td class="numeric">${escapeHtml(formatDuration(layer.latency_seconds))}</td>
        <td class="numeric share-cell"><span class="share-inline"><span class="micro-bar"><i style="--bar-width:${barWidth}%;--bar-color:${pathClass === "kda" ? "var(--accent)" : pathClass === "mla" ? "var(--purple)" : "var(--text-dim)"}"></i></span><span>${escapeHtml(formatPercent(share, 2))}</span></span></td>
        <td>${escapeHtml(dominant?.name || layer.dominant_operation)}</td>
        <td><span class="certificate-chips">${certificateChips(layer)}</span></td>
      </tr>`);
    if (expanded) rows.push(operationTable(layer, index, key));
  });
  $("#layer-table-body").innerHTML = rows.join("");
  $("#layer-chart-summary").textContent = `${formatNumber(result.layers.length)} stages · ${formatDuration(result.total_seconds)} latency lower bound`;
}

function operationTable(layer, index, key) {
  const rows = layer.operations
    .map((operation, operationIndex) => {
      const notes = (operation.notes || []).filter(Boolean);
      const notesList = notes
        .map((note) => `<li>${escapeHtml(note)}</li>`)
        .join("");
      const calculationId = `operator-calculation-${index}-${operationIndex}`;
      const operationKey = `${key}:${operationIndex}`;
      const expanded = state.expandedOperations.has(operationKey);
      return `
        <tr class="operation-row${expanded ? " is-expanded" : ""}" data-operation-index="${operationIndex}" data-operation-key="${escapeHtml(operationKey)}">
          <td class="operation-name">
            <button class="operation-toggle" type="button" aria-expanded="${expanded}" aria-controls="${calculationId}" aria-label="${expanded ? "Hide" : "Show"} calculations for ${escapeHtml(operation.name)}">
              <span class="operation-chevron" aria-hidden="true">›</span>
              <span><strong>${escapeHtml(operation.name)}</strong><small>${escapeHtml(operation.id)}</small></span>
            </button>
          </td>
          <td><span class="path-chip other">${escapeHtml(operation.category)}</span></td>
          <td class="numeric">${escapeHtml(formatFlops(operation.flops_per_rank, true))}</td>
          <td class="numeric">${escapeHtml(formatBytes(operation.hbm_bytes_per_rank, true))}</td>
          <td class="numeric">${escapeHtml(formatBytes(operation.logical_collective_bytes, true))}</td>
          <td class="numeric">${escapeHtml(formatDuration(operation.compute_seconds, true))}</td>
          <td class="numeric">${escapeHtml(formatDuration(operation.hbm_seconds, true))}</td>
          <td class="numeric">${escapeHtml(formatDuration(operation.communication_seconds, true))}</td>
          <td class="numeric"><strong>${escapeHtml(formatDuration(operation.duration_seconds))}</strong></td>
          <td><span class="bottleneck-chip ${normalizeFloor(operation.bottleneck)}">${escapeHtml(boundLabel(operation.bottleneck))}</span></td>
          <td>${notes.length ? `
            <details class="notes-disclosure">
              <summary class="notes-button" aria-label="Show notes for ${escapeHtml(operation.name)}">i</summary>
              <ul class="notes-list">${notesList}</ul>
            </details>` : "—"}</td>
        </tr>
        ${calculationTrace(operation, index, operationIndex, expanded)}`;
    })
    .join("");
  return `
    <tr class="operation-detail-row" id="ops-${index}">
      <td colspan="7">
        <div class="operation-wrap">
          <div class="operation-head"><strong>${formatNumber(layer.operations.length)} operator rooflines</strong><span>Select an operator to inspect its math. Times can overlap; do not sum rows.</span></div>
          <div class="operation-table-scroll">
            <table class="data-table operation-table">
              <thead><tr><th scope="col">Operator</th><th scope="col">Category</th><th scope="col" class="numeric">FLOPs / rank</th><th scope="col" class="numeric">HBM / rank</th><th scope="col" class="numeric">Collective payload</th><th scope="col" class="numeric">Compute floor</th><th scope="col" class="numeric">HBM floor</th><th scope="col" class="numeric">Comm floor</th><th scope="col" class="numeric">Op roofline</th><th scope="col">Bottleneck</th><th scope="col">Notes</th></tr></thead>
              <tbody>${rows}</tbody>
            </table>
          </div>
        </div>
      </td>
    </tr>`;
}

function toggleOperationRow(row) {
  const button = row.querySelector(".operation-toggle");
  const detail = button ? document.getElementById(button.getAttribute("aria-controls")) : null;
  if (!button || !detail) return;
  const shouldExpand = button.getAttribute("aria-expanded") !== "true";
  button.setAttribute("aria-expanded", String(shouldExpand));
  button.setAttribute(
    "aria-label",
    `${shouldExpand ? "Hide" : "Show"} calculations for ${button.querySelector("strong")?.textContent || "operator"}`,
  );
  row.classList.toggle("is-expanded", shouldExpand);
  detail.hidden = !shouldExpand;
  const key = row.dataset.operationKey;
  if (key) {
    if (shouldExpand) state.expandedOperations.add(key);
    else state.expandedOperations.delete(key);
  }
}

function toggleLayerRow(row) {
  const result = resultById($("#layer-hardware").value);
  if (!result) return;
  const index = Number(row.dataset.layerIndex);
  const layer = result.layers[index];
  const key = row.dataset.layerKey;
  const isExpanded = row.getAttribute("aria-expanded") === "true";
  if (isExpanded) {
    row.nextElementSibling?.classList.contains("operation-detail-row") && row.nextElementSibling.remove();
    row.setAttribute("aria-expanded", "false");
    row.querySelector(".expand-button").setAttribute("aria-expanded", "false");
    row.querySelector(".expand-button").setAttribute("aria-label", `Expand ${layer.name} operators`);
    state.expandedLayers.delete(key);
    return;
  }
  row.insertAdjacentHTML("afterend", operationTable(layer, index, key));
  row.setAttribute("aria-expanded", "true");
  row.querySelector(".expand-button").setAttribute("aria-expanded", "true");
  row.querySelector(".expand-button").setAttribute("aria-label", `Collapse ${layer.name} operators`);
  state.expandedLayers.add(key);
}

function renderLayerChart() {
  const result = resultById($("#layer-hardware").value);
  const canvas = $("#layer-chart");
  if (!result || !canvas || canvas.hidden) return;
  const rect = canvas.getBoundingClientRect();
  if (rect.width < 20) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = rect.width;
  const height = 320;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const context = canvas.getContext("2d");
  context.scale(dpr, dpr);

  const css = getComputedStyle(document.documentElement);
  const colors = {
    grid: css.getPropertyValue("--line-soft").trim(),
    text: css.getPropertyValue("--text-dim").trim(),
    other: css.getPropertyValue("--text-dim").trim(),
    kda: css.getPropertyValue("--accent").trim(),
    mla: css.getPropertyValue("--purple").trim(),
  };
  const padding = { top: 17, right: 10, bottom: 35, left: 54 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const values = result.layers.map((layer) => finite(layer.latency_seconds));
  const maxValue = Math.max(...values, 1e-12) * 1.08;
  const unit = maxValue < 1e-6 ? { multiplier: 1e9, label: "ns" } : maxValue < 1e-3 ? { multiplier: 1e6, label: "µs" } : maxValue < 1 ? { multiplier: 1e3, label: "ms" } : { multiplier: 1, label: "s" };
  context.clearRect(0, 0, width, height);
  context.font = "9px SFMono-Regular, Consolas, monospace";
  context.textBaseline = "middle";
  context.lineWidth = 1;

  for (let tick = 0; tick <= 4; tick += 1) {
    const ratio = tick / 4;
    const y = padding.top + plotHeight - ratio * plotHeight + 0.5;
    context.strokeStyle = colors.grid;
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(width - padding.right, y);
    context.stroke();
    context.fillStyle = colors.text;
    context.textAlign = "right";
    context.fillText(`${formatNumber(maxValue * ratio * unit.multiplier, 2)} ${unit.label}`, padding.left - 8, y);
  }

  const slot = plotWidth / result.layers.length;
  const barWidth = Math.max(1.2, Math.min(slot * 0.68, 8));
  state.chartHits = [];
  result.layers.forEach((layer, index) => {
    const value = values[index];
    const barHeight = (value / maxValue) * plotHeight;
    const x = padding.left + index * slot + (slot - barWidth) / 2;
    const y = padding.top + plotHeight - barHeight;
    const kind = layer.attention || "other";
    context.fillStyle = colors[kind] || colors.other;
    context.globalAlpha = kind === "other" ? 0.7 : 0.9;
    context.fillRect(x, y, barWidth, Math.max(barHeight, 1));
    context.globalAlpha = 1;
    state.chartHits.push({ x: x - Math.max(2, (slot - barWidth) / 2), width: Math.max(slot, 5), y, layer, index });
  });

  context.fillStyle = colors.text;
  context.textAlign = "center";
  const labels = new Map([
    [0, "Emb"],
    [1, "L1"],
    [13, "L13"],
    [25, "L25"],
    [37, "L37"],
    [49, "L49"],
    [61, "L61"],
    [73, "L73"],
    [85, "L85"],
    [93, "L93"],
    [94, "Norm"],
    [95, "Head"],
  ]);
  for (const [index, label] of labels) {
    const x = padding.left + index * slot + slot / 2;
    context.fillText(label, x, height - 14);
  }
  canvas.setAttribute(
    "aria-label",
    `${hardwareMeta(result).short} latency lower bounds across ${result.layers.length} model stages. Peak stage lower bound ${formatDuration(Math.max(...values))}.`,
  );
}

function showChartTooltip(event) {
  const canvas = $("#layer-chart");
  const tooltip = $("#chart-tooltip");
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const hit = state.chartHits.find((item) => x >= item.x && x <= item.x + item.width);
  if (!hit) {
    tooltip.hidden = true;
    return;
  }
  const certificates = layerCertificates(hit.layer).map(certificateLabel).join(" + ");
  tooltip.innerHTML = `<strong>${escapeHtml(hit.layer.number == null ? humanize(hit.layer.name) : `Layer ${hit.layer.number} · ${layerPathLabel(hit.layer)}`)}</strong><span>${escapeHtml(formatDuration(hit.layer.latency_seconds))} latency lower bound · ${escapeHtml(certificates)}</span>`;
  const left = Math.max(80, Math.min(rect.width - 80, hit.x + hit.width / 2));
  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${Math.max(58, hit.y)}px`;
  tooltip.hidden = false;
}

function renderMemory() {
  const result = resultById($("#memory-hardware").value);
  if (!result) return;
  const memory = result.memory;
  const total = finite(memory.total_accounted_peak_bytes_per_rank);
  const capacity = finite(memory.nominal_hbm_capacity_bytes_per_rank);
  const ratio = capacity ? total / capacity : 0;
  $("#memory-total").textContent = formatBytes(total);
  $("#memory-capacity").textContent = `${formatBytes(capacity)} nominal`;
  const fit = $("#memory-fit");
  const fits = memory.fits_nominal_capacity;
  fit.className = `fit-badge ${fits === true ? "fits" : fits === false ? "over" : "unknown"}`;
  fit.textContent = fits === true
    ? `${formatPercent(ratio)} · accounted peak fits nominal`
    : fits === false
      ? `${formatPercent(ratio)} · accounted lower bound exceeds nominal`
      : `${formatPercent(ratio)} · fit inconclusive; MegaMoE workspace excluded`;
  const bar = $("#capacity-bar");
  bar.classList.toggle("over", ratio > 1);
  bar.style.setProperty("--capacity-width", `${Math.min(ratio * 100, 100)}%`);
  bar.setAttribute("aria-valuenow", String(Math.min(Math.round(ratio * 100), 100)));
  bar.setAttribute("aria-valuemax", "100");
  bar.setAttribute("aria-label", `${formatPercent(ratio)} of nominal HBM accounted`);

  const components = [
    ["Static weights", memory.static_weight_bytes_per_rank],
    ["KDA recurrent state", memory.kda_state_bytes_per_rank],
    ["MLA KV cache", memory.mla_kv_cache_bytes_per_rank],
    ["Attention residual bank", memory.attention_residual_bank_bytes_per_rank],
  ];
  $("#memory-components").innerHTML = components
    .map(([label, value], index) => `<div class="memory-component" style="--item-color:${MEMORY_COLORS[index]}"><span>${escapeHtml(label)}</span><strong>${escapeHtml(formatBytes(value))}</strong></div>`)
    .join("");

  const weights = Object.entries(memory.weight_breakdown_bytes_per_rank || {}).sort((a, b) => b[1] - a[1]);
  const maxWeight = Math.max(...weights.map(([, value]) => finite(value)), 1);
  $("#weight-list").innerHTML = weights
    .map(([label, value], index) => `
      <div class="weight-row">
        <span title="${escapeHtml(humanize(label))}">${escapeHtml(humanize(label))}</span>
        <span class="micro-bar"><i style="--bar-width:${(finite(value) / maxWeight) * 100}%;--bar-color:${MEMORY_COLORS[index % MEMORY_COLORS.length]}"></i></span>
        <strong>${escapeHtml(formatBytes(value))}</strong>
      </div>`)
    .join("");
}

function renderManifest() {
  const model = state.response?.model || state.manifest?.model || {};
  const kdaCount = Array.isArray(model.kda_layers) ? model.kda_layers.length : 69;
  const mlaCount = Array.isArray(model.full_attention_layers) ? model.full_attention_layers.length : 24;
  const facts = [
    ["Decoder layers", model.num_hidden_layers ?? 93],
    ["Hidden width", model.hidden_size ?? 7168],
    ["KDA / MLA", `${kdaCount} / ${mlaCount}`],
    ["Attention heads", `${model.num_attention_heads ?? 96} × ${model.head_dim ?? 128}`],
    ["Routed experts", model.num_experts ?? 896],
    ["Experts / token", model.num_experts_per_token ?? 16],
    ["Shared experts", model.num_shared_experts ?? 2],
    ["Routed latent", model.routed_expert_hidden_size ?? 3584],
    ["Expert intermediate", model.moe_intermediate_size ?? 3072],
    ["Q / KV LoRA rank", `${model.q_lora_rank ?? 1536} / ${model.kv_lora_rank ?? 512}`],
    ["Vocabulary", formatNumber(model.vocab_size ?? 163840)],
    ["Max positions", formatNumber(model.max_position_embeddings ?? 1048576)],
  ];
  const sources = Array.isArray(model.sources) ? model.sources : [];
  const sourceHtml = sources
    .filter((source) => /^https?:\/\//.test(source.url || ""))
    .map((source) => `<li><a href="${escapeHtml(source.url)}" target="_blank" rel="noreferrer"><span>${escapeHtml(source.title)}</span><span aria-hidden="true">↗</span></a></li>`)
    .join("");
  $("#manifest-content").innerHTML = `
    <div class="manifest-facts">
      ${facts.map(([label, value]) => `<div class="manifest-fact"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("")}
    </div>
    <div class="manifest-side">
      <h4>Pinned checkpoint revision</h4>
      <code title="${escapeHtml(model.revision || state.results[0]?.model_revision || "")}">${escapeHtml(model.revision || state.results[0]?.model_revision || "unknown")}</code>
      ${sourceHtml ? `<ul class="source-list">${sourceHtml}</ul>` : ""}
    </div>`;
}

function exportJson() {
  if (!state.response) return;
  const blob = new Blob([`${JSON.stringify(state.response, null, 2)}\n`], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  const workload = state.results[0]?.workload;
  link.href = url;
  link.download = `k3-${workload?.phase || "estimate"}-${new Date().toISOString().slice(0, 10)}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function bindEvents() {
  $("#theme-toggle").addEventListener("click", () => {
    setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  });
  $$('input[name="phase"]').forEach((input) =>
    input.addEventListener("change", () => {
      updatePhaseFields();
      scheduleCalculate();
    }),
  );
  $$('input[name="hardware"]').forEach((input) =>
    input.addEventListener("change", () => {
      updateHardwareState();
      scheduleCalculate();
    }),
  );
  $$('input[name="sharding"]').forEach((input) =>
    input.addEventListener("change", () => {
      updateShardingCompatibility();
      updateSequenceLimit();
      scheduleCalculate();
    }),
  );
  for (const id of ["sequence-length", "batch-size", "context-length"]) {
    $(`#${id}`).addEventListener("input", () => scheduleCalculate());
  }
  $("#calculator-form").addEventListener("submit", (event) => {
    event.preventDefault();
    scheduleCalculate({ immediate: true });
  });
  $("#retry-button").addEventListener("click", () => scheduleCalculate({ immediate: true }));
  $("#export-json").addEventListener("click", exportJson);
  const analysisHardwareSelectors = [
    "breakdown-hardware",
    "roofline-hardware",
    "layer-hardware",
    "memory-hardware",
  ];
  for (const id of analysisHardwareSelectors) {
    $(`#${id}`).addEventListener("change", (event) => {
      for (const otherId of analysisHardwareSelectors) {
        $(`#${otherId}`).value = event.target.value;
      }
      renderBreakdown();
      renderRoofline();
      renderLayerAnalysis();
      renderMemory();
    });
  }
  $("#layer-filter").addEventListener("input", renderLayerTable);
  $("#layer-table-body").addEventListener("click", (event) => {
    const layerButton = event.target.closest(".expand-button");
    const layerRow = layerButton?.closest(".layer-row");
    if (layerRow) {
      toggleLayerRow(layerRow);
      return;
    }
    const operationRow = event.target.closest(".operation-row");
    if (!operationRow) return;
    const interactiveElement = event.target.closest(
      "button, a, input, select, textarea, details, summary",
    );
    if (interactiveElement && !interactiveElement.classList.contains("operation-toggle")) return;
    toggleOperationRow(operationRow);
  });
  $("#layer-chart").addEventListener("mousemove", showChartTooltip);
  $("#layer-chart").addEventListener("mouseleave", () => {
    $("#chart-tooltip").hidden = true;
  });
  const rooflineCanvas = $("#roofline-chart");
  rooflineCanvas.addEventListener("pointerdown", startRooflinePan);
  rooflineCanvas.addEventListener("pointermove", moveRooflinePointer);
  rooflineCanvas.addEventListener("pointerup", endRooflinePan);
  rooflineCanvas.addEventListener("pointercancel", endRooflinePan);
  rooflineCanvas.addEventListener("pointerleave", () => {
    if (!state.rooflineDrag) $("#roofline-tooltip").hidden = true;
  });
  rooflineCanvas.addEventListener("wheel", zoomRoofline, { passive: false });
  rooflineCanvas.addEventListener("dblclick", resetRooflineView);
  $("#roofline-reset").addEventListener("click", resetRooflineView);
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (state.results.length) {
        renderRooflineChart();
        renderLayerChart();
      }
    }, 120);
  });
}

async function init() {
  initTheme();
  bindEvents();
  updatePhaseFields();
  updateHardwareState();
  setView("loading");
  const connected = await loadManifest();
  state.ready = connected;
  if (connected) calculate();
}

if (typeof module === "object" && module.exports) {
  module.exports = { normalizeFloor, validateCalculatorResponse };
} else {
  document.addEventListener("DOMContentLoaded", init);
}
