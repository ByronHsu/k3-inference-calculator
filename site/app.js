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
  dependency: "#6aa6ff",
  compute: "#a88cff",
  hbm: "#5ed6d2",
  communication: "#ffad5c",
};

const MEMORY_COLORS = ["#b8f455", "#a88cff", "#5ed6d2", "#ffad5c", "#6aa6ff"];
const AUTO_CALCULATE_DELAY_MS = 300;

const OPERATION_CALCULATION_FIELDS = [
  { key: "flops_per_rank", label: "FLOPs / rank" },
  { key: "hbm_bytes_per_rank", label: "HBM bytes / rank" },
  { key: "logical_collective_bytes", label: "Logical collective bytes" },
  { key: "link_bytes_per_rank", label: "Physical link bytes / rank" },
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
  const lower = String(value || "dependency").toLowerCase();
  if (lower.includes("comm")) return "communication";
  if (lower.includes("hbm") || lower.includes("memory")) return "hbm";
  if (lower.includes("compute")) return "compute";
  return "dependency";
}

function parallelismLabel(hardware) {
  return `TP${hardware.tp_size}`;
}

function setView(view) {
  const welcome = $("#welcome-state");
  const loading = $("#loading-state");
  const error = $("#error-state");
  const results = $("#results");
  welcome.hidden = view !== "welcome";
  loading.hidden = view !== "loading";
  error.hidden = view !== "error";
  results.hidden = view !== "results";
  $("#report").setAttribute("aria-busy", view === "loading" ? "true" : "false");
  $("#calculation-status").textContent =
    {
      welcome: "Calculator ready.",
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
  if (state.results.length) requestAnimationFrame(renderLayerChart);
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

function selectedTpSize() {
  return Number($("input[name='tp-size']:checked")?.value);
}

function updateHardwareState() {
  const boxes = $$("input[name='hardware']");
  const checked = boxes.filter((box) => box.checked);
  const all = checked.length === boxes.length;
  $("#select-all-hardware").textContent = all ? "Clear all" : "Select all";
  applyTpAvailability();
  updateSequenceLimit();
}

function applyTpAvailability() {
  const selected = selectedHardware();
  const invalid = Array.isArray(state.manifest?.invalid_combinations)
    ? state.manifest.invalid_combinations
    : [];
  for (const input of $$("input[name='tp-size']")) {
    const tpSize = Number(input.value);
    const manifestInvalid = invalid.some((combination) => {
      if (typeof combination === "string") {
        return combination.toLowerCase().includes(`tp${tpSize}`);
      }
      if (!combination || Number(combination.tp_size) !== tpSize) return false;
      const familyValue =
        combination.hardware || combination.hardware_family || combination.family || [];
      const families = (Array.isArray(familyValue) ? familyValue : [familyValue]).map(
        (family) => String(family).toLowerCase(),
      );
      return !families.length || families.some((family) => selected.includes(family));
    });
    input.disabled = tpSize === 64 || manifestInvalid;
  }
  const checked = $("input[name='tp-size']:checked");
  if (checked?.disabled) {
    const fallback = $("input[name='tp-size']:not(:disabled)");
    if (fallback) fallback.checked = true;
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
  const tpSize = selectedTpSize();
  if (![8, 16, 32].includes(tpSize)) {
    return "Select a supported tensor-parallel size: TP8, TP16, or TP32.";
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
  const payload = {
    phase,
    hardware: selectedHardware(),
    tp_size: selectedTpSize(),
    batch_size: phase === "prefill" ? 1 : Number($("#batch-size").value),
  };
  if (phase === "prefill") payload.sequence_length = Number($("#sequence-length").value);
  else payload.context_length = Number($("#context-length").value);
  return payload;
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

  try {
    const response = await calculatorFetch("./api/calculate", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(requestPayload()),
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
    const results = Array.isArray(body?.results)
      ? body.results
      : Array.isArray(body)
        ? body
        : body?.hardware
          ? [body]
          : [];
    if (!results.length) throw new Error("The calculator returned no hardware results.");
    if (results.some((result) => !Array.isArray(result.layers) || !result.hardware)) {
      throw new Error("The calculator response is missing its layer or hardware inventory.");
    }
    state.response = body;
    state.results = results;
    state.manifest = { ...(state.manifest || {}), ...body, results: undefined };
    state.expandedLayers.clear();
    state.expandedOperations.clear();
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
      applyTpAvailability();
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
      : `batch ${formatNumber(first.workload.batch_size)}${first.workload.model_batch_size !== first.workload.batch_size ? ` → ${formatNumber(first.workload.model_batch_size)} graph rows` : ""} · ${formatNumber(first.workload.context_length)} context · ${first.decode_cuda_graph_replay ? "CUDA graph replay" : "eager"}`;
  const status = state.response?.analytical_status || state.manifest?.analytical_status || "optimistic lower bound; not measured";
  $("#report-status").textContent = `${status.charAt(0).toUpperCase()}${status.slice(1)}.`;
  const time = new Date();
  $("#generated-time").dateTime = time.toISOString();
  $("#generated-time").textContent = time.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  renderComparison();
  for (const id of ["breakdown-hardware", "layer-hardware", "memory-hardware"]) {
    populateHardwareSelect($(`#${id}`));
  }
  renderBreakdown();
  renderLayerAnalysis();
  renderMemory();
  renderManifest();
}

function renderComparison() {
  const cards = $("#comparison-cards");
  const fittingResults = state.results.filter(
    (result) => result.memory.fits_nominal_capacity,
  );
  const fastest = Math.min(
    ...fittingResults.map((result) => finite(result.total_seconds, Infinity)),
  );
  cards.style.setProperty("--comparison-columns", String(Math.min(state.results.length, 3)));
  cards.innerHTML = state.results
    .map((result) => {
      const meta = hardwareMeta(result);
      const parts = durationParts(result.total_seconds);
      const fits = result.memory.fits_nominal_capacity;
      const isFastest = fits && Math.abs(result.total_seconds - fastest) <= fastest * 1e-9;
      const ratio = result.total_seconds / fastest;
      const memoryRatio = result.memory.total_accounted_peak_bytes_per_rank / result.memory.nominal_hbm_capacity_bytes_per_rank;
      return `
        <article class="comparison-card ${isFastest ? "fastest" : ""} ${fits ? "" : "does-not-fit"}" style="--card-color:${meta.color}">
          <div class="card-system-row">
            <span class="system-name"><i class="system-dot"></i><strong>${escapeHtml(meta.short)} · ${escapeHtml(parallelismLabel(result.hardware))}</strong></span>
            ${!fits ? '<span class="capacity-warning-badge">Does not fit</span>' : isFastest ? '<span class="winner-badge">Fastest runnable</span>' : ""}
          </div>
          <div>
            <div class="card-latency">${escapeHtml(parts.value)} <small>${escapeHtml(parts.unit)}</small></div>
            <div class="card-subline">${!fits ? "Capacity-infeasible analytical result" : state.results.length === 1 ? "Selected configuration" : isFastest ? "Fastest selected runnable family" : `${formatNumber((ratio - 1) * 100, 1)}% slower than fastest runnable result`}</div>
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
          <td><span class="fit-badge ${result.memory.fits_nominal_capacity ? "fits" : "over"}">${result.memory.fits_nominal_capacity ? "Fits nominal" : "Does not fit nominal HBM"}</span></td>
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

  const floorCounts = { dependency: 0, compute: 0, hbm: 0, communication: 0 };
  for (const layer of result.layers) floorCounts[normalizeFloor(layer.limiting_floor)] += 1;
  const maxCount = Math.max(1, ...Object.values(floorCounts));
  $("#floor-chart").innerHTML = Object.entries(floorCounts)
    .map(([floor, count]) => `
      <div class="floor-row">
        <span>${escapeHtml(floor)}</span>
        <span class="micro-bar"><i style="--bar-width:${(count / maxCount) * 100}%;--bar-color:${FLOOR_COLORS[floor]}"></i></span>
        <strong>${formatNumber(count)} / ${formatNumber(result.layers.length)}</strong>
      </div>`)
    .join("");

  const resources = {
    dependency: result.layers.reduce((sumValue, layer) => sumValue + finite(layer.dependency_path_seconds), 0),
    compute: result.layers.reduce((sumValue, layer) => sumValue + finite(layer.compute_resource_seconds), 0),
    hbm: result.layers.reduce((sumValue, layer) => sumValue + finite(layer.hbm_resource_seconds), 0),
    communication: result.layers.reduce((sumValue, layer) => sumValue + finite(layer.communication_resource_seconds), 0),
  };
  $("#resource-bars").innerHTML = Object.entries(resources)
    .map(([resource, value]) => {
      const ratio = result.total_seconds ? Math.min(value / result.total_seconds, 1) : 0;
      return `
        <div class="resource-column">
          <div class="resource-column-head"><span>${escapeHtml(resource)}</span><strong>${escapeHtml(formatDuration(value))}</strong></div>
          <div class="resource-track" title="${escapeHtml(formatPercent(ratio))} of total if viewed independently"><i style="--bar-width:${ratio * 100}%;--bar-color:${FLOOR_COLORS[resource]}"></i></div>
        </div>`;
    })
    .join("");
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
              <span>Logical payload is separate from per-fabric link traffic; communication time sums the applicable fabric terms.</span>
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
    const haystack = [layer.name, layer.number, layer.attention, layer.ffn, dominant?.name, layer.limiting_floor]
      .join(" ")
      .toLowerCase();
    if (filter && !haystack.includes(filter)) return;
    const key = layerKey(result, index);
    const expanded = state.expandedLayers.has(key);
    const pathClass = layer.attention || "other";
    const share = result.total_seconds ? layer.latency_seconds / result.total_seconds : 0;
    const barWidth = (layer.latency_seconds / maxLatency) * 100;
    const floor = normalizeFloor(layer.limiting_floor);
    rows.push(`
      <tr class="layer-row" data-layer-index="${index}" data-layer-key="${escapeHtml(key)}" aria-expanded="${expanded}">
        <td class="expand-cell"><button class="expand-button" type="button" aria-expanded="${expanded}" aria-controls="ops-${index}" aria-label="${expanded ? "Collapse" : "Expand"} ${escapeHtml(layer.name)} operators">${expanded ? "×" : "+"}</button></td>
        <td class="stage-name-cell">${escapeHtml(layer.number == null ? humanize(layer.name) : `Layer ${layer.number}`)}<small>${escapeHtml(layer.name)}</small></td>
        <td><span class="path-chip ${pathClass}">${escapeHtml(layerPathLabel(layer))}</span></td>
        <td class="numeric">${escapeHtml(formatDuration(layer.latency_seconds))}</td>
        <td class="numeric share-cell"><span class="share-inline"><span class="micro-bar"><i style="--bar-width:${barWidth}%;--bar-color:${pathClass === "kda" ? "var(--accent)" : pathClass === "mla" ? "var(--purple)" : "var(--text-dim)"}"></i></span><span>${escapeHtml(formatPercent(share, 2))}</span></span></td>
        <td>${escapeHtml(dominant?.name || layer.dominant_operation)}</td>
        <td><span class="resource-chip ${floor}">${escapeHtml(floor)}</span></td>
      </tr>`);
    if (expanded) rows.push(operationTable(layer, index, key));
  });
  $("#layer-table-body").innerHTML = rows.join("");
  $("#layer-chart-summary").textContent = `${formatNumber(result.layers.length)} stages · ${formatDuration(result.total_seconds)} total`;
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
          <td><span class="bottleneck-chip ${normalizeFloor(operation.bottleneck)}">${escapeHtml(operation.bottleneck)}</span></td>
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
              <thead><tr><th scope="col">Operator</th><th scope="col">Category</th><th scope="col" class="numeric">FLOPs / rank</th><th scope="col" class="numeric">HBM / rank</th><th scope="col" class="numeric">Logical collective</th><th scope="col" class="numeric">Compute floor</th><th scope="col" class="numeric">HBM floor</th><th scope="col" class="numeric">Comm floor</th><th scope="col" class="numeric">Op roofline</th><th scope="col">Bottleneck</th><th scope="col">Notes</th></tr></thead>
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
    `${hardwareMeta(result).short} latency across ${result.layers.length} model stages. Peak stage ${formatDuration(Math.max(...values))}.`,
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
  tooltip.innerHTML = `<strong>${escapeHtml(hit.layer.number == null ? humanize(hit.layer.name) : `Layer ${hit.layer.number} · ${layerPathLabel(hit.layer)}`)}</strong><span>${escapeHtml(formatDuration(hit.layer.latency_seconds))} · ${escapeHtml(normalizeFloor(hit.layer.limiting_floor))} floor</span>`;
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
  fit.className = `fit-badge ${memory.fits_nominal_capacity ? "fits" : "over"}`;
  fit.textContent = memory.fits_nominal_capacity ? `${formatPercent(ratio)} · fits nominal` : `${formatPercent(ratio)} · over nominal`;
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
  $$('input[name="tp-size"]').forEach((input) =>
    input.addEventListener("change", () => {
      updateSequenceLimit();
      scheduleCalculate();
    }),
  );
  for (const id of ["sequence-length", "batch-size", "context-length"]) {
    $(`#${id}`).addEventListener("input", () => scheduleCalculate());
  }
  $("#select-all-hardware").addEventListener("click", () => {
    const boxes = $$("input[name='hardware']");
    const shouldCheck = !boxes.every((box) => box.checked);
    boxes.forEach((box) => {
      box.checked = shouldCheck;
    });
    updateHardwareState();
    scheduleCalculate();
  });
  $("#calculator-form").addEventListener("submit", (event) => {
    event.preventDefault();
    scheduleCalculate({ immediate: true });
  });
  $("#retry-button").addEventListener("click", () => scheduleCalculate({ immediate: true }));
  $("#export-json").addEventListener("click", exportJson);
  for (const id of ["breakdown-hardware", "layer-hardware", "memory-hardware"]) {
    $(`#${id}`).addEventListener("change", (event) => {
      for (const otherId of ["breakdown-hardware", "layer-hardware", "memory-hardware"]) {
        $(`#${otherId}`).value = event.target.value;
      }
      renderBreakdown();
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
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (state.results.length) renderLayerChart();
    }, 120);
  });
}

async function init() {
  initTheme();
  bindEvents();
  updatePhaseFields();
  updateHardwareState();
  const connected = await loadManifest();
  state.ready = connected;
  if (connected) calculate();
}

document.addEventListener("DOMContentLoaded", init);
