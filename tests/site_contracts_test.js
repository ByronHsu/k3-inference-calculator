"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");

globalThis.k3CalculatorFetch = async () => {
  throw new Error("The response-contract tests must not perform requests.");
};
const { normalizeFloor, validateCalculatorResponse } = require("../site/app.js");

function resultFor(
  family,
  {
    gpuCount = 16,
    tpSize = 16,
    moeSharding = "tp",
    epSize = 1,
    phase = "decode",
    batchSize = 1,
    sequenceLength = 4096,
    contextLength = 128,
    attentionTpSize,
    attentionDpSize,
    moeA2aBackend,
    moeBackend,
    spMoe,
    sharedExpertTpSize,
    executionContractId,
    dpLmHead,
    kvCacheBytesPerElement,
    kdaStateBytesPerElement,
    gpusPerNode,
    nodeCount,
    parallelWorkLedger,
    layers,
  } = {},
) {
  return {
    hardware: {
      family,
      gpu_count: gpuCount,
      tp_size: tpSize,
      moe_sharding: moeSharding,
      ep_size: epSize,
      attention_tp_size: attentionTpSize,
      attention_dp_size: attentionDpSize,
      moe_a2a_backend: moeA2aBackend,
      moe_backend: moeBackend,
      sp_moe: spMoe,
      shared_expert_tp_size: sharedExpertTpSize,
      execution_contract_id: executionContractId,
      dp_lm_head: dpLmHead,
      kv_cache_bytes_per_element: kvCacheBytesPerElement,
      kda_state_bytes_per_element: kdaStateBytesPerElement,
      gpus_per_node: gpusPerNode,
      node_count: nodeCount,
    },
    workload: {
      phase,
      batch_size: batchSize,
      ...(phase === "prefill"
        ? { sequence_length: sequenceLength }
        : { context_length: contextLength }),
    },
    parallel_work_ledger: parallelWorkLedger,
    layers: layers || [
      {
        critical_path_lower_bound_seconds: 1,
        compute_resource_seconds: 0,
        hbm_resource_seconds: 0,
        communication_resource_seconds: 0,
        latency_seconds: 1,
        limiting_certificates: ["critical_path"],
      },
    ],
  };
}

function responseFor(results, schemaVersion = 2) {
  return { schema_version: schemaVersion, results };
}

test("response validator accepts exact TP and H200 TP+EP scenarios", () => {
  const tpPayload = {
    phase: "decode",
    hardware: ["h200", "b300"],
    tp_size: 16,
    moe_sharding: "tp",
    batch_size: 1,
    context_length: 128,
  };
  const tpResults = [resultFor("h200"), resultFor("b300")];
  assert.equal(validateCalculatorResponse(tpPayload, responseFor(tpResults)), tpResults);

  const epPayload = {
    phase: "prefill",
    hardware: ["h200"],
    tp_size: 32,
    moe_sharding: "ep",
    batch_size: 1,
    sequence_length: 4096,
  };
  const epResults = [
    resultFor("h200", {
      gpuCount: 32,
      tpSize: 32,
      moeSharding: "ep",
      epSize: 32,
      phase: "prefill",
    }),
  ];
  assert.equal(validateCalculatorResponse(epPayload, responseFor(epResults)), epResults);
});

test("response validator rejects stale schemas and certificate fields", () => {
  const payload = {
    phase: "decode",
    hardware: ["h200"],
    tp_size: 16,
    moe_sharding: "tp",
    batch_size: 1,
    context_length: 128,
  };
  assert.throws(
    () => validateCalculatorResponse(payload, responseFor([resultFor("h200")], 1)),
    /unsupported response schema/,
  );
  const staleLayer = {
    dependency_path_seconds: 1,
    compute_resource_seconds: 0,
    hbm_resource_seconds: 0,
    communication_resource_seconds: 0,
    latency_seconds: 1,
    limiting_floor: "dependency",
  };
  assert.throws(
    () =>
      validateCalculatorResponse(
        payload,
        responseFor([resultFor("h200", { layers: [staleLayer] })]),
      ),
    /canonical lower-bound certificate fields/,
  );
});

function blackwellEpResult(family, tpSize, overrides = {}) {
  const gpusPerNode = family === "b300" ? 8 : 4;
  const attentionDpSize = tpSize / 8;
  return resultFor(family, {
    gpuCount: tpSize,
    tpSize,
    moeSharding: "ep",
    epSize: tpSize,
    attentionTpSize: 8,
    attentionDpSize,
    moeA2aBackend: "MegaMoE",
    moeBackend: "DeepGEMM MegaMoE MXFP4 W4A8",
    spMoe: true,
    sharedExpertTpSize: 1,
    executionContractId: "kimi_k3_blackwell_megamoe_sp_v1",
    dpLmHead: tpSize > 8,
    kvCacheBytesPerElement: 1,
    kdaStateBytesPerElement: 2,
    gpusPerNode,
    nodeCount: tpSize / gpusPerNode,
    parallelWorkLedger: {
      contract_id: "kimi_k3_blackwell_parallel_ledger_v1",
      bound_condition_id: "balanced_dp_fractional_uniform_ep_routing",
      bound_condition: "Conditional balanced DP assignment and fractional uniform EP routing.",
      topology_contract: "Recipe topology.",
      dp_mlp_aligned_rows: Array(attentionDpSize).fill(8),
      dp_model_rows: Array(attentionDpSize).fill(8),
      excluded_positive_term_ids: ["megamoe_alignment_padding", "collective_startup"],
    },
    ...overrides,
  });
}

test("response validator accepts every Blackwell matched TP+EP execution contract", () => {
  for (const family of ["b300", "gb300"]) {
    for (const tpSize of [8, 16, 32]) {
      const payload = {
        phase: "decode",
        hardware: [family],
        tp_size: tpSize,
        moe_sharding: "ep",
        batch_size: 1,
        context_length: 128,
      };
      const result = blackwellEpResult(family, tpSize);
      assert.equal(validateCalculatorResponse(payload, responseFor([result]))[0], result);
    }
  }
});

test("response validator rejects every stale Blackwell EP execution field", () => {
  const payload = {
    phase: "decode",
    hardware: ["gb300"],
    tp_size: 32,
    moe_sharding: "ep",
    batch_size: 1,
    context_length: 128,
  };
  const staleFields = [
    { executionContractId: "kimi_k3_plain_tp_or_h200_ep_v1" },
    { parallelWorkLedger: undefined },
    {
      parallelWorkLedger: {
        contract_id: "stale_ledger",
        bound_condition_id: "unconditional",
        bound_condition: "",
        topology_contract: "",
        dp_mlp_aligned_rows: [8, 8, 8, 8],
        dp_model_rows: [8, 8, 8, 8],
        excluded_positive_term_ids: [],
      },
    },
  ];
  for (const overrides of staleFields) {
    assert.throws(
      () =>
        validateCalculatorResponse(
          payload,
          responseFor([blackwellEpResult("gb300", 32, overrides)]),
        ),
      /Blackwell EP execution recipe that does not match/,
    );
  }
});

test("response validator rejects incomplete or duplicate hardware inventories", () => {
  const payload = {
    phase: "decode",
    hardware: ["h200", "b300"],
    tp_size: 16,
    moe_sharding: "tp",
    batch_size: 1,
    context_length: 128,
  };
  assert.throws(
    () => validateCalculatorResponse(payload, responseFor([resultFor("h200")])),
    /partial hardware comparison/,
  );
  assert.throws(
    () =>
      validateCalculatorResponse(
        payload,
        responseFor([resultFor("h200"), resultFor("h200")]),
      ),
    /unexpected or duplicate hardware family/,
  );
});

test("response validator rejects every sharding and workload mismatch", () => {
  const payload = {
    phase: "decode",
    hardware: ["h200"],
    tp_size: 16,
    moe_sharding: "tp",
    batch_size: 1,
    context_length: 128,
  };
  for (const override of [
    { gpuCount: 32 },
    { tpSize: 32 },
    { tpSize: "16" },
    { moeSharding: "ep", epSize: 16 },
    { epSize: 16 },
  ]) {
    assert.throws(
      () =>
        validateCalculatorResponse(
          payload,
          responseFor([resultFor("h200", override)]),
        ),
      /sharding topology that does not match/,
    );
  }
  assert.throws(
    () =>
      validateCalculatorResponse(
        payload,
        responseFor([resultFor("h200", { phase: "prefill" })]),
      ),
    /workload that does not match/,
  );
  for (const override of [{ batchSize: 999 }, { contextLength: 999 }]) {
    assert.throws(
      () =>
        validateCalculatorResponse(
          payload,
          responseFor([resultFor("h200", override)]),
        ),
      /workload that does not match/,
    );
  }
});

test("floor normalization preserves aliases and never guesses unknown values", () => {
  assert.equal(normalizeFloor("dependency"), "critical_path");
  assert.equal(normalizeFloor("compute_resource"), "compute");
  assert.equal(normalizeFloor("hbm_resource"), "hbm");
  assert.equal(normalizeFloor("communication_resource"), "communication");
  assert.equal(normalizeFloor("future_certificate"), "unknown");
  assert.equal(normalizeFloor(undefined), "unknown");
});
