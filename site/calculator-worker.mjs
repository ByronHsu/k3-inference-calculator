import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.4/full/pyodide.mjs";

const ready = (async () => {
  const pyodide = await loadPyodide();
  const archiveResponse = await fetch(new URL("./runtime.zip", import.meta.url));
  if (!archiveResponse.ok) {
    throw new Error(`Failed to load the calculator runtime (${archiveResponse.status}).`);
  }
  pyodide.unpackArchive(await archiveResponse.arrayBuffer(), "zip");
  await pyodide.runPythonAsync(`
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "benchmark"))
import kimi_k3_inference_calculator as calculator

def browser_manifest_json():
    return json.dumps(
        {"status": 200, "body": calculator.manifest_payload()},
        separators=(",", ":"),
        allow_nan=False,
    )

def browser_calculate_json(request_json):
    try:
        body = calculator.calculate_payload(json.loads(request_json))
        status = 200
    except calculator.ApiError as error:
        body = error.to_dict()
        status = 400
    except Exception:
        body = {
            "error": {
                "type": "internal_error",
                "message": "The calculator failed while evaluating this request.",
            }
        }
        status = 500
    return json.dumps(
        {"status": status, "body": body},
        separators=(",", ":"),
        allow_nan=False,
    )
`);
  return pyodide;
})();

self.addEventListener("message", async (event) => {
  const { id, path, body } = event.data;
  try {
    const pyodide = await ready;
    let envelopeJson;
    if (path.endsWith("/api/manifest")) {
      envelopeJson = pyodide.runPython("browser_manifest_json()");
    } else if (path.endsWith("/api/calculate")) {
      pyodide.globals.set("_browser_request_json", JSON.stringify(body));
      try {
        envelopeJson = pyodide.runPython(
          "browser_calculate_json(_browser_request_json)",
        );
      } finally {
        pyodide.globals.delete("_browser_request_json");
      }
    } else {
      envelopeJson = JSON.stringify({
        status: 404,
        body: { error: { type: "not_found", message: "API route not found." } },
      });
    }
    self.postMessage({ id, ...JSON.parse(envelopeJson) });
  } catch (error) {
    self.postMessage({
      id,
      error: error instanceof Error ? error.message : "The calculation engine failed.",
    });
  }
});
