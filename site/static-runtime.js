"use strict";

(() => {
  const worker = new Worker(new URL("./calculator-worker.mjs", document.baseURI), {
    type: "module",
  });
  const pending = new Map();
  let nextRequestId = 1;

  worker.addEventListener("message", (event) => {
    const request = pending.get(event.data?.id);
    if (!request) return;
    pending.delete(event.data.id);
    request.cleanup();
    if (event.data.error) {
      request.reject(new Error(event.data.error));
      return;
    }
    request.resolve(
      new Response(JSON.stringify(event.data.body), {
        status: event.data.status,
        headers: { "Content-Type": "application/json" },
      }),
    );
  });

  worker.addEventListener("error", (event) => {
    for (const request of pending.values()) {
      request.cleanup();
      request.reject(new Error(event.message || "The browser calculation engine failed to load."));
    }
    pending.clear();
  });

  globalThis.k3CalculatorFetch = (resource, options = {}) => {
    const path = new URL(resource, document.baseURI).pathname;
    const signal = options.signal;
    if (signal?.aborted) {
      return Promise.reject(new DOMException("The request was aborted.", "AbortError"));
    }

    let body = null;
    if (options.body) {
      try {
        body = JSON.parse(options.body);
      } catch {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              error: { type: "invalid_json", message: "Request body is not valid JSON." },
            }),
            { status: 400, headers: { "Content-Type": "application/json" } },
          ),
        );
      }
    }

    return new Promise((resolve, reject) => {
      const id = nextRequestId++;
      const abort = () => {
        pending.delete(id);
        reject(new DOMException("The request was aborted.", "AbortError"));
      };
      const cleanup = () => signal?.removeEventListener("abort", abort);
      signal?.addEventListener("abort", abort, { once: true });
      pending.set(id, { resolve, reject, cleanup });
      worker.postMessage({ id, path, body });
    });
  };
})();
