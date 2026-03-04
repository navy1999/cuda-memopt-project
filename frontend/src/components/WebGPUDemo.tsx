"use client";

import { useCallback, useEffect, useState } from "react";

const TILE = 16;
const DEFAULT_N = 256;
const LOCAL_HELPER_URL = "http://127.0.0.1:9000";

export default function WebGPUDemo() {
  const [supported, setSupported] = useState<boolean | null>(null);
  const [result, setResult] = useState<{ timeMs: number; N: number; ok: boolean } | null>(null);
  const [running, setRunning] = useState(false);
  const [N, setN] = useState(DEFAULT_N);
  const [localHelperAvailable, setLocalHelperAvailable] = useState(false);
  const [localVariant, setLocalVariant] = useState<"naive" | "tiled" | "vec" | "unroll" | "tuned">("tiled");
  const [localResult, setLocalResult] = useState<{ timeMs: number; variant: string; N: number; ok: boolean } | null>(null);
  const [localRunning, setLocalRunning] = useState(false);

  useEffect(() => {
    fetch(`${LOCAL_HELPER_URL}/health`, { mode: "cors" })
      .then((r) => r.json())
      .then((d) => setLocalHelperAvailable(d?.ok === true))
      .catch(() => setLocalHelperAvailable(false));
  }, []);

  const checkSupport = useCallback(() => {
    const ok = typeof navigator !== "undefined" && !!navigator.gpu;
    setSupported(ok);
    return ok;
  }, []);

  const runLocal = useCallback(async () => {
    setLocalRunning(true);
    setLocalResult(null);
    try {
      const res = await fetch(`${LOCAL_HELPER_URL}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ variant: localVariant, N }),
      });
      const data = await res.json();
      setLocalResult({
        timeMs: data.time_ms ?? 0,
        variant: localVariant,
        N,
        ok: data.validation_ok === true && !data.error,
      });
    } catch {
      setLocalResult({ timeMs: 0, variant: localVariant, N, ok: false });
    }
    setLocalRunning(false);
  }, [localVariant, N]);

  const runDemo = useCallback(async () => {
    if (typeof navigator === "undefined" || !navigator.gpu) {
      setSupported(false);
      return;
    }
    setSupported(true);
    setRunning(true);
    setResult(null);
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        setResult({ timeMs: 0, N, ok: false });
        setRunning(false);
        return;
      }
      const device = await adapter.requestDevice();
      const size = N * N;
      const bytes = size * 4;

      const bufA = device.createBuffer({ size: bytes, usage: 0x8 | 0x4 }); // STORAGE | COPY_SRC
      const bufB = device.createBuffer({ size: bytes, usage: 0x8 | 0x4 });
      const bufC = device.createBuffer({ size: bytes, usage: 0x8 | 0x10 }); // STORAGE | COPY_DST

      const stagingA = device.createBuffer({ size: bytes, usage: 0x1 }); // MAP_READ | COPY_DST
      const stagingB = device.createBuffer({ size: bytes, usage: 0x1 });
      const stagingC = device.createBuffer({ size: bytes, usage: 0x2 | 0x10 }); // MAP_WRITE | COPY_SRC

      const initData = new Float32Array(size);
      for (let i = 0; i < size; i++) initData[i] = 1;
      device.queue.writeBuffer(stagingA, 0, initData);
      for (let i = 0; i < size; i++) initData[i] = 2;
      device.queue.writeBuffer(stagingB, 0, initData);

      device.queue.copyBufferToBuffer(stagingA, 0, bufA, 0, bytes);
      device.queue.copyBufferToBuffer(stagingB, 0, bufB, 0, bytes);

      const wgsl = `
        @group(0) @binding(0) var<storage, read> A: array<f32>;
        @group(0) @binding(1) var<storage, read> B: array<f32>;
        @group(0) @binding(2) var<storage, read_write> C: array<f32>;

        @compute @workgroup_size(${TILE}, ${TILE})
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {
          let n = ${N}u;
          let row = id.y;
          let col = id.x;
          if (row >= n || col >= n) { return; }
          var sum = 0.0;
          for (var k = 0u; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
          }
          C[row * n + col] = sum;
        }
      `;

      const module = device.createShaderModule({ code: wgsl });
      const pipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });

      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: bufA } },
          { binding: 1, resource: { buffer: bufB } },
          { binding: 2, resource: { buffer: bufC } },
        ],
      });

      const workgroups = Math.ceil(N / TILE);
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(workgroups, workgroups);
      pass.end();

      const start = performance.now();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      const elapsed = performance.now() - start;

      encoder.reset();
      const copyEnc = device.createCommandEncoder();
      copyEnc.copyBufferToBuffer(bufC, 0, stagingC, 0, bytes);
      device.queue.submit([copyEnc.finish()]);
      await device.queue.onSubmittedWorkDone();

      const mapped = await stagingC.mapAsync(1);
      const out = new Float32Array(mapped.byteLength / 4);
      out.set(new Float32Array(mapped));
      stagingC.unmap();
      const ok = Math.abs(out[0] - 2 * N) < 0.1;

      bufA.destroy();
      bufB.destroy();
      bufC.destroy();
      stagingA.destroy();
      stagingB.destroy();
      stagingC.destroy();
      device.destroy();

      setResult({ timeMs: elapsed, N, ok });
    } catch (e) {
      setResult({ timeMs: 0, N, ok: false });
      console.error(e);
    }
    setRunning(false);
  }, [N]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center gap-4">
        <button
          type="button"
          onClick={checkSupport}
          className="rounded-lg bg-zinc-200 dark:bg-zinc-700 px-4 py-2 font-medium"
        >
          Check WebGPU support
        </button>
        {supported === true && (
          <span className="text-green-600 dark:text-green-400">WebGPU is supported</span>
        )}
        {supported === false && (
          <span className="text-amber-600 dark:text-amber-400">
            WebGPU not available (try Chrome/Edge with flags or latest)
          </span>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-4">
        <label className="flex items-center gap-2">
          <span className="text-sm">Matrix size N:</span>
          <input
            type="number"
            min={16}
            max={512}
            step={16}
            value={N}
            onChange={(e) => setN(Number(e.target.value))}
            className="w-24 rounded border border-zinc-300 dark:border-zinc-600 bg-white dark:bg-zinc-800 px-2 py-1"
          />
        </label>
        <button
          type="button"
          onClick={runDemo}
          disabled={running}
          className="rounded-lg bg-zinc-900 dark:bg-zinc-100 text-white dark:text-zinc-900 px-4 py-2 font-medium disabled:opacity-50"
        >
          {running ? "Running…" : "Run WebGPU matmul"}
        </button>
      </div>

      {result && (
        <div className="rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 p-4">
          <p className="font-mono text-sm">
            N={result.N} — Time: <strong>{result.timeMs.toFixed(2)} ms</strong>
            {result.ok ? " — Validation C[0] OK" : " — Validation failed or error"}
          </p>
        </div>
      )}

      {localHelperAvailable ? (
        <div className="rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-900 p-4">
          <h3 className="font-semibold mb-2">Local CUDA helper (connected)</h3>
          <div className="flex flex-wrap items-center gap-4 mb-2">
            <label className="flex items-center gap-2">
              <span className="text-sm">Variant:</span>
              <select
                value={localVariant}
                onChange={(e) => setLocalVariant(e.target.value as typeof localVariant)}
                className="rounded border border-zinc-300 dark:border-zinc-600 bg-white dark:bg-zinc-800 px-2 py-1"
              >
                <option value="naive">naive</option>
                <option value="tiled">tiled</option>
                <option value="vec">vec</option>
                <option value="unroll">unroll</option>
                <option value="tuned">tuned</option>
              </select>
            </label>
            <button
              type="button"
              onClick={runLocal}
              disabled={localRunning}
              className="rounded-lg bg-emerald-600 text-white px-4 py-2 font-medium disabled:opacity-50"
            >
              {localRunning ? "Running…" : "Run on local GPU"}
            </button>
          </div>
          {localResult && (
            <p className="font-mono text-sm text-zinc-700 dark:text-zinc-300">
              {localResult.variant} N={localResult.N} — {localResult.timeMs.toFixed(2)} ms
              {localResult.ok ? " — OK" : " — Error or validation failed"}
            </p>
          )}
        </div>
      ) : (
        <div className="rounded-lg border border-zinc-200 dark:border-zinc-700 bg-zinc-100 dark:bg-zinc-800/50 p-4 text-sm text-zinc-700 dark:text-zinc-300">
          <h3 className="font-semibold mb-2">Local CUDA helper</h3>
          <p className="mb-2">
            To run real CUDA kernels from this page, start the local helper from the project root:{" "}
            <code className="bg-zinc-200 dark:bg-zinc-700 px-1 rounded">python local-helper/server.py</code>, then refresh.
            The demo will detect the helper and offer a &quot;Run on local GPU&quot; option.
          </p>
        </div>
      )}
    </div>
  );
}
