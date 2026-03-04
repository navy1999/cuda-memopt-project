# Remote GPU Backend — Design Outline

This document outlines an optional **remote GPU backend** so the Vercel-hosted frontend can run real CUDA benchmarks without users installing anything. The frontend would call this backend instead of (or in addition to) the local helper.

## Goals

- Zero install for visitors: run a benchmark from the browser and see results.
- Same API shape as the local helper where possible, so the frontend can switch between “local” and “cloud” with one URL/config.
- Control cost and abuse (rate limits, optional auth).

## API Surface

Mirror the local helper so the demo page can use the same payloads:

- **GET /health**  
  Returns `{"ok": true}` when the service and GPU are available.

- **POST /run**  
  - Request: `{"variant": "naive"|"tiled"|"vec"|"unroll"|"tuned", "N": 512}`  
  - Response: `{"time_ms": <float>, "validation_ok": <bool>, "error": null|<string>}`  
  - Optional: add `"gpu_name": "..."` in the response for display.

Optional extensions:

- **POST /run-batch**  
  Request: `{"variants": ["naive","tiled"], "N": 1024}`  
  Response: `{"results": [{"variant": "...", "time_ms": ..., "validation_ok": ...}, ...]}`  
  Reduces round-trips when comparing several kernels.

- **GET /info**  
  Returns GPU name, driver version, and max recommended N (e.g. to avoid OOM).

## Deployment Options

1. **Single GPU VM (e.g. cloud provider)**  
   - One machine with an NVIDIA GPU, Docker container running the CUDA executables + a small HTTP server (e.g. the same logic as `local-helper/server.py` but talking to prebuilt binaries in the image).  
   - Use a reverse proxy (nginx/Caddy) for TLS and optional rate limiting.

2. **Serverless GPU (e.g. Modal, RunPod, Lambda with GPU)**  
   - Run the benchmark in a short-lived GPU container; return results via HTTP.  
   - May require adapting the server to start a job and poll or use webhooks.

3. **Queue + worker**  
   - Frontend submits a job to a queue; a GPU worker runs it and stores the result; frontend polls or uses Server-Sent Events.  
   - Better for long-running or batch jobs; more moving parts.

## Security and Limits

- **Rate limiting**: per-IP or per-session limits (e.g. 10 runs/minute) to avoid abuse.
- **Input validation**: reject N &gt; 4096 or non-whitelisted variants to protect GPU memory and runtime.
- **Auth (optional)**: API key or signed token for higher limits or batch access; not required for a minimal public demo.
- **CORS**: allow the Vercel frontend origin only in production.

## Frontend Integration

- Add an environment variable, e.g. `NEXT_PUBLIC_GPU_BACKEND_URL`, set to the cloud backend base URL (or leave unset to use only WebGPU + local helper).
- In the demo page:
  - If `NEXT_PUBLIC_GPU_BACKEND_URL` is set, call `GET ${url}/health` on load; if ok, show “Run on cloud GPU” with the same variant/N controls and `POST ${url}/run`.
  - Keep “Run on local GPU” when the local helper is detected, and “Run WebGPU” for in-browser only.

## Implementation Sketch

1. **Docker image**  
   - Base: NVIDIA CUDA runtime image.  
   - Copy project `build/` (or build inside image) so `matmul_naive`, `matmul_tiled`, etc. are present.  
   - Copy and run a small HTTP server (Python or Go) that execs the right binary and parses stdout, same as `local-helper/server.py`.

2. **Orchestration**  
   - Deploy the container on a GPU VM; expose one port; put behind HTTPS (e.g. Cloudflare or provider LB).  
   - Optionally add a simple rate limiter (e.g. in-memory or Redis) in front of the server.

3. **Vercel**  
   - No server-side proxy needed: the browser calls the GPU backend URL directly. Set `NEXT_PUBLIC_GPU_BACKEND_URL` in Vercel project settings.

This design keeps the backend stateless and the frontend unchanged except for a second “cloud” backend option when the env var is set.
