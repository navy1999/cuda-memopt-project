# Local CUDA Helper

Small HTTP server that runs the project’s CUDA matmul executables so the Vercel-hosted frontend can trigger real GPU runs from your machine.

## Setup

1. **Python 3.8+** with no extra deps (uses only the standard library).
2. Build the CUDA executables from the **project root**:
   ```bash
   cmake -S . -B build --config Release
   cmake --build build --config Release
   ```
   Or compile manually into `build/` (see main README).

## Run

From the **project root** (so paths to `build/` and `src/` are correct):

```bash
python local-helper/server.py
```

By default the server listens on **http://127.0.0.1:9000**. The frontend demo page will try this URL when you click “Use local GPU”.

## Endpoints

- **GET /health** — Returns `{"ok": true}` if the server is up.
- **POST /run** — Run a single benchmark.
  - Body: `{"variant": "naive"|"tiled"|"vec"|"unroll"|"tuned", "N": 512}`
  - Returns: `{"time_ms": <float>, "validation_ok": <bool>, "error": null|<string>}`

## CORS

The server allows requests from any origin so the browser (e.g. your Vercel-deployed app) can call it. Run only on localhost; do not expose to the network.
