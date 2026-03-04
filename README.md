# CUDA Memopt — Frontend + CUDA Project

This repository is set up so **Vercel deploys the root** as a Next.js app (no need to set a root directory).

- **Root (this directory):** Next.js app — charts, report viewer, WebGPU demo, and optional local-CUDA helper UI.  
  - Run: `npm install && npm run dev`  
  - Deploy: connect the repo to Vercel; it will build from the root automatically.

- **`cuda/`:** CUDA matrix multiplication project (kernels, LLVM pass, scripts, report).  
  - See **[cuda/README.md](cuda/README.md)** for building and running benchmarks.  
  - From `cuda/`, run `python scripts/benchmarking.py` and `python scripts/autotune.py`; then rebuild the frontend (or run `node scripts/copy-data.js && npm run build`) so the site shows the latest results.
