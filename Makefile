SHELL := /bin/bash
PY := . .venv/bin/activate && python
.PHONY: help setup models data chain cache eval export serve web demo tunnel contracts test
help:            ## list targets
	@grep -E '^[a-z-]+:.*##' Makefile | sed 's/:.*##/ —/'
setup:           ## npm deps (python env: see PROGRESS.md P0.1)
	npm install
models:          ## download weights (resumable)
	bash ml/data/download/weights.sh
data:            ## manifests + identity-disjoint splits
	$(PY) -m ml.data.manifests && $(PY) -m ml.data.splits
chain:           ## WhatsApp ladders (background job)
	$(PY) -m ml.forwardchain.ladder
cache:           ## preprocessing cache
	$(PY) -m ml.data.build_cache
eval:            ## ONNX end-to-end eval -> ml/reports/benchmark.json
	$(PY) -m ml.eval.run_eval
export:          ## ONNX + INT8 + parity + registry
	$(PY) -m ml.export.to_onnx && $(PY) -m ml.export.quantize && $(PY) -m ml.export.parity
serve:           ## API on :8000
	$(PY) -m uvicorn server.main:app --port 8000
web:             ## Next dev on :3000
	npm run dev
tunnel:          ## HTTPS tunnel to the API
	cloudflared tunnel --url http://localhost:8000
contracts:       ## codegen pydantic + zod from contracts/analysis.schema.json
	$(PY) -m ml.data.gen_contracts
test:
	$(PY) -m pytest -q
