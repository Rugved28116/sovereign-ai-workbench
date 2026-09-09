# Sovereign API

This service is the first local backend slice: FastAPI, registry validation, sovereign model eligibility, deterministic routing, and a development-only mock provider.

## Setup

Requires Python 3.12 and `uv`.

```bash
uv sync
```

## Run

```bash
SOVEREIGN_ENV=development uv run uvicorn sovereign_api.main:app --reload
```

## Test

```bash
uv run pytest
```

## Environment

`SOVEREIGN_ENV` is mandatory and accepts exactly `development`, `on-prem`, or `air-gapped`. Missing or unknown values fail application startup.

`POST /v1/generate` accepts prompts up to 32,768 characters, 1–16 unique required capabilities, and capability names up to 64 characters. Empty prompts, empty capability names, and duplicate capabilities are rejected. The endpoint also has a 256 KiB request-body limit.

## Current limitations

Only deterministic routing and the offline mock provider are implemented. The supplied mock models are `development`-only, so generation fails closed in `on-prem` and `air-gapped`. Authentication, real inference runtimes, orchestration, retrieval, tools, and other later architecture are out of scope.
