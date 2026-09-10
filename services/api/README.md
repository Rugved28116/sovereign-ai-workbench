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

`POST /v1/generate` accepts prompts up to 32,768 characters. Callers may explicitly supply 1–16 unique `required_capabilities`, with capability names up to 64 characters; explicit capabilities bypass automatic classification. When the field is omitted, deterministic classifier rules infer one or more required capabilities. Empty prompts, explicit `null`, empty capability lists or names, and duplicate capabilities are rejected. The endpoint also has a 256 KiB request-body limit.

## Deterministic task classification

Automatic classification is provider-neutral and uses normalized prompt text with primary-class precedence: `vision` → `coding` → `document` → `reasoning` → `general`. A frozen task-requirements value can contain one or more capabilities. Explicit combination rules cover document/vision/reasoning, vision/coding, document/coding, vision/reasoning, document/reasoning, document/vision, and coding/reasoning requests; signals are not blindly unioned. Ambiguous requests default to `general`/`chat`; classification never selects a model or provider and cannot bypass Stage 1 eligibility or environment restrictions.

Inferred capabilities use AND semantics and are emitted in deterministic `document`, `vision`, `coding`, `reasoning`, `chat` order as applicable. Stage 1 requires one eligible model to support all of them; capabilities are never dropped, and decomposition across multiple models is not supported. Explicit client capabilities remain authoritative, retain caller order, and bypass inference entirely.

The classifier is currently a conservative keyword-and-phrase ruleset. It performs no model calls, embeddings, semantic routing, network access, or learned scoring. In `development`, response routing metadata identifies whether capabilities were `explicit` or `inferred` and includes the inferred task class and required capabilities. These classifier details are omitted from successful responses in other environments.

## Two-stage routing

Routing first applies the mandatory, fail-closed sovereign eligibility filter. Only models approved for enablement, the exact active environment, mock-provider restrictions, and all requested capabilities reach the Stage 2 optimizer.

Stage 2 uses a provider-neutral optimizer interface over immutable candidate projections; registry models and provider configuration are not exposed to it. Its current deterministic implementation selects by ascending numeric priority and then ascending lexicographic model ID, preserving existing behavior. Optimizer output is checked against a private snapshot of the exact Stage 1-approved state; returning an unknown, rejected, or altered candidate raises a typed routing error instead of falling back or bypassing eligibility. No learned classification, semantic routing, telemetry, scoring, or load balancing is implemented.

### Local OpenAI-compatible provider

Registry entries may use the provider key `local-openai-compatible`. When an enabled entry is eligible for the active environment, provider-private configuration must set:

```bash
SOVEREIGN_LOCAL_OPENAI_BASE_URL=http://127.0.0.1:8000/v1
```

The URL may use HTTP or HTTPS with exact `localhost`, IPv4 loopback, RFC1918 private IPv4, or IPv6 loopback. Exact `localhost` is converted to `127.0.0.1` before requests, so accepted endpoints never depend on DNS or hosts-file resolution. Public IP addresses, other DNS hostnames, control characters, and malformed URLs are rejected during configuration. Configuration is not stored in `models/registry.yaml`, redirects are not followed, and environment proxy settings are ignored.

Provider responses are streamed and limited to 4 MiB, including unsuccessful responses. Oversized declared or streamed bodies fail with a typed provider response error. No local runtime or model is installed by this provider adapter.

The generic `local-openai-compatible` provider remains supported for backward compatibility with local compatible servers.

### vLLM provider

Registry entries may use the runtime-specific provider key `vllm`. When an enabled vLLM entry is eligible for the active environment, set its provider-private endpoint configuration with:

```bash
SOVEREIGN_VLLM_BASE_URL=http://127.0.0.1:8000/v1
```

The vLLM adapter reuses the same sovereign endpoint validation, DNS avoidance, proxy and redirect controls, and 4 MiB streamed response limit described above. This milestone implements only the basic text-generation protocol adapter; it does not install a vLLM server or model, and it does not support client streaming, tools, vision, embeddings, or model management.

## Current limitations

Only deterministic routing, the offline mock provider, and local HTTP protocol adapters are implemented. The supplied mock models are `development`-only, so generation fails closed in `on-prem` and `air-gapped`. No real inference runtime or model is included. Authentication, orchestration, retrieval, tools, and other later architecture are out of scope.
