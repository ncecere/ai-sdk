# ai-sdk Examples

This directory contains small, runnable examples that demonstrate how to use the Go ai-sdk with different providers and features.

Each example is a standalone `main` package. Run them from the `ai-sdk/ai-sdk` module root with `go run`.

---

## OpenAI

Env:

- `OPENAI_API_KEY` – required.

Examples:

- `examples/http_server` – Simple HTTP chat endpoint using OpenAI chat.
- `examples/cli_stream` – CLI streaming text example using `StreamText`.
- `examples/http_image` – HTTP image generation using `GenerateImage`.
- `examples/cli_transcribe` – CLI transcription using `Transcribe`.
- `examples/openai_json` – Structured JSON output using `GenerateObject`.
- `examples/openai_tools` – Tool calling with an `add` tool.

Run (examples share the same env):

```bash
cd ai-sdk/ai-sdk
export OPENAI_API_KEY=your_key

# structured output
go run ./examples/openai_json

# tools
go run ./examples/openai_tools

# streaming
go run ./examples/cli_stream
```

---

## Anthropic

Env:

- `ANTHROPIC_API_KEY` – required.

Examples:

- `examples/anthropic_json` – Structured JSON output via `GenerateObject`.
- `examples/anthropic_tools` – Tool calling with an `add` tool.
- `examples/anthropic_stream` – Streaming text via `StreamText`.

Run:

```bash
cd ai-sdk/ai-sdk
export ANTHROPIC_API_KEY=your_key

go run ./examples/anthropic_json
go run ./examples/anthropic_tools
go run ./examples/anthropic_stream
```

---

## Groq (OpenAI-compatible)

Env:

- `GROQ_API_KEY` – required.
- `GROQ_MODEL_ID` – optional model ID (defaults to `llama-3.1-8b-instant`).

Examples:

- `examples/groq_json` – Structured JSON output with Groq chat.
- `examples/groq_tools` – Tool calling with an `add` tool.
- `examples/groq_stream` – Streaming text via `StreamText`.

Run:

```bash
cd ai-sdk/ai-sdk
export GROQ_API_KEY=your_key
# optional: export GROQ_MODEL_ID=your_model

go run ./examples/groq_json
go run ./examples/groq_tools
go run ./examples/groq_stream
```

---

## OpenAI-compatible backends

These examples target generic OpenAI-compatible endpoints (for example, the UFL AI gateway).

Env:

- `OPENAI_COMPATIBLE_BASE_URL` – required (e.g. `https://api.ai.it.ufl.edu/v1`).
- `OPENAI_COMPATIBLE_API_KEY` or `OPENAI_API_KEY` – required.
- `COMPAT_MODEL_ID` – optional model ID (defaults to `gpt-oss-20b`).

Examples:

- `examples/compat_text` – Basic text generation against a compatible backend.
- `examples/compat_json` – Structured JSON output with `GenerateObject`.
- `examples/compat_tools` – Tool calling with an `add` tool.
- `examples/compat_stream` – Streaming text via `StreamText`.

Run:

```bash
cd ai-sdk/ai-sdk
export OPENAI_COMPATIBLE_BASE_URL=...
export OPENAI_COMPATIBLE_API_KEY=your_key
# optional: export COMPAT_MODEL_ID=your_model

go run ./examples/compat_text
go run ./examples/compat_json
go run ./examples/compat_tools
go run ./examples/compat_stream
```

---

## Registry and Agent examples

These examples demonstrate higher-level packages like the registry and agent.

- `examples/http_registry` – HTTP chat endpoint resolving models via `registry.InMemoryRegistry`.
- `examples/http_completion_registry` – HTTP completion endpoint using `CompletionModel` via registry.
- `examples/agent_cli` – CLI agent with a simple tool loop.
- `examples/http_agent` – HTTP SSE agent endpoint using `agent.RunWithEvents`.

These share the same provider-specific env vars as above (OpenAI or compatible backends depending on how you configure the registry).
