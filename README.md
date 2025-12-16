# ai-sdk (Go)

A Go library inspired by the Vercel AI SDK, providing a provider-agnostic API for language models and embeddings with first-class support for OpenAI and OpenAI-compatible backends.

## Installation

```bash
go get github.com/ncecere/ai-sdk@latest
```

This installs the core `ai` package and the `openai` provider package.

## Configuration

### OpenAI

By default the OpenAI provider reads configuration from environment variables:

- `OPENAI_API_KEY` – required.
- `OPENAI_BASE_URL` – optional (defaults to `https://api.openai.com`).

Example:

```bash
export OPENAI_API_KEY="sk-..."
# optional
export OPENAI_BASE_URL="https://api.openai.com"
```

### OpenAI-Compatible Backends

For OpenAI-compatible endpoints (e.g. gateways that expose `/v1/chat/completions`):

- `OPENAI_COMPATIBLE_API_KEY` – optional, falls back to `OPENAI_API_KEY`.
- `OPENAI_COMPATIBLE_BASE_URL` – required when using `CompatibleClient`.

Example for a `/v1`-style base URL:

```bash
export OPENAI_COMPATIBLE_BASE_URL="https://api.ai.it.ufl.edu/v1"
export OPENAI_COMPATIBLE_API_KEY="sk-..."
```

The client logic handles both `https://...` and `https://.../v1` style base URLs:

- `https://api.openai.com` → `https://api.openai.com/v1/chat/completions`.
- `https://api.ai.it.ufl.edu/v1` → `https://api.ai.it.ufl.edu/v1/chat/completions`.

## Quickstart

### Basic Text Generation

```go
package main

import (
    "context"
    "log"

    ai "github.com/ncecere/ai-sdk"
    "github.com/ncecere/ai-sdk/openai"
    "github.com/ncecere/ai-sdk/provider"
)

func main() {
    client, err := openai.NewClient(provider.ClientOptions{})
    if err != nil {
        log.Fatalf("failed to create OpenAI client: %v", err)
    }

    model := client.ChatModel("gpt-4o-mini")

    ctx := context.Background()
    text, err := ai.GenerateSimpleText(ctx, model, "Say hello from Go.")
    if err != nil {
        log.Fatalf("generate error: %v", err)
    }

    log.Println("response:", text)
}
```

### Conversation Helper

Use the `Conversation` helper to build message histories:

```go
conv := ai.NewConversation().
    System("You are a helpful assistant.").
    User("Hello!")

res, err := ai.GenerateText(ctx, ai.GenerateTextRequest{
    Model:    model,
    Messages: conv.Messages,
})
```

### Streaming Text

```go
ctx := context.Background()
stream, err := ai.StreamText(ctx, ai.GenerateTextRequest{
    Model: model,
    Messages: []ai.Message{{
        Role:    ai.RoleUser,
        Content: "Stream a short message.",
    }},
})
if err != nil {
    log.Fatalf("stream error: %v", err)
}
defer stream.Close()

for {
    delta, err := stream.Next(ctx)
    if err != nil {
        log.Fatalf("next error: %v", err)
    }
    if delta.Done {
        break
    }
    if delta.Text != "" {
        log.Print(delta.Text)
    }
}
```

### Streaming Over HTTP (SSE)

The `ai` package provides a helper to write a `TextStream` as Server-Sent Events:

```go
http.HandleFunc("/stream", func(w http.ResponseWriter, r *http.Request) {
    ctx := r.Context()

    stream, err := ai.StreamText(ctx, ai.GenerateTextRequest{
        Model: model,
        Messages: []ai.Message{{
            Role:    ai.RoleUser,
            Content: "Stream over SSE.",
        }},
    })
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }

    if err := ai.WriteTextStreamAsSSE(ctx, w, stream); err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
    }
})
```

### Embeddings

```go
embModel := client.EmbeddingModel("text-embedding-3-small")

embRes, err := embModel.Generate(ctx, &provider.EmbeddingRequest{
    Input:  []string{"hello", "world"},
    UserID: "user-123",
})
if err != nil {
    log.Fatalf("embedding error: %v", err)
}

log.Printf("got %d embeddings\n", len(embRes.Embeddings))
```

## OpenAI-Compatible Example

See `examples/compat_text` for a small program that targets OpenAI-compatible backends. Example usage:

```bash
export OPENAI_COMPATIBLE_BASE_URL="https://api.ai.it.ufl.edu/v1"
export OPENAI_COMPATIBLE_API_KEY="sk-..."
export COMPAT_MODEL_ID="gpt-oss-20b"  # optional, default is gpt-oss-20b

go run ./examples/compat_text --prompt "Say hello from a compatible backend."
```

## Other Examples

- `examples/http_server` – basic `net/http` handler using `GenerateText`.
- `examples/cli_stream` – CLI program streaming output to stdout.
- `examples/fiber_stream` – Fiber v2 example streaming SSE-style responses.

## Roadmap (High-Level)

Planned areas for future work (non-binding):

- Additional providers (Anthropic, Groq, etc.) plugging into the same `provider` interfaces.
- Additional modalities (images, audio, video) when the text APIs are mature.
- More advanced features:
  - Batching and retries.
  - Metrics, tracing, and structured logging hooks.
  - Rate limiting and backoff strategies.
- Expanded test coverage and CI integration.
