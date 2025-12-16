package provider

import (
	"context"
	"net/http"
)

// HTTPClient is the minimal interface required from an HTTP client.
// It matches the Do method on *http.Client and allows callers to
// substitute custom clients or middleware.
type HTTPClient interface {
	Do(req *http.Request) (*http.Response, error)
}

// ClientOptions are shared options for all provider clients.
// Providers typically accept these options in their constructors.
type ClientOptions struct {
	// BaseURL is the root URL of the provider API.
	BaseURL string
	// APIKey is the API key or bearer token used for authentication.
	APIKey string
	// HTTPClient is the underlying HTTP client. If nil, a default
	// client should be used by the provider.
	HTTPClient HTTPClient
	// Headers contains additional HTTP headers that providers should
	// attach to every outbound request. Provider implementations
	// decide how these interact with their own required headers.
	Headers http.Header
}

// LanguageModel is the low-level provider-facing interface for chat models.
// Implementations are responsible for mapping LanguageModelRequest values
// to the provider's chat/completions API.
type LanguageModel interface {
	Generate(ctx context.Context, req *LanguageModelRequest) (*LanguageModelResponse, error)
	Stream(ctx context.Context, req *LanguageModelRequest) (LanguageModelStream, error)
}

// LanguageModelRequest is a provider-level request structure close to
// the wire format used by chat APIs.
type LanguageModelRequest struct {
	Model       string
	Messages    []Message
	Temperature *float64
	TopP        *float64
	MaxTokens   *int
	Stop        []string
	JSONSchema  []byte
	Tools       []ToolDefinition
}

// Message is a provider-level chat message.
// Providers are free to map Role and Content to whatever structure
// their HTTP API expects.
type Message struct {
	Role    string
	Content string
}

// ToolDefinition describes a tool with JSON schema parameters.
// The Parameters byte slice typically contains a JSON Schema document.
type ToolDefinition struct {
	Name        string
	Description string
	Parameters  []byte
}

// ToolCall represents a tool invocation emitted by the model.
// RawArguments contains the JSON-encoded arguments payload.
type ToolCall struct {
	ID           string
	Name         string
	RawArguments []byte
}

// LanguageModelResponse is a provider-level response from a chat model.
type LanguageModelResponse struct {
	Text       string
	StopReason string
	ToolCalls  []ToolCall
}

// LanguageModelStream represents an incremental streaming interface.
// Next should block until a new delta is available or the stream ends.
type LanguageModelStream interface {
	Next(ctx context.Context) (*LanguageModelDelta, error)
	Close() error
}

// LanguageModelDelta is a single streamed update from a chat model.
type LanguageModelDelta struct {
	Text      string
	ToolCalls []ToolCall
	Done      bool
}

// EmbeddingModel is the provider-level interface for embeddings.
// Implementations map EmbeddingRequest to the provider's embedding API.
type EmbeddingModel interface {
	Generate(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error)
}

// EmbeddingRequest describes inputs for embeddings.
type EmbeddingRequest struct {
	Model  string
	Input  []string
	UserID string
}

// EmbeddingResponse contains embedding vectors.
type EmbeddingResponse struct {
	Embeddings [][]float32
}
