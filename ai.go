package ai

import (
	"context"

	"github.com/ncecere/ai-sdk/provider"
)

// Role constants for chat messages.
// These match the roles used by OpenAI-style chat endpoints.
const (
	RoleUser      = "user"
	RoleSystem    = "system"
	RoleAssistant = "assistant"
	RoleTool      = "tool"
)

// Aliases to provider-level types so users can work through the ai package
// while providers implement the shared interfaces.
type (
	// Message is a single chat message with role and content.
	Message = provider.Message
	// ToolDefinition describes a callable tool with a JSON schema.
	ToolDefinition = provider.ToolDefinition
	// ToolCall represents a tool invocation emitted by the model.
	ToolCall = provider.ToolCall

	// LanguageModel is a provider-agnostic chat-oriented model.
	LanguageModel = provider.LanguageModel
	// EmbeddingModel is a provider-agnostic embedding model.
	EmbeddingModel = provider.EmbeddingModel

	// TextDelta is a single streamed text update.
	TextDelta = provider.LanguageModelDelta
	// TextStream is an iterator-style stream of text deltas.
	TextStream = provider.LanguageModelStream
)

// Tool calling pattern
//
// A typical tool-calling loop with this package looks like:
//
//  1. Define tools with JSON schemas using ToolDefinition and, if desired,
//     JSONSchemaFromType to build the schema from a Go struct.
//  2. Call GenerateText or StreamText with Tools populated.
//  3. Inspect the ToolCalls field on the response (or TextDelta) and for
//     each ToolCall:
//     - Decode ToolCall.RawArguments into a Go struct.
//     - Execute the corresponding tool in your application.
//     - Append a new Message with RoleTool whose Content contains the
//       JSON-encoded tool result and whose name is represented in the
//       tool metadata (e.g., in the arguments you pass back to the model).
//  4. Call GenerateText again with the extended Messages slice to let the
//     model continue the conversation with tool results included.
//
// This mirrors the OpenAI function and tool-calling semantics while
// keeping execution of the tools firmly in your own Go code.

// GenerateTextRequest is a high-level request for text generation.
type GenerateTextRequest struct {
	// Model is the language model used to generate the response.
	Model LanguageModel
	// Messages is the ordered chat history passed to the model.
	Messages []Message
	// Temperature controls randomness of the output.
	Temperature *float64
	// TopP controls nucleus sampling for the output.
	TopP *float64
	// MaxTokens limits the number of tokens produced.
	MaxTokens *int
	// Stop contains stop sequences that will truncate the output.
	Stop []string
	// JSONSchema, if set, requests a structured JSON response from the model.
	JSONSchema []byte
	// Tools defines tools the model may call during generation.
	Tools []ToolDefinition
}

// GenerateTextResponse is the result of a non-streaming text generation call.
type GenerateTextResponse struct {
	// Text is the concatenated assistant text returned by the model.
	Text string
	// StopReason describes why generation stopped (if available).
	StopReason string
	// ToolCalls contains any tool invocations emitted by the model.
	ToolCalls []ToolCall
}

// GenerateText calls the underlying LanguageModel.Generate and returns a
// simplified response structure.
//
// Errors:
//   - ErrMissingModel if req.Model is nil.
//   - Any error returned by the underlying provider implementation. For
//     the OpenAI provider this includes HTTP and JSON decoding errors
//     originating from the OpenAI API.
func GenerateText(ctx context.Context, req GenerateTextRequest) (GenerateTextResponse, error) {
	if req.Model == nil {
		return GenerateTextResponse{}, ErrMissingModel
	}

	lmReq := &provider.LanguageModelRequest{
		Messages:    req.Messages,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		MaxTokens:   req.MaxTokens,
		Stop:        req.Stop,
		JSONSchema:  req.JSONSchema,
		Tools:       req.Tools,
	}

	lmRes, err := req.Model.Generate(ctx, lmReq)
	if err != nil {
		return GenerateTextResponse{}, err
	}

	return GenerateTextResponse{
		Text:       lmRes.Text,
		StopReason: lmRes.StopReason,
		ToolCalls:  lmRes.ToolCalls,
	}, nil
}

// StreamText calls the underlying LanguageModel.Stream and returns a
// TextStream that yields incremental deltas until Done is true.
//
// Errors:
//   - ErrMissingModel if req.Model is nil.
//   - Any error returned by the underlying provider implementation when
//     establishing the stream.
func StreamText(ctx context.Context, req GenerateTextRequest) (TextStream, error) {
	if req.Model == nil {
		return nil, ErrMissingModel
	}

	lmReq := &provider.LanguageModelRequest{
		Messages:    req.Messages,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		MaxTokens:   req.MaxTokens,
		Stop:        req.Stop,
		JSONSchema:  req.JSONSchema,
		Tools:       req.Tools,
	}

	return req.Model.Stream(ctx, lmReq)
}

// GenerateSimpleText is a convenience helper for the common case of
// a single user prompt and plain text response.
//
// It constructs a GenerateTextRequest with a single user message and
// returns only the response text.
func GenerateSimpleText(ctx context.Context, model LanguageModel, prompt string) (string, error) {
	res, err := GenerateText(ctx, GenerateTextRequest{
		Model: model,
		Messages: []Message{{
			Role:    RoleUser,
			Content: prompt,
		}},
	})
	if err != nil {
		return "", err
	}
	return res.Text, nil
}

// EmbeddingRequest describes an embedding generation request.
type EmbeddingRequest struct {
	// Model is the embedding model used to generate vectors.
	Model EmbeddingModel
	// Input is the list of text inputs to embed.
	Input []string
	// UserID is an optional identifier used for provider-side logging.
	UserID string
}

// EmbeddingResponse contains embedding vectors.
type EmbeddingResponse struct {
	// Embeddings is a slice of embedding vectors, one per input.
	Embeddings [][]float32
}

// GenerateEmbeddings calls the underlying EmbeddingModel.Generate and
// returns the resulting vectors.
//
// Errors:
//   - ErrMissingEmbeddingModel if req.Model is nil.
//   - Any error returned by the underlying provider implementation.
func GenerateEmbeddings(ctx context.Context, req EmbeddingRequest) (EmbeddingResponse, error) {
	if req.Model == nil {
		return EmbeddingResponse{}, ErrMissingEmbeddingModel
	}

	embReq := &provider.EmbeddingRequest{
		Input:  req.Input,
		UserID: req.UserID,
	}

	embRes, err := req.Model.Generate(ctx, embReq)
	if err != nil {
		return EmbeddingResponse{}, err
	}

	return EmbeddingResponse{Embeddings: embRes.Embeddings}, nil
}
