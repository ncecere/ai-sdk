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

// ImageModel is the provider-level interface for image generation.
// Implementations map ImageRequest values to the provider's image API.
type ImageModel interface {
	Generate(ctx context.Context, req *ImageRequest) (*ImageResponse, error)
}

// ImageRequest describes inputs for image generation.
type ImageRequest struct {
	// Model is the image model identifier.
	Model string
	// Prompt is the primary text prompt used to generate the image.
	Prompt string
	// Size is an optional provider-specific size hint (e.g. "1024x1024").
	Size string
	// NumberOfImages controls how many images to generate. Zero means provider default.
	NumberOfImages int
	// ResponseFormat controls how images are returned (e.g. "url", "b64_json").
	ResponseFormat string
	// UserID is an optional identifier used for provider-side logging.
	UserID string
}

// Image contains a single generated image.
type Image struct {
	// URL is set when the provider returns a remote image URL.
	URL string
	// Data contains raw image bytes when the provider returns binary data
	// or a decoded base64 payload.
	Data []byte
}

// ImageResponse contains generated images.
type ImageResponse struct {
	Images []Image
}

// SpeechModel is the provider-level interface for text-to-speech.
// Implementations map SpeechRequest values to the provider's TTS API.
type SpeechModel interface {
	Generate(ctx context.Context, req *SpeechRequest) (*SpeechResponse, error)
}

// SpeechRequest describes inputs for text-to-speech.
type SpeechRequest struct {
	// Model is the speech model identifier.
	Model string
	// Input is the text to synthesize.
	Input string
	// Voice is an optional voice selection (provider-specific).
	Voice string
	// Format is the desired audio container/codec (e.g. "mp3", "wav").
	Format string
	// Language is an optional BCP-47 language tag.
	Language string
	// UserID is an optional identifier used for provider-side logging.
	UserID string
}

// SpeechResponse contains generated audio.
type SpeechResponse struct {
	// Audio is the synthesized audio bytes.
	Audio []byte
	// MimeType is the content type of the audio payload (e.g. "audio/mpeg").
	MimeType string
}

// TranscriptionModel is the provider-level interface for speech-to-text transcription.
// Implementations map TranscriptionRequest values to the provider's transcription API.
type TranscriptionModel interface {
	Generate(ctx context.Context, req *TranscriptionRequest) (*TranscriptionResponse, error)
}

// TranscriptionRequest describes inputs for transcription.
type TranscriptionRequest struct {
	// Model is the transcription model identifier.
	Model string
	// Audio is the audio payload to transcribe.
	Audio []byte
	// FileName is an optional original file name (used for metadata/content type hints).
	FileName string
	// MimeType is an optional content type for the audio payload.
	MimeType string
	// Language is an optional expected language for the transcription.
	Language string
	// Prompt is an optional text prompt or hint for the transcription.
	Prompt string
	// Temperature controls sampling for models that support it.
	Temperature *float64
	// UserID is an optional identifier used for provider-side logging.
	UserID string
}

// TranscriptionResponse contains the transcription text.
type TranscriptionResponse struct {
	Text string
}

// RerankModel is the provider-level interface for reranking documents.
// Implementations map RerankRequest values to the provider's rerank API.
type RerankModel interface {
	Generate(ctx context.Context, req *RerankRequest) (*RerankResponse, error)
}

// RerankRequest describes inputs for reranking operations.
type RerankRequest struct {
	// Model is the rerank model identifier.
	Model string
	// Query is the query text used to evaluate document relevance.
	Query string
	// Documents is the list of documents or passages to score.
	Documents []string
	// TopK limits the number of results returned. Zero means provider default.
	TopK int
	// UserID is an optional identifier used for provider-side logging.
	UserID string
}

// RerankResult represents a single reranked document.
type RerankResult struct {
	// Index is the index of the document in the original Documents slice.
	Index int
	// Score is the relevance score assigned by the model (higher is better).
	Score float64
}

// RerankResponse contains reranked results ordered by score.
type RerankResponse struct {
	Results []RerankResult
}
