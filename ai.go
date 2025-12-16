package ai

import (
	"context"

	"github.com/ncecere/ai-sdk/provider"
	"github.com/ncecere/ai-sdk/registry"
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
	// CompletionModel is a provider-agnostic completion-style model.
	CompletionModel = provider.CompletionModel
	// ImageModel is a provider-agnostic image generation model.
	ImageModel = provider.ImageModel
	// SpeechModel is a provider-agnostic text-to-speech model.
	SpeechModel = provider.SpeechModel
	// TranscriptionModel is a provider-agnostic speech-to-text model.
	TranscriptionModel = provider.TranscriptionModel
	// RerankModel is a provider-agnostic rerank model.
	RerankModel = provider.RerankModel

	// Image is a generated image returned by image models.
	Image = provider.Image
	// RerankResult is a single scored document returned by rerank models.
	RerankResult = provider.RerankResult

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

// GenerateSimpleTextWithRegistry is a convenience helper for the common
// case of a single user prompt and plain text response when using a
// registry to look up the model by name.
//
// It constructs a GenerateTextRequest with a single user message and
// delegates to GenerateTextWithRegistry.
func GenerateSimpleTextWithRegistry(ctx context.Context, reg registry.Registry, modelName, prompt string) (string, error) {
	res, err := GenerateTextWithRegistry(ctx, reg, modelName, GenerateTextRequest{
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

// GenerateTextWithRegistry is a convenience helper that looks up the
// language model by name in the provided registry and then delegates
// to GenerateText. Any Model value in req is ignored and replaced
// with the resolved model.
//
// Errors:
//   - InvalidArgumentError if reg is nil.
//   - Any error returned by reg.LanguageModel.
//   - Any error returned by GenerateText.
func GenerateTextWithRegistry(ctx context.Context, reg registry.Registry, modelName string, req GenerateTextRequest) (GenerateTextResponse, error) {
	if reg == nil {
		return GenerateTextResponse{}, &InvalidArgumentError{Parameter: "reg", Value: nil, Message: "registry must not be nil"}
	}

	lm, err := reg.LanguageModel(modelName)
	if err != nil {
		return GenerateTextResponse{}, err
	}

	req.Model = lm
	return GenerateText(ctx, req)
}

// StreamTextWithRegistry is a convenience helper that looks up the
// language model by name in the provided registry and then delegates
// to StreamText. Any Model value in req is ignored and replaced with
// the resolved model.
//
// Errors:
//   - InvalidArgumentError if reg is nil.
//   - Any error returned by reg.LanguageModel.
//   - Any error returned by StreamText when establishing the stream.
func StreamTextWithRegistry(ctx context.Context, reg registry.Registry, modelName string, req GenerateTextRequest) (TextStream, error) {
	if reg == nil {
		return nil, &InvalidArgumentError{Parameter: "reg", Value: nil, Message: "registry must not be nil"}
	}

	lm, err := reg.LanguageModel(modelName)
	if err != nil {
		return nil, err
	}

	req.Model = lm
	return StreamText(ctx, req)
}

// CompletionRequest describes a completion-style text generation request.
type CompletionRequest struct {
	// Model is the completion model used to generate the response.
	Model CompletionModel
	// Prompt is the input text for the completion.
	Prompt string
	// Temperature controls randomness of the output.
	Temperature *float64
	// TopP controls nucleus sampling for the output.
	TopP *float64
	// MaxTokens limits the number of tokens produced.
	MaxTokens *int
	// Stop contains stop sequences that will truncate the output.
	Stop []string
	// UserID is an optional identifier used for provider-side logging.
	UserID string
}

// CompletionResponse is the result of a completion-style text generation call.
type CompletionResponse struct {
	// Text is the generated completion text.
	Text string
	// StopReason describes why generation stopped (if available).
	StopReason string
}

// GenerateCompletion calls the underlying CompletionModel.Generate and returns
// a simplified response structure.
//
// Errors:
//   - ErrMissingModel if req.Model is nil.
//   - Any error returned by the underlying provider implementation.
func GenerateCompletion(ctx context.Context, req CompletionRequest) (CompletionResponse, error) {
	if req.Model == nil {
		return CompletionResponse{}, ErrMissingModel
	}

	cReq := &provider.CompletionRequest{
		Prompt:      req.Prompt,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		MaxTokens:   req.MaxTokens,
		Stop:        req.Stop,
		UserID:      req.UserID,
	}

	cRes, err := req.Model.Generate(ctx, cReq)
	if err != nil {
		return CompletionResponse{}, err
	}

	return CompletionResponse{
		Text:       cRes.Text,
		StopReason: cRes.StopReason,
	}, nil
}

// GenerateCompletionWithRegistry is a convenience helper that looks up the
// completion model by name in the provided registry and then delegates to
// GenerateCompletion. Any Model value in req is ignored and replaced with the
// resolved model.
//
// Errors:
//   - InvalidArgumentError if reg is nil.
//   - Any error returned by reg.CompletionModel (when additional providers
//     expose them via the registry).
//   - Any error returned by GenerateCompletion.
func GenerateCompletionWithRegistry(ctx context.Context, reg registry.Registry, modelName string, req CompletionRequest) (CompletionResponse, error) {
	if reg == nil {
		return CompletionResponse{}, &InvalidArgumentError{Parameter: "reg", Value: nil, Message: "registry must not be nil"}
	}

	model, err := reg.CompletionModel(modelName)
	if err != nil {
		return CompletionResponse{}, err
	}

	req.Model = model
	return GenerateCompletion(ctx, req)
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

// GenerateEmbeddingsWithRegistry is a convenience helper that looks up
// the embedding model by name in the provided registry and then
// delegates to GenerateEmbeddings. Any Model value in req is ignored and
// replaced with the resolved model.
//
// Errors:
//   - InvalidArgumentError if reg is nil.
//   - Any error returned by reg.EmbeddingModel.
//   - Any error returned by GenerateEmbeddings.
func GenerateEmbeddingsWithRegistry(ctx context.Context, reg registry.Registry, modelName string, req EmbeddingRequest) (EmbeddingResponse, error) {
	if reg == nil {
		return EmbeddingResponse{}, &InvalidArgumentError{Parameter: "reg", Value: nil, Message: "registry must not be nil"}
	}

	model, err := reg.EmbeddingModel(modelName)
	if err != nil {
		return EmbeddingResponse{}, err
	}

	req.Model = model
	return GenerateEmbeddings(ctx, req)
}

// ImageRequest describes an image generation request.
type ImageRequest struct {
	// Model is the image model used to generate images.
	Model ImageModel
	// Prompt is the primary text prompt used to generate the image.
	Prompt string
	// Size is an optional size hint (e.g. "1024x1024").
	Size string
	// NumberOfImages controls how many images to generate. Zero means provider default.
	NumberOfImages int
	// ResponseFormat controls how images are returned (e.g. "url", "b64_json").
	ResponseFormat string
	// UserID is an optional identifier used for provider-side logging.
	UserID string
}

// ImageResponse contains generated images.
type ImageResponse struct {
	// Images is the set of generated images.
	Images []Image
}

// GenerateImage calls the underlying ImageModel.Generate and returns generated images.
//
// Errors:
//   - ErrMissingModel if req.Model is nil.
//   - Any error returned by the underlying provider implementation.
func GenerateImage(ctx context.Context, req ImageRequest) (ImageResponse, error) {
	if req.Model == nil {
		return ImageResponse{}, ErrMissingModel
	}

	imgReq := &provider.ImageRequest{
		Prompt:         req.Prompt,
		Size:           req.Size,
		NumberOfImages: req.NumberOfImages,
		ResponseFormat: req.ResponseFormat,
		UserID:         req.UserID,
	}

	imgRes, err := req.Model.Generate(ctx, imgReq)
	if err != nil {
		return ImageResponse{}, err
	}

	return ImageResponse{Images: imgRes.Images}, nil
}

// GenerateImageWithRegistry is a convenience helper that looks up the
// image model by name in the provided registry and then delegates to
// GenerateImage. Any Model value in req is ignored and replaced with the
// resolved model.
//
// Errors:
//   - InvalidArgumentError if reg is nil.
//   - Any error returned by reg.ImageModel.
//   - Any error returned by GenerateImage.
func GenerateImageWithRegistry(ctx context.Context, reg registry.Registry, modelName string, req ImageRequest) (ImageResponse, error) {
	if reg == nil {
		return ImageResponse{}, &InvalidArgumentError{Parameter: "reg", Value: nil, Message: "registry must not be nil"}
	}

	model, err := reg.ImageModel(modelName)
	if err != nil {
		return ImageResponse{}, err
	}

	req.Model = model
	return GenerateImage(ctx, req)
}

// SpeechRequest describes a text-to-speech generation request.
type SpeechRequest struct {
	// Model is the speech model used to generate audio.
	Model SpeechModel
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

// SpeechResponse contains synthesized audio.
type SpeechResponse struct {
	// Audio is the synthesized audio bytes.
	Audio []byte
	// MimeType is the content type of the audio payload (e.g. "audio/mpeg").
	MimeType string
}

// GenerateSpeech calls the underlying SpeechModel.Generate and returns synthesized audio.
//
// Errors:
//   - ErrMissingModel if req.Model is nil.
//   - Any error returned by the underlying provider implementation.
func GenerateSpeech(ctx context.Context, req SpeechRequest) (SpeechResponse, error) {
	if req.Model == nil {
		return SpeechResponse{}, ErrMissingModel
	}

	spReq := &provider.SpeechRequest{
		Input:    req.Input,
		Voice:    req.Voice,
		Format:   req.Format,
		Language: req.Language,
		UserID:   req.UserID,
	}

	spRes, err := req.Model.Generate(ctx, spReq)
	if err != nil {
		return SpeechResponse{}, err
	}

	return SpeechResponse{
		Audio:    spRes.Audio,
		MimeType: spRes.MimeType,
	}, nil
}

// GenerateSpeechWithRegistry is a convenience helper that looks up the
// speech model by name in the provided registry and then delegates to
// GenerateSpeech. Any Model value in req is ignored and replaced with
// the resolved model.
//
// Errors:
//   - InvalidArgumentError if reg is nil.
//   - Any error returned by reg.SpeechModel.
//   - Any error returned by GenerateSpeech.
func GenerateSpeechWithRegistry(ctx context.Context, reg registry.Registry, modelName string, req SpeechRequest) (SpeechResponse, error) {
	if reg == nil {
		return SpeechResponse{}, &InvalidArgumentError{Parameter: "reg", Value: nil, Message: "registry must not be nil"}
	}

	model, err := reg.SpeechModel(modelName)
	if err != nil {
		return SpeechResponse{}, err
	}

	req.Model = model
	return GenerateSpeech(ctx, req)
}

// TranscriptionRequest describes a speech-to-text transcription request.
type TranscriptionRequest struct {
	// Model is the transcription model used to produce text.
	Model TranscriptionModel
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
	// Text is the transcribed text.
	Text string
}

// Transcribe calls the underlying TranscriptionModel.Generate and returns the transcription text.
//
// Errors:
//   - ErrMissingModel if req.Model is nil.
//   - Any error returned by the underlying provider implementation.
func Transcribe(ctx context.Context, req TranscriptionRequest) (TranscriptionResponse, error) {
	if req.Model == nil {
		return TranscriptionResponse{}, ErrMissingModel
	}

	trReq := &provider.TranscriptionRequest{
		Audio:       req.Audio,
		FileName:    req.FileName,
		MimeType:    req.MimeType,
		Language:    req.Language,
		Prompt:      req.Prompt,
		Temperature: req.Temperature,
		UserID:      req.UserID,
	}

	trRes, err := req.Model.Generate(ctx, trReq)
	if err != nil {
		return TranscriptionResponse{}, err
	}

	return TranscriptionResponse{
		Text: trRes.Text,
	}, nil
}

// TranscribeWithRegistry is a convenience helper that looks up the
// transcription model by name in the provided registry and then
// delegates to Transcribe. Any Model value in req is ignored and
// replaced with the resolved model.
//
// Errors:
//   - InvalidArgumentError if reg is nil.
//   - Any error returned by reg.TranscriptionModel.
//   - Any error returned by Transcribe.
func TranscribeWithRegistry(ctx context.Context, reg registry.Registry, modelName string, req TranscriptionRequest) (TranscriptionResponse, error) {
	if reg == nil {
		return TranscriptionResponse{}, &InvalidArgumentError{Parameter: "reg", Value: nil, Message: "registry must not be nil"}
	}

	model, err := reg.TranscriptionModel(modelName)
	if err != nil {
		return TranscriptionResponse{}, err
	}

	req.Model = model
	return Transcribe(ctx, req)
}

// RerankRequest describes a reranking request over a set of documents.
type RerankRequest struct {
	// Model is the rerank model used to score documents.
	Model RerankModel
	// Query is the query text used to evaluate document relevance.
	Query string
	// Documents is the list of documents or passages to score.
	Documents []string
	// TopK limits the number of results returned. Zero means provider default.
	TopK int
	// UserID is an optional identifier used for provider-side logging.
	UserID string
}

// RerankResponse contains reranked results ordered by score.
type RerankResponse struct {
	// Results is the list of scored documents ordered by descending score.
	Results []RerankResult
}

// Rerank calls the underlying RerankModel.Generate and returns scored documents.
//
// Errors:
//   - ErrMissingModel if req.Model is nil.
//   - Any error returned by the underlying provider implementation.
func Rerank(ctx context.Context, req RerankRequest) (RerankResponse, error) {
	if req.Model == nil {
		return RerankResponse{}, ErrMissingModel
	}

	rrReq := &provider.RerankRequest{
		Query:     req.Query,
		Documents: req.Documents,
		TopK:      req.TopK,
		UserID:    req.UserID,
	}

	rrRes, err := req.Model.Generate(ctx, rrReq)
	if err != nil {
		return RerankResponse{}, err
	}

	return RerankResponse{
		Results: rrRes.Results,
	}, nil
}

// RerankWithRegistry is a convenience helper that looks up the rerank
// model by name in the provided registry and then delegates to Rerank.
// Any Model value in req is ignored and replaced with the resolved
// model.
//
// Errors:
//   - InvalidArgumentError if reg is nil.
//   - Any error returned by reg.RerankModel.
//   - Any error returned by Rerank.
func RerankWithRegistry(ctx context.Context, reg registry.Registry, modelName string, req RerankRequest) (RerankResponse, error) {
	if reg == nil {
		return RerankResponse{}, &InvalidArgumentError{Parameter: "reg", Value: nil, Message: "registry must not be nil"}
	}

	model, err := reg.RerankModel(modelName)
	if err != nil {
		return RerankResponse{}, err
	}

	req.Model = model
	return Rerank(ctx, req)
}
