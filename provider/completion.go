package provider

import "context"

// CompletionModel is the provider-level interface for completion-style
// models (legacy /v1/completions APIs).
//
// Implementations map CompletionRequest values to the provider's
// completion API.
type CompletionModel interface {
	Generate(ctx context.Context, req *CompletionRequest) (*CompletionResponse, error)
}

// CompletionRequest describes inputs for text completions.
type CompletionRequest struct {
	Model       string
	Prompt      string
	Temperature *float64
	TopP        *float64
	MaxTokens   *int
	Stop        []string
	UserID      string
}

// CompletionResponse contains the resulting completion text.
type CompletionResponse struct {
	Text       string
	StopReason string
}
