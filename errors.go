package ai

import "errors"

// Package-level error values returned by the ai package.
var (
	// ErrMissingModel is returned when a GenerateText or StreamText
	// request does not specify a LanguageModel.
	ErrMissingModel = errors.New("ai: missing LanguageModel in request")

	// ErrMissingEmbeddingModel is returned when a GenerateEmbeddings
	// request does not specify an EmbeddingModel.
	ErrMissingEmbeddingModel = errors.New("ai: missing EmbeddingModel in request")
)
