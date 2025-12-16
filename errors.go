package ai

import "errors"

// Package-level error values and types returned by the ai package.
var (
	// ErrMissingModel is returned when a GenerateText or StreamText
	// request does not specify a LanguageModel.
	ErrMissingModel = errors.New("ai: missing LanguageModel in request")

	// ErrMissingEmbeddingModel is returned when a GenerateEmbeddings
	// request does not specify an EmbeddingModel.
	ErrMissingEmbeddingModel = errors.New("ai: missing EmbeddingModel in request")

	// ErrNoObjectGenerated is returned by helpers such as GenerateObject
	// when the model does not produce any structured JSON output.
	ErrNoObjectGenerated = errors.New("ai: no object generated")

	// ErrInvalidObjectJSON is returned when a model response that is
	// expected to contain JSON cannot be decoded into the requested
	// Go type.
	ErrInvalidObjectJSON = errors.New("ai: generated text is not valid JSON for target type")

	// ErrNoEmbeddingGenerated is returned when an embedding request
	// completes successfully but does not return any vectors.
	ErrNoEmbeddingGenerated = errors.New("ai: no embedding generated")
)

// InvalidArgumentError indicates that a function argument is invalid.
// It is intended for validation of ai package helper arguments, such
// as call settings or prompt construction helpers.
type InvalidArgumentError struct {
	// Parameter is the name of the invalid parameter.
	Parameter string
	// Value is the offending value.
	Value any
	// Message describes why the value is considered invalid.
	Message string
}

func (e *InvalidArgumentError) Error() string {
	if e == nil {
		return "<nil>"
	}
	return "ai: invalid argument for parameter " + e.Parameter + ": " + e.Message
}

// UnsupportedFunctionalityError indicates that a requested feature is
// not supported by the current implementation.
type UnsupportedFunctionalityError struct {
	// Feature describes the unsupported feature, e.g. "image generation".
	Feature string
	// Message is an optional explanatory message.
	Message string
}

func (e *UnsupportedFunctionalityError) Error() string {
	if e == nil {
		return "<nil>"
	}
	if e.Message != "" {
		return "ai: unsupported functionality (" + e.Feature + "): " + e.Message
	}
	return "ai: unsupported functionality (" + e.Feature + ")"
}
