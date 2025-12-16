package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// GenerateObject generates a structured object using a language model
// and JSON schema. It infers a JSON schema for the target type T when
// none is provided and decodes the model output into a Go value of
// type T.
//
// This helper is built on top of GenerateText and the provider's
// JSON schema / JSON mode support.
func GenerateObject[T any](ctx context.Context, model LanguageModel, messages []Message) (T, error) {
	var zero T

	schema, err := JSONSchemaFromType(zero)
	if err != nil {
		return zero, fmt.Errorf("ai: building JSON schema for object: %w", err)
	}

	res, err := GenerateText(ctx, GenerateTextRequest{
		Model:      model,
		Messages:   messages,
		JSONSchema: schema,
	})
	if err != nil {
		return zero, err
	}

	text := strings.TrimSpace(res.Text)
	if text == "" {
		return zero, ErrNoObjectGenerated
	}

	var out T
	if err := json.Unmarshal([]byte(text), &out); err != nil {
		// Wrap JSON errors in a typed error for callers that want to
		// distinguish parsing failures from model failures.
		return zero, fmt.Errorf("%w: %v", ErrInvalidObjectJSON, err)
	}

	return out, nil
}

// DecodeToolCallArgs decodes the JSON arguments of a ToolCall into v.
// It is a small convenience helper around json.Unmarshal.
func DecodeToolCallArgs[T any](tc ToolCall, v *T) error {
	if len(tc.RawArguments) == 0 {
		return errors.New("ai: tool call has no arguments")
	}
	if err := json.Unmarshal(tc.RawArguments, v); err != nil {
		return fmt.Errorf("ai: decoding tool call arguments: %w", err)
	}
	return nil
}
