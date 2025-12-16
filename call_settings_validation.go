package ai

import "fmt"

// NewCallSettings constructs a CallSettings instance and performs
// basic validation on the provided parameters. It returns an
// InvalidArgumentError for values that are clearly out of range.
//
// This helper is optional: callers can still construct CallSettings
// directly when they prefer not to perform validation.
func NewCallSettings(temperature *float64, topP *float64, maxTokens *int, stop []string) (*CallSettings, error) {
	if temperature != nil {
		if *temperature < 0 || *temperature > 2 {
			return nil, &InvalidArgumentError{
				Parameter: "temperature",
				Value:     *temperature,
				Message:   "must be between 0 and 2",
			}
		}
	}
	if topP != nil {
		if *topP <= 0 || *topP > 1 {
			return nil, &InvalidArgumentError{
				Parameter: "topP",
				Value:     *topP,
				Message:   "must be in the range (0, 1]",
			}
		}
	}
	if maxTokens != nil {
		if *maxTokens <= 0 {
			return nil, &InvalidArgumentError{
				Parameter: "maxTokens",
				Value:     *maxTokens,
				Message:   "must be greater than 0",
			}
		}
	}

	// No validation for stop sequences; providers may impose limits.

	return &CallSettings{
		Temperature: temperature,
		TopP:        topP,
		MaxTokens:   maxTokens,
		Stop:        stop,
	}, nil
}

// MustNewCallSettings constructs CallSettings and panics if validation
// fails. It is intended for configuration that should be validated at
// startup, not for user input.
func MustNewCallSettings(temperature *float64, topP *float64, maxTokens *int, stop []string) *CallSettings {
	cs, err := NewCallSettings(temperature, topP, maxTokens, stop)
	if err != nil {
		panic(fmt.Sprintf("ai: invalid call settings: %v", err))
	}
	return cs
}
