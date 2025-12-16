package ai

// CallSettings groups common generation parameters such as temperature,
// top-p, max tokens, and stop sequences. This is a convenience struct
// for sharing settings across multiple GenerateTextRequest values.
//
// CallSettings do not affect any requests automatically; callers are
// expected to apply them when constructing requests.
type CallSettings struct {
	// Temperature controls randomness of the output.
	Temperature *float64
	// TopP controls nucleus sampling for the output.
	TopP *float64
	// MaxTokens limits the number of tokens produced.
	MaxTokens *int
	// Stop contains stop sequences that will truncate the output.
	Stop []string
}

// ApplyTo copies the non-nil/non-zero fields from the CallSettings
// into the given GenerateTextRequest.
func (s *CallSettings) ApplyTo(req *GenerateTextRequest) {
	if s == nil {
		return
	}
	if s.Temperature != nil {
		req.Temperature = s.Temperature
	}
	if s.TopP != nil {
		req.TopP = s.TopP
	}
	if s.MaxTokens != nil {
		req.MaxTokens = s.MaxTokens
	}
	if len(s.Stop) > 0 {
		req.Stop = s.Stop
	}
}

// NewGenerateTextRequest constructs a GenerateTextRequest from the
// provided model, messages, and optional CallSettings. This avoids
// repeating call settings wiring at every call site.
func NewGenerateTextRequest(model LanguageModel, messages []Message, settings *CallSettings) GenerateTextRequest {
	req := GenerateTextRequest{
		Model:    model,
		Messages: messages,
	}
	if settings != nil {
		settings.ApplyTo(&req)
	}
	return req
}

// Helper constructors for common message types. These are small
// conveniences for building []Message slices.

// SystemMessage creates a system message with the given content.
func SystemMessage(content string) Message {
	return Message{Role: RoleSystem, Content: content}
}

// UserMessage creates a user message with the given content.
func UserMessage(content string) Message {
	return Message{Role: RoleUser, Content: content}
}

// AssistantMessage creates an assistant message with the given content.
func AssistantMessage(content string) Message {
	return Message{Role: RoleAssistant, Content: content}
}
