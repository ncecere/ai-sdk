package groq

import (
	"fmt"
	"os"
	"strings"

	"github.com/ncecere/ai-sdk/openai"
	"github.com/ncecere/ai-sdk/provider"
)

// NewClient creates a new Groq client by configuring the existing OpenAI
// client with Groq-specific defaults.
//
// This mirrors the behavior of the TypeScript @ai-sdk/groq provider,
// which targets the Groq OpenAI-compatible endpoint
// https://api.groq.com/openai/v1.
//
// Environment variables:
//   - GROQ_API_KEY  (used if opts.APIKey is empty)
//   - GROQ_BASE_URL (optional, defaults to https://api.groq.com/openai/v1)
func NewClient(opts provider.ClientOptions) (*openai.Client, error) {
	if opts.APIKey == "" {
		opts.APIKey = os.Getenv("GROQ_API_KEY")
	}
	if opts.APIKey == "" {
		return nil, fmt.Errorf("groq: missing API key; set ClientOptions.APIKey or GROQ_API_KEY")
	}

	if opts.BaseURL == "" {
		// Allow overriding the base URL via GROQ_BASE_URL, otherwise default
		// to the documented Groq OpenAI-compatible endpoint.
		baseURL := os.Getenv("GROQ_BASE_URL")
		if baseURL == "" {
			baseURL = "https://api.groq.com/openai/v1"
		}
		opts.BaseURL = strings.TrimRight(baseURL, "/")
	}

	return openai.NewClient(opts)
}
