package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	ai "github.com/ncecere/ai-sdk"
	"github.com/ncecere/ai-sdk/openai"
	"github.com/ncecere/ai-sdk/provider"
)

// This example is intended for OpenAI-compatible backends such as the
// UFL AI gateway, e.g.:
//
//	base URL: https://api.ai.it.ufl.edu/v1
//	model:    gpt-oss-20b
//
// It uses the following environment variables:
//
//	OPENAI_COMPATIBLE_BASE_URL  - required (e.g. https://api.ai.it.ufl.edu/v1)
//	OPENAI_COMPATIBLE_API_KEY   - optional, falls back to OPENAI_API_KEY
//	COMPAT_MODEL_ID             - optional, defaults to gpt-oss-20b
func main() {
	baseURL := os.Getenv("OPENAI_COMPATIBLE_BASE_URL")
	if baseURL == "" {
		log.Fatal("OPENAI_COMPATIBLE_BASE_URL must be set (e.g. https://api.ai.it.ufl.edu/v1)")
	}
	apiKey := os.Getenv("OPENAI_COMPATIBLE_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}
	if apiKey == "" {
		log.Fatal("OPENAI_COMPATIBLE_API_KEY or OPENAI_API_KEY must be set")
	}

	modelID := os.Getenv("COMPAT_MODEL_ID")
	if modelID == "" {
		modelID = "gpt-oss-20b"
	}

	prompt := flag.String("prompt", "Say hello from the UFL-compatible backend.", "prompt text")
	flag.Parse()

	client, err := openai.NewClient(provider.ClientOptions{
		BaseURL:    baseURL,
		APIKey:     apiKey,
		HTTPClient: openai.WithHTTPTimeout(30 * time.Second),
	})
	if err != nil {
		log.Fatalf("failed to create OpenAI-compatible client: %v", err)
	}

	model := client.ChatModel(modelID)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	text, err := ai.GenerateSimpleText(ctx, model, *prompt)
	if err != nil {
		log.Fatalf("generation error: %v", err)
	}

	fmt.Println(text)
}
