package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"

	ai "github.com/ncecere/ai-sdk"
	"github.com/ncecere/ai-sdk/openai"
	"github.com/ncecere/ai-sdk/provider"
	"github.com/ncecere/ai-sdk/registry"
)

// http_completion_registry is a minimal HTTP example that demonstrates
// using the registry with a completion-style model.
//
// It expects:
//
//	OPENAI_API_KEY  - your OpenAI (or compatible) API key
//	OPENAI_BASE_URL - optional, for OpenAI-compatible endpoints
//
// The server listens on :8085 and exposes:
//
//	GET /completion?model=<name>&prompt=<text>
//
// Where model is a logical name registered in the in-memory registry
// (for example, "completion:default"). If model is omitted,
// "completion:default" is used.
func main() {
	if os.Getenv("OPENAI_API_KEY") == "" {
		log.Fatal("OPENAI_API_KEY must be set")
	}

	client, err := openai.NewClient(provider.ClientOptions{})
	if err != nil {
		log.Fatalf("failed to create OpenAI client: %v", err)
	}

	reg := registry.NewInMemoryRegistry()
	// Register a completion-style model under a logical name. Adjust the
	// underlying model ID as needed for your account.
	reg.RegisterCompletionModel("completion:default", client.CompletionModel("gpt-3.5-turbo-instruct"))

	http.HandleFunc("/completion", func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()

		modelName := r.URL.Query().Get("model")
		if modelName == "" {
			modelName = "completion:default"
		}

		prompt := r.URL.Query().Get("prompt")
		if prompt == "" {
			prompt = "Write a short haiku about Go routines."
		}

		res, err := ai.GenerateCompletionWithRegistry(ctx, reg, modelName, ai.CompletionRequest{
			Prompt: prompt,
		})
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			_ = json.NewEncoder(w).Encode(map[string]any{"error": err.Error()})
			return
		}

		_ = json.NewEncoder(w).Encode(map[string]any{
			"model":      modelName,
			"prompt":     prompt,
			"text":       res.Text,
			"stopReason": res.StopReason,
		})
	})

	log.Println("completion registry server listening on :8085/completion?prompt=...&model=completion:default")
	log.Fatal(http.ListenAndServe(":8085", nil))
}
