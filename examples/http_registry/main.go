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

// http_registry is a minimal HTTP example that demonstrates how to use
// the ai-sdk registry package to decouple application code from concrete
// provider implementations.
//
// It expects:
//
//	OPENAI_API_KEY  - your OpenAI (or compatible) API key
//	OPENAI_BASE_URL - optional, for OpenAI-compatible endpoints
//
// The server listens on :8083 and exposes:
//
//	GET /chat?model=<name>&prompt=<text>
//
// Where model is a logical name registered in the in-memory registry
// (for example, "chat:default"). If model is omitted, "chat:default"
// is used.
func main() {
	if os.Getenv("OPENAI_API_KEY") == "" {
		log.Fatal("OPENAI_API_KEY must be set")
	}

	client, err := openai.NewClient(provider.ClientOptions{})
	if err != nil {
		log.Fatalf("failed to create OpenAI client: %v", err)
	}

	// Wire models into a registry so the rest of the application only
	// needs to know about logical names, not concrete provider packages.
	reg := registry.NewInMemoryRegistry()
	reg.RegisterLanguageModel("chat:default", client.ChatModel("gpt-4o-mini"))

	http.HandleFunc("/chat", func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()

		modelName := r.URL.Query().Get("model")
		if modelName == "" {
			modelName = "chat:default"
		}

		prompt := r.URL.Query().Get("prompt")
		if prompt == "" {
			prompt = "Hello from ai-sdk registry example!"
		}

		res, err := ai.GenerateTextWithRegistry(ctx, reg, modelName, ai.GenerateTextRequest{
			Messages: []ai.Message{{
				Role:    ai.RoleUser,
				Content: prompt,
			}},
		})
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			_ = json.NewEncoder(w).Encode(map[string]any{"error": err.Error()})
			return
		}

		_ = json.NewEncoder(w).Encode(map[string]any{
			"model":  modelName,
			"prompt": prompt,
			"text":   res.Text,
		})
	})

	log.Println("registry-based chat server listening on :8083/chat?prompt=...&model=chat:default")
	log.Fatal(http.ListenAndServe(":8083", nil))
}
