package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"time"

	ai "github.com/ncecere/ai-sdk"
	"github.com/ncecere/ai-sdk/agent"
	"github.com/ncecere/ai-sdk/openai"
	"github.com/ncecere/ai-sdk/provider"
	"github.com/ncecere/ai-sdk/registry"
)

// http_agent exposes a simple agent endpoint over HTTP using SSE.
func main() {
	if os.Getenv("OPENAI_API_KEY") == "" {
		log.Fatal("OPENAI_API_KEY must be set")
	}

	client, err := openai.NewClient(provider.ClientOptions{})
	if err != nil {
		log.Fatalf("failed to create OpenAI client: %v", err)
	}

	reg := registry.NewInMemoryRegistry()
	reg.RegisterLanguageModel("chat:default", client.ChatModel("gpt-4o-mini"))

	// Simple echo tool that returns the provided text.
	tools := map[string]agent.Tool{
		"echo": {
			Name:        "echo",
			Description: "Echo back the provided text",
			Execute: func(ctx context.Context, args json.RawMessage) (any, error) {
				var payload struct {
					Text string `json:"text"`
				}
				if err := json.Unmarshal(args, &payload); err != nil {
					return nil, err
				}
				return map[string]any{"echo": payload.Text}, nil
			},
		},
	}

	http.HandleFunc("/agent", func(w http.ResponseWriter, r *http.Request) {
		prompt := r.URL.Query().Get("prompt")
		if prompt == "" {
			prompt = "Use the echo tool to repeat 'hello world'."
		}

		cfg := agent.Config{
			Registry:  reg,
			ModelName: "chat:default",
			Tools:     tools,
			MaxSteps:  8,
		}

		ctx, cancel := context.WithTimeout(r.Context(), 2*time.Minute)
		defer cancel()

		initial := []ai.Message{{
			Role:    ai.RoleUser,
			Content: prompt,
		}}

		_, err := agent.WriteRunAsSSE(ctx, w, cfg, initial)
		if err != nil {
			log.Printf("agent error: %v", err)
		}
	})

	log.Println("agent SSE server listening on :8084/agent?prompt=...")
	log.Fatal(http.ListenAndServe(":8084", nil))
}
