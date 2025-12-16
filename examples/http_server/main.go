package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"

	"github.com/ncecere/ai-sdk"
	"github.com/ncecere/ai-sdk/openai"
	"github.com/ncecere/ai-sdk/provider"
)

func main() {
	if os.Getenv("OPENAI_API_KEY") == "" {
		log.Fatal("OPENAI_API_KEY must be set")
	}

	client, err := openai.NewClient(provider.ClientOptions{})
	if err != nil {
		log.Fatalf("failed to create OpenAI client: %v", err)
	}

	model := client.ChatModel("gpt-4o-mini")

	http.HandleFunc("/chat", func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()
		res, err := ai.GenerateText(ctx, ai.GenerateTextRequest{
			Model: model,
			Messages: []ai.Message{{
				Role:    "user",
				Content: "Hello from ai-sdk in Go!",
			}},
		})
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			_ = json.NewEncoder(w).Encode(map[string]any{"error": err.Error()})
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"text": res.Text})
	})

	log.Println("listening on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
