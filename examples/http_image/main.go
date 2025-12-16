package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"

	ai "github.com/ncecere/ai-sdk"
	"github.com/ncecere/ai-sdk/openai"
	"github.com/ncecere/ai-sdk/provider"
)

// http_image is a minimal HTTP example that exposes an
// endpoint for generating images from text prompts using
// the OpenAI image generation API via ai-sdk.
//
// It expects:
//
//	OPENAI_API_KEY  - your OpenAI (or compatible) API key
//	OPENAI_BASE_URL - optional, for OpenAI-compatible endpoints
//
// The server listens on :8082 and exposes:
//
//	GET /image?prompt=...
//
// The response is a small JSON object containing the
// prompt and any image URLs returned by the provider.
func main() {
	if os.Getenv("OPENAI_API_KEY") == "" {
		log.Fatal("OPENAI_API_KEY must be set")
	}

	client, err := openai.NewClient(provider.ClientOptions{})
	if err != nil {
		log.Fatalf("failed to create OpenAI client: %v", err)
	}

	// gpt-image-1 is the current OpenAI image model.
	imgModel := client.ImageModel("gpt-image-1")

	http.HandleFunc("/image", func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()
		prompt := r.URL.Query().Get("prompt")
		if prompt == "" {
			prompt = "A Go gopher exploring the stars"
		}

		res, err := ai.GenerateImage(ctx, ai.ImageRequest{
			Model:          imgModel,
			Prompt:         prompt,
			Size:           "1024x1024",
			NumberOfImages: 1,
			ResponseFormat: "url",
		})
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			_ = json.NewEncoder(w).Encode(map[string]any{"error": err.Error()})
			return
		}

		var urls []string
		for _, img := range res.Images {
			if img.URL != "" {
				urls = append(urls, img.URL)
			}
		}

		_ = json.NewEncoder(w).Encode(map[string]any{
			"prompt": prompt,
			"urls":   urls,
		})
	})

	log.Println("image generation server listening on :8082/image?prompt=...")
	log.Fatal(http.ListenAndServe(":8082", nil))
}
