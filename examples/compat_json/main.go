package main

import (
	"context"
	"encoding/json"
	"log"
	"os"
	"time"

	ai "github.com/ncecere/ai-sdk"
	"github.com/ncecere/ai-sdk/openai"
)

type WeatherReport struct {
	City               string  `json:"city"`
	TemperatureCelsius float64 `json:"temperatureCelsius"`
	Description        string  `json:"description"`
}

func main() {
	// CompatibleClient expects OPENAI_COMPATIBLE_BASE_URL to be set and uses
	// OPENAI_COMPATIBLE_API_KEY or OPENAI_API_KEY for authentication.
	client, err := openai.CompatibleClient()
	if err != nil {
		log.Fatalf("failed to create OpenAI-compatible client: %v", err)
	}

	modelID := os.Getenv("COMPAT_MODEL_ID")
	if modelID == "" {
		modelID = "gpt-oss-20b"
	}
	model := client.ChatModel(modelID)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	report, err := ai.GenerateObject[WeatherReport](ctx, model, []ai.Message{
		{
			Role:    ai.RoleUser,
			Content: "Provide a short fictional weather report.",
		},
	})
	if err != nil {
		log.Fatalf("GenerateObject failed: %v", err)
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(report); err != nil {
		log.Fatalf("failed to encode report: %v", err)
	}
}
