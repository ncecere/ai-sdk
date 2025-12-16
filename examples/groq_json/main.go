package main

import (
	"context"
	"encoding/json"
	"log"
	"os"
	"time"

	ai "github.com/ncecere/ai-sdk"
	"github.com/ncecere/ai-sdk/groq"
	"github.com/ncecere/ai-sdk/provider"
)

type WeatherReport struct {
	City               string  `json:"city"`
	TemperatureCelsius float64 `json:"temperatureCelsius"`
	Description        string  `json:"description"`
}

func main() {
	if os.Getenv("GROQ_API_KEY") == "" {
		log.Fatal("GROQ_API_KEY must be set")
	}

	client, err := groq.NewClient(provider.ClientOptions{})
	if err != nil {
		log.Fatalf("failed to create Groq client: %v", err)
	}

	modelID := os.Getenv("GROQ_MODEL_ID")
	if modelID == "" {
		modelID = "llama-3.1-8b-instant"
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
