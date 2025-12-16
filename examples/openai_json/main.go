package main

import (
	"context"
	"encoding/json"
	"log"
	"os"
	"time"

	ai "github.com/ncecere/ai-sdk"
	"github.com/ncecere/ai-sdk/openai"
	"github.com/ncecere/ai-sdk/provider"
)

type WeatherReport struct {
	City               string  `json:"city"`
	TemperatureCelsius float64 `json:"temperatureCelsius"`
	Description        string  `json:"description"`
}

func main() {
	if os.Getenv("OPENAI_API_KEY") == "" {
		log.Fatal("OPENAI_API_KEY must be set")
	}

	client, err := openai.NewClient(provider.ClientOptions{})
	if err != nil {
		log.Fatalf("failed to create OpenAI client: %v", err)
	}

	model := client.ChatModel("gpt-4o-mini")

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
