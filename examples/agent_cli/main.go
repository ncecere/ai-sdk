package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	ai "github.com/ncecere/ai-sdk"
	"github.com/ncecere/ai-sdk/agent"
	"github.com/ncecere/ai-sdk/openai"
	"github.com/ncecere/ai-sdk/provider"
	"github.com/ncecere/ai-sdk/registry"
)

// agent_cli is a small CLI example that demonstrates the agent package
// using a simple math tool.
func main() {
	if os.Getenv("OPENAI_API_KEY") == "" {
		log.Fatal("OPENAI_API_KEY must be set")
	}

	prompt := flag.String("prompt", "What is 2 + 3? Use the add tool.", "agent prompt")
	flag.Parse()

	client, err := openai.NewClient(provider.ClientOptions{})
	if err != nil {
		log.Fatalf("failed to create OpenAI client: %v", err)
	}

	reg := registry.NewInMemoryRegistry()
	reg.RegisterLanguageModel("chat:default", client.ChatModel("gpt-4o-mini"))

	tools := map[string]agent.Tool{
		"add": {
			Name:        "add",
			Description: "Add two integers a and b",
			Execute: func(ctx context.Context, args json.RawMessage) (any, error) {
				var payload struct {
					A int `json:"a"`
					B int `json:"b"`
				}
				if err := json.Unmarshal(args, &payload); err != nil {
					return nil, err
				}
				return map[string]any{"sum": payload.A + payload.B}, nil
			},
		},
	}

	cfg := agent.Config{
		Registry:  reg,
		ModelName: "chat:default",
		Tools:     tools,
		MaxSteps:  8,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	initial := []ai.Message{{
		Role:    ai.RoleUser,
		Content: *prompt,
	}}

	fmt.Println("Running agent...")

	_, err = agent.RunWithEvents(ctx, cfg, initial, func(e agent.Event) {
		switch e.Type {
		case agent.EventTypeMessage:
			fmt.Printf("[assistant] %s\n", e.Content)
		case agent.EventTypeToolStart:
			fmt.Printf("[tool start] %s (step %d)\n", e.Tool, e.Step)
		case agent.EventTypeToolResult:
			fmt.Printf("[tool result] %s (step %d)\n", e.Tool, e.Step)
		case agent.EventTypeError:
			fmt.Printf("[error] %s\n", e.Content)
		case agent.EventTypeDone:
			fmt.Println("[done]")
		}
	})
	if err != nil {
		log.Fatalf("agent error: %v", err)
	}
}
