package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

	ai "github.com/ncecere/ai-sdk"
	"github.com/ncecere/ai-sdk/anthropic"
	"github.com/ncecere/ai-sdk/provider"
)

type AddArgs struct {
	A float64 `json:"a"`
	B float64 `json:"b"`
}

func main() {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		log.Fatal("ANTHROPIC_API_KEY must be set")
	}

	client, err := anthropic.NewClient(provider.ClientOptions{})
	if err != nil {
		log.Fatalf("failed to create Anthropic client: %v", err)
	}

	model := client.ChatModel("claude-3-5-sonnet-20240620")

	schema, err := ai.JSONSchemaFromType(AddArgs{})
	if err != nil {
		log.Fatalf("failed to build JSON schema: %v", err)
	}

	tool := ai.ToolDefinition{
		Name:        "add",
		Description: "Add two numbers and return the sum.",
		Parameters:  schema,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// First call: let the model decide whether to call the tool.
	res, err := ai.GenerateText(ctx, ai.GenerateTextRequest{
		Model: model,
		Messages: []ai.Message{
			{Role: ai.RoleUser, Content: "Use the add tool to add 3 and 5."},
		},
		Tools: []ai.ToolDefinition{tool},
	})
	if err != nil {
		log.Fatalf("GenerateText failed: %v", err)
	}

	fmt.Printf("Assistant: %s\n", res.Text)

	if len(res.ToolCalls) == 0 {
		fmt.Println("No tool calls returned.")
		return
	}

	// Execute the first tool call locally.
	tc := res.ToolCalls[0]
	var args AddArgs
	if err := ai.DecodeToolCallArgs(tc, &args); err != nil {
		log.Fatalf("failed to decode tool args: %v", err)
	}

	sum := args.A + args.B
	toolResult := map[string]any{
		"tool":   tc.Name,
		"result": sum,
	}
	payload, err := json.Marshal(toolResult)
	if err != nil {
		log.Fatalf("failed to marshal tool result: %v", err)
	}

	// Second call: send the tool result back to the model.
	res2, err := ai.GenerateText(ctx, ai.GenerateTextRequest{
		Model: model,
		Messages: []ai.Message{
			{Role: ai.RoleUser, Content: "Use the add tool to add 3 and 5."},
			{Role: ai.RoleAssistant, Content: res.Text},
			{Role: ai.RoleTool, Content: string(payload)},
		},
	})
	if err != nil {
		log.Fatalf("GenerateText (with tool result) failed: %v", err)
	}

	fmt.Printf("Assistant (after tool): %s\n", res2.Text)
}
