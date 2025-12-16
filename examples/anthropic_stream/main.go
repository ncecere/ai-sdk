package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	ai "github.com/ncecere/ai-sdk"
	"github.com/ncecere/ai-sdk/anthropic"
	"github.com/ncecere/ai-sdk/provider"
)

func main() {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		log.Fatal("ANTHROPIC_API_KEY must be set")
	}

	client, err := anthropic.NewClient(provider.ClientOptions{})
	if err != nil {
		log.Fatalf("failed to create Anthropic client: %v", err)
	}

	model := client.ChatModel("claude-3-5-sonnet-20240620")

	baseCtx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	ctx, cancel := context.WithTimeout(baseCtx, 60*time.Second)
	defer cancel()

	stream, err := ai.StreamText(ctx, ai.GenerateTextRequest{
		Model: model,
		Messages: []ai.Message{
			{Role: ai.RoleUser, Content: "Stream a short message from Anthropic."},
		},
	})
	if err != nil {
		log.Fatalf("stream error: %v", err)
	}
	defer stream.Close()

	writer := bufio.NewWriter(os.Stdout)
	defer writer.Flush()

	for {
		delta, err := stream.Next(ctx)
		if err != nil {
			log.Fatalf("stream next error: %v", err)
		}
		if delta.Done {
			break
		}
		fmt.Fprint(writer, delta.Text)
		writer.Flush()
	}

	fmt.Fprintln(writer)
}
