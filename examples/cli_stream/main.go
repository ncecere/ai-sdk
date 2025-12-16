package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

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

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	stream, err := ai.StreamText(ctx, ai.GenerateTextRequest{
		Model: model,
		Messages: []ai.Message{{
			Role:    "user",
			Content: "Stream a short message.",
		}},
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
