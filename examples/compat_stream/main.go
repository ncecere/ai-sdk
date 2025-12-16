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
	"github.com/ncecere/ai-sdk/openai"
)

func main() {
	client, err := openai.CompatibleClient()
	if err != nil {
		log.Fatalf("failed to create OpenAI-compatible client: %v", err)
	}

	modelID := os.Getenv("COMPAT_MODEL_ID")
	if modelID == "" {
		modelID = "gpt-oss-20b"
	}
	model := client.ChatModel(modelID)

	baseCtx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	ctx, cancel := context.WithTimeout(baseCtx, 60*time.Second)
	defer cancel()

	stream, err := ai.StreamText(ctx, ai.GenerateTextRequest{
		Model: model,
		Messages: []ai.Message{
			{Role: ai.RoleUser, Content: "Stream a short message from the compatible backend."},
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
