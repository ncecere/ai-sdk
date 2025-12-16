package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/gofiber/fiber/v2"
	ai "github.com/ncecere/ai-sdk"
	"github.com/ncecere/ai-sdk/openai"
	"github.com/ncecere/ai-sdk/provider"
)

// This example demonstrates integrating ai-sdk with the Fiber web
// framework, streaming chat completions to the client using an
// SSE-style response.
//
// It expects:
//
//	OPENAI_API_KEY  - your OpenAI (or compatible) API key
//	OPENAI_BASE_URL - optional, for OpenAI-compatible endpoints
func main() {
	if os.Getenv("OPENAI_API_KEY") == "" {
		log.Fatal("OPENAI_API_KEY must be set")
	}

	client, err := openai.NewClient(provider.ClientOptions{})
	if err != nil {
		log.Fatalf("failed to create OpenAI client: %v", err)
	}

	model := client.ChatModel("gpt-4o-mini")

	app := fiber.New()

	app.Get("/stream", func(c *fiber.Ctx) error {
		prompt := c.Query("prompt", "Stream a response from Fiber.")

		ctx, cancel := context.WithTimeout(c.Context(), 2*time.Minute)
		defer cancel()

		stream, err := ai.StreamText(ctx, ai.GenerateTextRequest{
			Model: model,
			Messages: []ai.Message{{
				Role:    ai.RoleUser,
				Content: prompt,
			}},
		})
		if err != nil {
			return fiber.NewError(fiber.StatusInternalServerError, err.Error())
		}
		defer stream.Close()

		c.Set("Content-Type", "text/event-stream")
		c.Set("Cache-Control", "no-cache")
		c.Set("Connection", "keep-alive")

		writer := c.Context().Response.BodyWriter()

		for {
			if err := ctx.Err(); err != nil {
				return err
			}
			delta, err := stream.Next(ctx)
			if err != nil {
				return err
			}
			if delta.Done {
				break
			}
			if delta.Text == "" {
				continue
			}
			if _, err := fmt.Fprintf(writer, "data: %s\n\n", delta.Text); err != nil {
				return err
			}
		}
		if _, err := fmt.Fprint(writer, "data: [DONE]\n\n"); err != nil {
			return err
		}
		return nil
	})

	log.Println("Fiber SSE-style chat streaming on :8081/stream?prompt=...")
	if err := app.Listen(":8081"); err != nil {
		log.Fatalf("fiber listen error: %v", err)
	}
}
