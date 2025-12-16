package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	ai "github.com/ncecere/ai-sdk"
	"github.com/ncecere/ai-sdk/openai"
	"github.com/ncecere/ai-sdk/provider"
)

// cli_transcribe is a small CLI example that demonstrates
// using ai-sdk with the OpenAI transcription API.
//
// It expects:
//
//	OPENAI_API_KEY  - your OpenAI (or compatible) API key
//	OPENAI_BASE_URL - optional, for OpenAI-compatible endpoints
//
// Usage:
//
//	go run ./examples/cli_transcribe -file path/to/audio.wav \
//	  -model gpt-4o-transcribe -lang en
func main() {
	if os.Getenv("OPENAI_API_KEY") == "" {
		log.Fatal("OPENAI_API_KEY must be set")
	}

	filePath := flag.String("file", "", "path to audio file to transcribe")
	modelID := flag.String("model", "gpt-4o-transcribe", "transcription model ID")
	lang := flag.String("lang", "", "optional language hint (e.g. en)")
	flag.Parse()

	if *filePath == "" {
		log.Fatal("-file must be provided")
	}

	data, err := os.ReadFile(*filePath)
	if err != nil {
		log.Fatalf("failed to read audio file: %v", err)
	}

	client, err := openai.NewClient(provider.ClientOptions{})
	if err != nil {
		log.Fatalf("failed to create OpenAI client: %v", err)
	}

	transModel := client.TranscriptionModel(*modelID)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	res, err := ai.Transcribe(ctx, ai.TranscriptionRequest{
		Model:    transModel,
		Audio:    data,
		FileName: filepath.Base(*filePath),
		Language: *lang,
	})
	if err != nil {
		log.Fatalf("transcription error: %v", err)
	}

	fmt.Println(res.Text)
}
