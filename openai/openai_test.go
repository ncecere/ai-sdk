package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ncecere/ai-sdk/provider"
)

func float64Ptr(v float64) *float64 { return &v }

func TestChatModelGenerate_MapsRequestAndResponse(t *testing.T) {
	ctx := context.Background()

	var recordedReq openAIChatRequest

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("unexpected method: %s", r.Method)
		}
		if got := r.URL.Path; got != "/v1/chat/completions" {
			t.Fatalf("unexpected path: %s", got)
		}
		if auth := r.Header.Get("Authorization"); !strings.HasPrefix(auth, "Bearer ") {
			t.Fatalf("missing bearer auth header: %q", auth)
		}

		if err := json.NewDecoder(r.Body).Decode(&recordedReq); err != nil {
			t.Fatalf("failed to decode request: %v", err)
		}

		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{
			"choices": [
				{
					"finish_reason": "stop",
					"message": {
						"role": "assistant",
						"content": "hello from test",
						"tool_calls": [
							{
								"id": "call-1",
								"type": "function",
								"function": {
									"name": "testTool",
									"arguments": "{\"foo\":\"bar\"}"
								}
							}
						]
					}
				}
			]
		}`)
	}))
	defer ts.Close()

	client, err := NewClient(provider.ClientOptions{
		BaseURL:    ts.URL + "/v1",
		APIKey:     "test-key",
		HTTPClient: ts.Client(),
	})
	if err != nil {
		t.Fatalf("NewClient error: %v", err)
	}

	model := client.ChatModel("test-model")

	temp := float64Ptr(0.5)
	res, err := model.Generate(ctx, &provider.LanguageModelRequest{
		Messages:    []provider.Message{{Role: "user", Content: "hi"}},
		Temperature: temp,
		JSONSchema:  []byte(`{"type":"object"}`),
		Tools: []provider.ToolDefinition{{
			Name:        "testTool",
			Description: "test tool",
			Parameters:  []byte(`{"type":"object"}`),
		}},
	})
	if err != nil {
		t.Fatalf("Generate error: %v", err)
	}

	// Check request mapping
	if recordedReq.Model != "test-model" {
		t.Fatalf("expected model 'test-model', got %q", recordedReq.Model)
	}
	if len(recordedReq.Messages) != 1 || recordedReq.Messages[0].Role != "user" || recordedReq.Messages[0].Content != "hi" {
		t.Fatalf("unexpected messages: %+v", recordedReq.Messages)
	}
	if recordedReq.Temperature == nil || *recordedReq.Temperature != *temp {
		t.Fatalf("temperature not propagated: %+v", recordedReq.Temperature)
	}
	if recordedReq.ResponseFormat == nil || recordedReq.ResponseFormat.Type != "json_schema" {
		t.Fatalf("expected json_schema response format, got %+v", recordedReq.ResponseFormat)
	}
	if len(recordedReq.Tools) != 1 || recordedReq.Tools[0].Function.Name != "testTool" {
		t.Fatalf("tools not propagated: %+v", recordedReq.Tools)
	}

	// Check response mapping
	if res.Text != "hello from test" {
		t.Fatalf("unexpected text: %q", res.Text)
	}
	if res.StopReason != "stop" {
		t.Fatalf("unexpected stop reason: %q", res.StopReason)
	}
	if len(res.ToolCalls) != 1 || res.ToolCalls[0].Name != "testTool" {
		t.Fatalf("unexpected tool calls: %+v", res.ToolCalls)
	}
}

func TestChatModelStream_ParsesSSEChunks(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		// First chunk
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n")
		// Second chunk with finish_reason
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n")
		// Done marker
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer ts.Close()

	client, err := NewClient(provider.ClientOptions{
		BaseURL:    ts.URL + "/v1",
		APIKey:     "test-key",
		HTTPClient: ts.Client(),
	})
	if err != nil {
		t.Fatalf("NewClient error: %v", err)
	}

	model := client.ChatModel("stream-model")
	stream, err := model.Stream(ctx, &provider.LanguageModelRequest{
		Messages: []provider.Message{{Role: "user", Content: "hi"}},
	})
	if err != nil {
		t.Fatalf("Stream error: %v", err)
	}
	defer stream.Close()

	var text strings.Builder
	for {
		delta, err := stream.Next(ctx)
		if err != nil {
			t.Fatalf("Next error: %v", err)
		}
		text.WriteString(delta.Text)
		if delta.Done {
			break
		}
	}
	if got := text.String(); got != "Hello" {
		t.Fatalf("unexpected concatenated text: %q", got)
	}
}

func TestEmbeddingModelGenerate_MapsRequestAndResponse(t *testing.T) {
	ctx := context.Background()

	var recordedReq openAIEmbeddingRequest

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&recordedReq); err != nil {
			t.Fatalf("failed to decode embedding request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"data":[{"embedding":[1,2,3]}]}`)
	}))
	defer ts.Close()

	client, err := NewClient(provider.ClientOptions{
		BaseURL:    ts.URL + "/v1",
		APIKey:     "test-key",
		HTTPClient: ts.Client(),
	})
	if err != nil {
		t.Fatalf("NewClient error: %v", err)
	}

	model := client.EmbeddingModel("embed-model")
	res, err := model.Generate(ctx, &provider.EmbeddingRequest{
		Input:  []string{"hello"},
		UserID: "user-1",
	})
	if err != nil {
		t.Fatalf("Generate embedding error: %v", err)
	}

	if recordedReq.Model != "embed-model" {
		t.Fatalf("expected model 'embed-model', got %q", recordedReq.Model)
	}
	if len(recordedReq.Input) != 1 || recordedReq.Input[0] != "hello" {
		t.Fatalf("unexpected input: %+v", recordedReq.Input)
	}
	if recordedReq.User != "user-1" {
		t.Fatalf("unexpected user: %q", recordedReq.User)
	}
	if len(res.Embeddings) != 1 || len(res.Embeddings[0]) != 3 {
		t.Fatalf("unexpected embeddings: %+v", res.Embeddings)
	}
}

func TestChatModelGenerate_PropagatesHTTPError(t *testing.T) {
	ctx := context.Background()

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprint(w, "internal error")
	}))
	defer ts.Close()

	client, err := NewClient(provider.ClientOptions{
		BaseURL:    ts.URL + "/v1",
		APIKey:     "test-key",
		HTTPClient: ts.Client(),
	})
	if err != nil {
		t.Fatalf("NewClient error: %v", err)
	}

	model := client.ChatModel("test-model")
	_, err = model.Generate(ctx, &provider.LanguageModelRequest{
		Messages: []provider.Message{{Role: "user", Content: "hi"}},
	})
	if err == nil {
		t.Fatalf("expected error from HTTP 500, got nil")
	}
	if !strings.Contains(err.Error(), "http status 500") {
		t.Fatalf("expected http status 500 in error, got %v", err)
	}
}
