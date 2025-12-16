package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/ncecere/ai-sdk/provider"
	"github.com/ncecere/ai-sdk/providerutil"
)

// Client is an OpenAI provider client implementing chat and embeddings.
//
// It can be configured explicitly via ClientOptions or implicitly via
// environment variables. See NewClient and CompatibleClient for
// configuration details.
type Client struct {
	baseURL    string
	apiKey     string
	httpClient provider.HTTPClient
	headers    http.Header
}

func (c *Client) chatCompletionsURL() string {
	if strings.HasSuffix(c.baseURL, "/v1") {
		return c.baseURL + "/chat/completions"
	}
	return c.baseURL + "/v1/chat/completions"
}

func (c *Client) embeddingsURL() string {
	if strings.HasSuffix(c.baseURL, "/v1") {
		return c.baseURL + "/embeddings"
	}
	return c.baseURL + "/v1/embeddings"
}

// NewClient creates a new OpenAI client.
// It follows Vercel's pattern of reading configuration from environment
// variables by default.
//
// Environment variables:
//   - OPENAI_API_KEY (required if opts.APIKey is empty)
//   - OPENAI_BASE_URL (optional, defaults to https://api.openai.com)
func NewClient(opts provider.ClientOptions) (*Client, error) {
	apiKey := opts.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}
	if apiKey == "" {
		return nil, fmt.Errorf("openai: missing API key; set ClientOptions.APIKey or OPENAI_API_KEY")
	}

	baseURL := opts.BaseURL
	if baseURL == "" {
		baseURL = os.Getenv("OPENAI_BASE_URL")
		if baseURL == "" {
			baseURL = "https://api.openai.com"
		}
	}
	baseURL = strings.TrimRight(baseURL, "/")

	hc := opts.HTTPClient
	if hc == nil {
		hc = providerutil.DefaultHTTPClient()
	}

	return &Client{
		baseURL:    baseURL,
		apiKey:     apiKey,
		httpClient: hc,
		headers:    opts.Headers,
	}, nil
}

// ChatModel returns a LanguageModel for the given chat model ID.
func (c *Client) ChatModel(model string) provider.LanguageModel {
	return &chatModel{client: c, model: model}
}

// EmbeddingModel returns an EmbeddingModel for the given embedding model ID.
func (c *Client) EmbeddingModel(model string) provider.EmbeddingModel {
	return &embeddingModel{client: c, model: model}
}

type chatModel struct {
	client *Client
	model  string
}

type openAIChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAIChatTool struct {
	Type     string             `json:"type"`
	Function openAIFunctionTool `json:"function"`
}

type openAIFunctionTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type openAIChatRequest struct {
	Model          string                `json:"model"`
	Messages       []openAIChatMessage   `json:"messages"`
	Temperature    *float64              `json:"temperature,omitempty"`
	TopP           *float64              `json:"top_p,omitempty"`
	MaxTokens      *int                  `json:"max_tokens,omitempty"`
	Stop           []string              `json:"stop,omitempty"`
	ResponseFormat *openAIResponseFormat `json:"response_format,omitempty"`
	Tools          []openAIChatTool      `json:"tools,omitempty"`
	ToolChoice     any                   `json:"tool_choice,omitempty"`
	Stream         bool                  `json:"stream,omitempty"`
}

type openAIResponseFormat struct {
	Type       string            `json:"type"`
	JSONSchema *openAIJSONSchema `json:"json_schema,omitempty"`
}

type openAIJSONSchema struct {
	Name   string          `json:"name"`
	Schema json.RawMessage `json:"schema"`
}

type openAIChatResponse struct {
	Choices []struct {
		FinishReason string `json:"finish_reason"`
		Message      struct {
			Role      string `json:"role"`
			Content   string `json:"content"`
			ToolCalls []struct {
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string          `json:"name"`
					Arguments json.RawMessage `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"message"`
	} `json:"choices"`
}

type openAIChatStreamChunk struct {
	Choices []struct {
		Delta struct {
			Content   string `json:"content"`
			ToolCalls []struct {
				ID       string `json:"id"`
				Type     string `json:"type"`
				Function struct {
					Name      string          `json:"name"`
					Arguments json.RawMessage `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
}

func (m *chatModel) Generate(ctx context.Context, req *provider.LanguageModelRequest) (*provider.LanguageModelResponse, error) {
	body := openAIChatRequest{
		Model: m.model,
	}
	for _, msg := range req.Messages {
		body.Messages = append(body.Messages, openAIChatMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}
	body.Temperature = req.Temperature
	body.TopP = req.TopP
	body.MaxTokens = req.MaxTokens
	body.Stop = req.Stop

	if len(req.JSONSchema) > 0 {
		body.ResponseFormat = &openAIResponseFormat{
			Type: "json_schema",
			JSONSchema: &openAIJSONSchema{
				Name:   "response",
				Schema: json.RawMessage(req.JSONSchema),
			},
		}
	}

	if len(req.Tools) > 0 {
		for _, t := range req.Tools {
			body.Tools = append(body.Tools, openAIChatTool{
				Type: "function",
				Function: openAIFunctionTool{
					Name:        t.Name,
					Description: t.Description,
					Parameters:  json.RawMessage(t.Parameters),
				},
			})
		}
	}

	buf, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, m.client.chatCompletionsURL(), bytes.NewReader(buf))
	if err != nil {
		return nil, err
	}
	// Attach any custom headers first, then enforce required headers.
	for k, vs := range m.client.headers {
		for _, v := range vs {
			if v == "" {
				continue
			}
			httpReq.Header.Add(k, v)
		}
	}
	httpReq.Header.Set("Authorization", "Bearer "+m.client.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := m.client.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}

	var out openAIChatResponse
	if err := providerutil.ReadJSON(resp, &out); err != nil {
		return nil, err
	}
	if len(out.Choices) == 0 {
		return &provider.LanguageModelResponse{}, nil
	}

	choice := out.Choices[0]
	lmResp := &provider.LanguageModelResponse{
		Text:       choice.Message.Content,
		StopReason: choice.FinishReason,
	}
	for _, tc := range choice.Message.ToolCalls {
		if tc.Type != "function" {
			continue
		}
		lmResp.ToolCalls = append(lmResp.ToolCalls, provider.ToolCall{
			ID:           tc.ID,
			Name:         tc.Function.Name,
			RawArguments: []byte(tc.Function.Arguments),
		})
	}

	return lmResp, nil
}

func (m *chatModel) Stream(ctx context.Context, req *provider.LanguageModelRequest) (provider.LanguageModelStream, error) {
	body := openAIChatRequest{
		Model:  m.model,
		Stream: true,
	}
	for _, msg := range req.Messages {
		body.Messages = append(body.Messages, openAIChatMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}
	body.Temperature = req.Temperature
	body.TopP = req.TopP
	body.MaxTokens = req.MaxTokens
	body.Stop = req.Stop

	if len(req.JSONSchema) > 0 {
		body.ResponseFormat = &openAIResponseFormat{
			Type: "json_schema",
			JSONSchema: &openAIJSONSchema{
				Name:   "response",
				Schema: json.RawMessage(req.JSONSchema),
			},
		}
	}

	if len(req.Tools) > 0 {
		for _, t := range req.Tools {
			body.Tools = append(body.Tools, openAIChatTool{
				Type: "function",
				Function: openAIFunctionTool{
					Name:        t.Name,
					Description: t.Description,
					Parameters:  json.RawMessage(t.Parameters),
				},
			})
		}
	}

	buf, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, m.client.chatCompletionsURL(), bytes.NewReader(buf))
	if err != nil {
		return nil, err
	}
	for k, vs := range m.client.headers {
		for _, v := range vs {
			if v == "" {
				continue
			}
			httpReq.Header.Add(k, v)
		}
	}
	httpReq.Header.Set("Authorization", "Bearer "+m.client.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := m.client.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}

	return newChatStream(resp.Body), nil
}

type chatStream struct {
	body    io.ReadCloser
	scanner *bufio.Scanner
	done    bool
}

func newChatStream(body io.ReadCloser) provider.LanguageModelStream {
	scanner := bufio.NewScanner(body)
	// Increase buffer for long lines
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)
	return &chatStream{
		body:    body,
		scanner: scanner,
	}
}

func (s *chatStream) Next(ctx context.Context) (*provider.LanguageModelDelta, error) {
	if s.done {
		return &provider.LanguageModelDelta{Done: true}, nil
	}

	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if !s.scanner.Scan() {
			if err := s.scanner.Err(); err != nil {
				return nil, err
			}
			s.done = true
			return &provider.LanguageModelDelta{Done: true}, nil
		}
		line := strings.TrimSpace(s.scanner.Text())
		if line == "" {
			continue
		}
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			s.done = true
			return &provider.LanguageModelDelta{Done: true}, nil
		}

		var chunk openAIChatStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return nil, err
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		choice := chunk.Choices[0]
		delta := &provider.LanguageModelDelta{
			Text: choice.Delta.Content,
		}
		for _, tc := range choice.Delta.ToolCalls {
			if tc.Type != "function" {
				continue
			}
			delta.ToolCalls = append(delta.ToolCalls, provider.ToolCall{
				ID:           tc.ID,
				Name:         tc.Function.Name,
				RawArguments: []byte(tc.Function.Arguments),
			})
		}
		if choice.FinishReason != "" {
			delta.Done = true
			s.done = true
		}
		return delta, nil
	}
}

func (s *chatStream) Close() error {
	s.done = true
	return s.body.Close()
}

type embeddingModel struct {
	client *Client
	model  string
}

type openAIEmbeddingRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
	User  string   `json:"user,omitempty"`
}

type openAIEmbeddingResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
}

func (m *embeddingModel) Generate(ctx context.Context, req *provider.EmbeddingRequest) (*provider.EmbeddingResponse, error) {
	body := openAIEmbeddingRequest{
		Model: m.model,
		Input: req.Input,
		User:  req.UserID,
	}

	buf, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, m.client.embeddingsURL(), bytes.NewReader(buf))
	if err != nil {
		return nil, err
	}
	for k, vs := range m.client.headers {
		for _, v := range vs {
			if v == "" {
				continue
			}
			httpReq.Header.Add(k, v)
		}
	}
	httpReq.Header.Set("Authorization", "Bearer "+m.client.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := m.client.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}

	var out openAIEmbeddingResponse
	if err := providerutil.ReadJSON(resp, &out); err != nil {
		return nil, err
	}

	res := &provider.EmbeddingResponse{}
	for _, d := range out.Data {
		res.Embeddings = append(res.Embeddings, d.Embedding)
	}
	return res, nil
}

// CompatibleClient returns a new Client configured for an OpenAI-compatible endpoint.
//
// It reads from environment variables by default:
//   - OPENAI_COMPATIBLE_API_KEY (fallback to OPENAI_API_KEY)
//   - OPENAI_COMPATIBLE_BASE_URL (required if not using standard OpenAI)
//
// For more control (custom headers, HTTP client, etc.), prefer calling
// NewClient directly with a populated provider.ClientOptions.
func CompatibleClient() (*Client, error) {
	apiKey := os.Getenv("OPENAI_COMPATIBLE_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
	}
	baseURL := os.Getenv("OPENAI_COMPATIBLE_BASE_URL")
	if baseURL == "" {
		return nil, fmt.Errorf("openai: missing OPENAI_COMPATIBLE_BASE_URL for compatible client")
	}
	baseURL = strings.TrimRight(baseURL, "/")

	return NewClient(provider.ClientOptions{
		BaseURL: baseURL,
		APIKey:  apiKey,
	})
}

// WithHTTPTimeout is a helper to wrap the default HTTP client with a timeout.
func WithHTTPTimeout(d time.Duration) provider.HTTPClient {
	return &http.Client{Timeout: d}
}
