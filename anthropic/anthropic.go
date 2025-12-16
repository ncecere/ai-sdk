package anthropic

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

	"github.com/ncecere/ai-sdk/provider"
	"github.com/ncecere/ai-sdk/providerutil"
)

// Client is an Anthropic provider client implementing chat models via
// the Messages API.
type Client struct {
	baseURL    string
	apiKey     string
	httpClient provider.HTTPClient
	headers    http.Header
}

// NewClient creates a new Anthropic client.
//
// Environment variables:
//   - ANTHROPIC_API_KEY (required if opts.APIKey is empty)
//   - ANTHROPIC_BASE_URL (optional, defaults to https://api.anthropic.com)
//   - ANTHROPIC_VERSION (optional, defaults to 2023-06-01)
func NewClient(opts provider.ClientOptions) (*Client, error) {
	apiKey := opts.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("ANTHROPIC_API_KEY")
	}
	if apiKey == "" {
		return nil, fmt.Errorf("anthropic: missing API key; set ClientOptions.APIKey or ANTHROPIC_API_KEY")
	}

	baseURL := opts.BaseURL
	if baseURL == "" {
		baseURL = os.Getenv("ANTHROPIC_BASE_URL")
		if baseURL == "" {
			baseURL = "https://api.anthropic.com"
		}
	}
	baseURL = strings.TrimRight(baseURL, "/")

	hc := opts.HTTPClient
	if hc == nil {
		hc = providerutil.DefaultHTTPClient()
	}

	headers := opts.Headers
	if headers == nil {
		headers = make(http.Header)
	}

	// Ensure required Anthropic headers are present.
	if headers.Get("anthropic-version") == "" {
		version := os.Getenv("ANTHROPIC_VERSION")
		if version == "" {
			version = "2023-06-01"
		}
		headers.Set("anthropic-version", version)
	}

	return &Client{
		baseURL:    baseURL,
		apiKey:     apiKey,
		httpClient: hc,
		headers:    headers,
	}, nil
}

func (c *Client) messagesURL() string {
	if strings.HasSuffix(c.baseURL, "/v1") {
		return c.baseURL + "/messages"
	}
	return c.baseURL + "/v1/messages"
}

// ChatModel returns a LanguageModel for the given Anthropic model ID.
func (c *Client) ChatModel(model string) provider.LanguageModel {
	return &messagesModel{client: c, model: model}
}

type messagesModel struct {
	client *Client
	model  string
}

const jsonToolName = "json"

type anthropicMessage struct {
	Role    string                  `json:"role"`
	Content []anthropicContentBlock `json:"content"`
}

type anthropicContentBlock struct {
	Type  string          `json:"type"`
	Text  string          `json:"text,omitempty"`
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
}

type anthropicTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema,omitempty"`
}

type anthropicMessagesRequest struct {
	Model         string             `json:"model"`
	System        string             `json:"system,omitempty"`
	Messages      []anthropicMessage `json:"messages"`
	MaxTokens     int                `json:"max_tokens"`
	Temperature   *float64           `json:"temperature,omitempty"`
	TopP          *float64           `json:"top_p,omitempty"`
	StopSequences []string           `json:"stop_sequences,omitempty"`
	Tools         []anthropicTool    `json:"tools,omitempty"`
	ToolChoice    any                `json:"tool_choice,omitempty"`
	Stream        bool               `json:"stream,omitempty"`
}

type anthropicMessagesResponse struct {
	Content    []anthropicContentBlock `json:"content"`
	StopReason string                  `json:"stop_reason"`
}

func (m *messagesModel) Generate(ctx context.Context, req *provider.LanguageModelRequest) (*provider.LanguageModelResponse, error) {
	var systemParts []string
	var messages []anthropicMessage
	for _, msg := range req.Messages {
		switch msg.Role {
		case "system":
			systemParts = append(systemParts, msg.Content)
		case "tool":
			// Anthropic does not support a dedicated tool role; map tool
			// messages to user messages containing the tool result JSON.
			messages = append(messages, anthropicMessage{
				Role: "user",
				Content: []anthropicContentBlock{{
					Type: "text",
					Text: msg.Content,
				}},
			})
		default:
			messages = append(messages, anthropicMessage{
				Role: msg.Role,
				Content: []anthropicContentBlock{{
					Type: "text",
					Text: msg.Content,
				}},
			})
		}
	}

	maxTokens := 1024
	if req.MaxTokens != nil && *req.MaxTokens > 0 {
		maxTokens = *req.MaxTokens
	}

	body := anthropicMessagesRequest{
		Model:     m.model,
		Messages:  messages,
		MaxTokens: maxTokens,
	}
	if len(systemParts) > 0 {
		body.System = strings.Join(systemParts, "\n")
	}
	body.Temperature = req.Temperature
	body.TopP = req.TopP
	if len(req.Stop) > 0 {
		body.StopSequences = req.Stop
	}

	useJSONTool := false
	if len(req.Tools) > 0 {
		tools := make([]anthropicTool, 0, len(req.Tools))
		for _, t := range req.Tools {
			tools = append(tools, anthropicTool{
				Name:        t.Name,
				Description: t.Description,
				InputSchema: json.RawMessage(t.Parameters),
			})
		}
		body.Tools = tools
	} else if len(req.JSONSchema) > 0 {
		useJSONTool = true
		body.Tools = []anthropicTool{{
			Name:        jsonToolName,
			Description: "Respond with a JSON object that matches the given schema.",
			InputSchema: json.RawMessage(req.JSONSchema),
		}}
		body.ToolChoice = map[string]string{
			"type": "tool",
			"name": jsonToolName,
		}
	}

	buf, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, m.client.messagesURL(), bytes.NewReader(buf))
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
	httpReq.Header.Set("x-api-key", m.client.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := m.client.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}

	var out anthropicMessagesResponse
	if err := providerutil.ReadJSON(resp, &out); err != nil {
		return nil, err
	}

	lmRes := &provider.LanguageModelResponse{}
	for _, c := range out.Content {
		switch c.Type {
		case "text":
			lmRes.Text += c.Text
		case "tool_use":
			lmRes.ToolCalls = append(lmRes.ToolCalls, provider.ToolCall{
				ID:           c.ID,
				Name:         c.Name,
				RawArguments: c.Input,
			})
			if useJSONTool && c.Name == jsonToolName && len(c.Input) > 0 && lmRes.Text == "" {
				lmRes.Text = normalizeJSON(c.Input)
			}
		}
	}
	lmRes.StopReason = out.StopReason
	return lmRes, nil
}

func (m *messagesModel) Stream(ctx context.Context, req *provider.LanguageModelRequest) (provider.LanguageModelStream, error) {
	var systemParts []string
	var messages []anthropicMessage
	for _, msg := range req.Messages {
		switch msg.Role {
		case "system":
			systemParts = append(systemParts, msg.Content)
		case "tool":
			messages = append(messages, anthropicMessage{
				Role: "user",
				Content: []anthropicContentBlock{{
					Type: "text",
					Text: msg.Content,
				}},
			})
		default:
			messages = append(messages, anthropicMessage{
				Role: msg.Role,
				Content: []anthropicContentBlock{{
					Type: "text",
					Text: msg.Content,
				}},
			})
		}
	}

	maxTokens := 1024
	if req.MaxTokens != nil && *req.MaxTokens > 0 {
		maxTokens = *req.MaxTokens
	}

	body := anthropicMessagesRequest{
		Model:     m.model,
		Messages:  messages,
		MaxTokens: maxTokens,
		Stream:    true,
	}
	if len(systemParts) > 0 {
		body.System = strings.Join(systemParts, "\n")
	}
	body.Temperature = req.Temperature
	body.TopP = req.TopP
	if len(req.Stop) > 0 {
		body.StopSequences = req.Stop
	}

	if len(req.Tools) > 0 {
		tools := make([]anthropicTool, 0, len(req.Tools))
		for _, t := range req.Tools {
			tools = append(tools, anthropicTool{
				Name:        t.Name,
				Description: t.Description,
				InputSchema: json.RawMessage(t.Parameters),
			})
		}
		body.Tools = tools
	} else if len(req.JSONSchema) > 0 {
		body.Tools = []anthropicTool{{
			Name:        jsonToolName,
			Description: "Respond with a JSON object that matches the given schema.",
			InputSchema: json.RawMessage(req.JSONSchema),
		}}
		body.ToolChoice = map[string]string{
			"type": "tool",
			"name": jsonToolName,
		}
	}

	buf, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, m.client.messagesURL(), bytes.NewReader(buf))
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
	httpReq.Header.Set("x-api-key", m.client.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")

	resp, err := m.client.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}

	return newMessagesStream(resp.Body), nil
}

// messagesStream implements provider.LanguageModelStream for Anthropic messages.

type messagesStream struct {
	body    io.ReadCloser
	scanner *bufio.Scanner
	done    bool
}

func newMessagesStream(body io.ReadCloser) provider.LanguageModelStream {
	scanner := bufio.NewScanner(body)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)
	return &messagesStream{
		body:    body,
		scanner: scanner,
	}
}

type anthropicStreamEvent struct {
	Type  string          `json:"type"`
	Delta *anthropicDelta `json:"delta,omitempty"`
}

type anthropicDelta struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

func (s *messagesStream) Next(ctx context.Context) (*provider.LanguageModelDelta, error) {
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

		var ev anthropicStreamEvent
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			return nil, err
		}

		switch ev.Type {
		case "content_block_delta":
			if ev.Delta != nil && ev.Delta.Type == "text_delta" && ev.Delta.Text != "" {
				return &provider.LanguageModelDelta{Text: ev.Delta.Text}, nil
			}
		case "message_stop":
			s.done = true
			return &provider.LanguageModelDelta{Done: true}, nil
		}
	}
}

func (s *messagesStream) Close() error {
	s.done = true
	return s.body.Close()
}

func normalizeJSON(raw json.RawMessage) string {
	b := bytes.TrimSpace(raw)
	if len(b) == 0 {
		return ""
	}
	var v any
	if err := json.Unmarshal(b, &v); err != nil {
		return string(b)
	}
	normalized, err := json.Marshal(v)
	if err != nil {
		return string(b)
	}
	return string(normalized)
}
