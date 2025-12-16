package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"strings"

	"github.com/ncecere/ai-sdk/provider"
	"github.com/ncecere/ai-sdk/providerutil"
)

// completionModel implements provider.CompletionModel for the OpenAI
// /v1/completions endpoint.
type completionModel struct {
	client *Client
	model  string
}

type openAICompletionRequest struct {
	Model       string   `json:"model"`
	Prompt      string   `json:"prompt"`
	Temperature *float64 `json:"temperature,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	MaxTokens   *int     `json:"max_tokens,omitempty"`
	Stop        []string `json:"stop,omitempty"`
	User        string   `json:"user,omitempty"`
}

type openAICompletionResponse struct {
	Choices []struct {
		Text         string `json:"text"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
}

func (c *Client) completionsURL() string {
	if strings.HasSuffix(c.baseURL, "/v1") {
		return c.baseURL + "/completions"
	}
	return c.baseURL + "/v1/completions"
}

// CompletionModel returns a CompletionModel for the given completion
// model ID.
func (c *Client) CompletionModel(model string) provider.CompletionModel {
	return &completionModel{client: c, model: model}
}

func (m *completionModel) Generate(ctx context.Context, req *provider.CompletionRequest) (*provider.CompletionResponse, error) {
	body := openAICompletionRequest{
		Model:       m.model,
		Prompt:      req.Prompt,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		MaxTokens:   req.MaxTokens,
		Stop:        req.Stop,
		User:        req.UserID,
	}

	buf, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, m.client.completionsURL(), bytes.NewReader(buf))
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

	var out openAICompletionResponse
	if err := providerutil.ReadJSON(resp, &out); err != nil {
		return nil, err
	}
	if len(out.Choices) == 0 {
		return &provider.CompletionResponse{}, nil
	}

	choice := out.Choices[0]
	return &provider.CompletionResponse{
		Text:       choice.Text,
		StopReason: choice.FinishReason,
	}, nil
}
