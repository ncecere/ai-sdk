package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	ai "github.com/ncecere/ai-sdk"
)

// WriteRunAsSSE executes an agent run and streams agent events as
// Server-Sent Events (SSE) to the provided ResponseWriter.
//
// Each event is encoded as a single JSON object and sent using the
// "data: <json>\n\n" framing. The function returns when the agent run
// completes or an error occurs.
func WriteRunAsSSE(ctx context.Context, w http.ResponseWriter, cfg Config, initialMessages []ai.Message) (*Result, error) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return nil, fmt.Errorf("agent: response writer does not support flushing")
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	encoder := json.NewEncoder(w)

	emit := func(e Event) {
		select {
		case <-ctx.Done():
			return
		default:
		}

		b, err := json.Marshal(e)
		if err != nil {
			return
		}
		if _, err := fmt.Fprintf(w, "data: %s\n\n", b); err != nil {
			return
		}
		flusher.Flush()
	}

	res, err := RunWithEvents(ctx, cfg, initialMessages, emit)
	if err != nil {
		return nil, err
	}

	// Send a final done event to ensure clients see completion even if
	// the agent terminated without emitting an explicit done event.
	_ = encoder.Encode(Event{Type: EventTypeDone})
	flusher.Flush()

	return res, nil
}
