package ai

import (
	"context"
	"fmt"
	"net/http"
)

// WriteTextStreamAsSSE writes a TextStream to an http.ResponseWriter
// using the Server-Sent Events (SSE) format.
//
// It sets the standard SSE headers and then sends each non-empty
// TextDelta.Text value as a separate `data:` event line.
// The stream terminates when a delta with Done=true is received or
// when the context is canceled.
func WriteTextStreamAsSSE(ctx context.Context, w http.ResponseWriter, stream TextStream) error {
	defer stream.Close()

	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")

	flusher, _ := w.(http.Flusher)

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

		if _, err := fmt.Fprintf(w, "data: %s\n\n", delta.Text); err != nil {
			return err
		}
		if flusher != nil {
			flusher.Flush()
		}
	}

	// Send a final [DONE] marker for convenience.
	if _, err := fmt.Fprint(w, "data: [DONE]\n\n"); err != nil {
		return err
	}
	if flusher != nil {
		flusher.Flush()
	}

	return nil
}
