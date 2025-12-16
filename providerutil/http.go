package providerutil

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// ReadJSON decodes a JSON response body into v and closes the body.
//
// If the response status code is not in the 2xx range, ReadJSON
// returns an error of the form:
//
//	provider: http status <code>: <truncated-body>
//
// Callers can inspect this error string or wrap it in higher-level
// errors as needed.
func ReadJSON(resp *http.Response, v any) error {
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 8*1024))
		return fmt.Errorf("provider: http status %d: %s", resp.StatusCode, string(b))
	}
	dec := json.NewDecoder(resp.Body)
	return dec.Decode(v)
}

// DefaultHTTPClient returns the default HTTP client used when none is provided.
func DefaultHTTPClient() *http.Client {
	return http.DefaultClient
}
