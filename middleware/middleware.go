package middleware

import (
	"context"
	"errors"
	"log"
	"net"
	"time"

	"github.com/ncecere/ai-sdk/provider"
)

// Logger is the minimal logging interface used by the middleware package.
// It matches the Printf method on *log.Logger so callers can pass
// log.Default() or a custom logger implementation.
type Logger interface {
	Printf(format string, v ...any)
}

// LanguageModelMiddleware wraps a provider.LanguageModel with additional
// behavior such as logging, retries, or telemetry.
type LanguageModelMiddleware func(provider.LanguageModel) provider.LanguageModel

// WrapLanguageModel applies the provided middlewares around the base
// language model. Middlewares are applied in the order provided, so the
// first middleware becomes the outermost wrapper.
func WrapLanguageModel(base provider.LanguageModel, mws ...LanguageModelMiddleware) provider.LanguageModel {
	wrapped := base
	for i := len(mws) - 1; i >= 0; i-- {
		wrapped = mws[i](wrapped)
	}
	return wrapped
}

// LoggingOptions controls which aspects of a language-model call are
// logged by the logging middleware.
type LoggingOptions struct {
	// Logger is the destination for log output. If nil, log.Default() is used.
	Logger Logger
	// LogRequest controls whether request metadata (model name) is logged.
	LogRequest bool
	// LogResponse controls whether successful responses are logged.
	LogResponse bool
	// LogErrors controls whether errors are logged.
	LogErrors bool
	// LogDuration controls whether call duration is logged.
	LogDuration bool
}

// defaultLoggingOptions returns a LoggingOptions value with sensible
// defaults for typical usage.
func defaultLoggingOptions(opts LoggingOptions) LoggingOptions {
	if opts.Logger == nil {
		opts.Logger = log.Default()
	}
	// By default, log request metadata, errors, and duration.
	if !opts.LogRequest && !opts.LogResponse && !opts.LogErrors && !opts.LogDuration {
		opts.LogRequest = true
		opts.LogErrors = true
		opts.LogDuration = true
	}
	return opts
}

// LoggingLanguageModel returns a LanguageModelMiddleware that logs
// Generate and Stream calls using the provided options. Logs focus on
// high-level metadata (model name, duration, and error state) and do
// not include full request/response bodies.
func LoggingLanguageModel(opts LoggingOptions) LanguageModelMiddleware {
	opts = defaultLoggingOptions(opts)

	return func(next provider.LanguageModel) provider.LanguageModel {
		return &loggingLanguageModel{
			next:  next,
			opts:  opts,
			logFn: opts.Logger.Printf,
		}
	}
}

type loggingLanguageModel struct {
	next  provider.LanguageModel
	opts  LoggingOptions
	logFn func(format string, v ...any)
}

func (l *loggingLanguageModel) Generate(ctx context.Context, req *provider.LanguageModelRequest) (*provider.LanguageModelResponse, error) {
	start := time.Now()
	if l.opts.LogRequest {
		l.logFn("lm.generate start model=%s", req.Model)
	}

	res, err := l.next.Generate(ctx, req)
	dur := time.Since(start)

	if err != nil {
		if l.opts.LogErrors {
			if l.opts.LogDuration {
				l.logFn("lm.generate error model=%s duration=%s err=%v", req.Model, dur, err)
			} else {
				l.logFn("lm.generate error model=%s err=%v", req.Model, err)
			}
		}
		return nil, err
	}

	if l.opts.LogResponse {
		if l.opts.LogDuration {
			l.logFn("lm.generate success model=%s duration=%s", req.Model, dur)
		} else {
			l.logFn("lm.generate success model=%s", req.Model)
		}
	} else if l.opts.LogDuration {
		l.logFn("lm.generate done model=%s duration=%s", req.Model, dur)
	}

	return res, nil
}

func (l *loggingLanguageModel) Stream(ctx context.Context, req *provider.LanguageModelRequest) (provider.LanguageModelStream, error) {
	if l.opts.LogRequest {
		l.logFn("lm.stream start model=%s", req.Model)
	}

	stream, err := l.next.Stream(ctx, req)
	if err != nil {
		if l.opts.LogErrors {
			l.logFn("lm.stream error model=%s err=%v", req.Model, err)
		}
		return nil, err
	}

	if l.opts.LogResponse {
		l.logFn("lm.stream established model=%s", req.Model)
	}

	return stream, nil
}

// RetryOptions configures the retry middleware for language-model calls.
type RetryOptions struct {
	// MaxAttempts is the maximum number of attempts, including the first
	// call. If zero or negative, a default of 3 attempts is used.
	MaxAttempts int
	// InitialBackoff is the delay before the first retry. If zero, a
	// default of 100ms is used.
	InitialBackoff time.Duration
	// MaxBackoff caps the backoff delay. If zero, no cap is applied.
	MaxBackoff time.Duration
	// ShouldRetry determines whether a given error is considered
	// transient and should be retried. If nil, a default implementation
	// that treats temporary and timeout network errors as transient is
	// used.
	ShouldRetry func(error) bool
}

func defaultRetryOptions(opts RetryOptions) RetryOptions {
	if opts.MaxAttempts <= 0 {
		opts.MaxAttempts = 3
	}
	if opts.InitialBackoff <= 0 {
		opts.InitialBackoff = 100 * time.Millisecond
	}
	if opts.ShouldRetry == nil {
		opts.ShouldRetry = isTransientError
	}
	return opts
}

// RetryLanguageModel returns a LanguageModelMiddleware that retries
// Generate and Stream calls when ShouldRetry returns true for the
// encountered error. Retries respect the provided context for
// cancellation.
func RetryLanguageModel(opts RetryOptions) LanguageModelMiddleware {
	opts = defaultRetryOptions(opts)

	return func(next provider.LanguageModel) provider.LanguageModel {
		return &retryLanguageModel{
			next: next,
			opt:  opts,
		}
	}
}

type retryLanguageModel struct {
	next provider.LanguageModel
	opt  RetryOptions
}

func (r *retryLanguageModel) Generate(ctx context.Context, req *provider.LanguageModelRequest) (*provider.LanguageModelResponse, error) {
	var lastErr error

	backoff := r.opt.InitialBackoff
	for attempt := 1; attempt <= r.opt.MaxAttempts; attempt++ {
		if attempt > 1 {
			if err := sleepWithContext(ctx, backoff); err != nil {
				return nil, err
			}
			backoff = nextBackoff(backoff, r.opt.MaxBackoff)
		}

		res, err := r.next.Generate(ctx, req)
		if err == nil {
			return res, nil
		}
		// Do not retry on context cancellation.
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return nil, err
		}
		if !r.opt.ShouldRetry(err) {
			return nil, err
		}
		lastErr = err
	}

	if lastErr != nil {
		return nil, lastErr
	}
	return nil, errors.New("middleware: retry: exhausted attempts with no result")
}

func (r *retryLanguageModel) Stream(ctx context.Context, req *provider.LanguageModelRequest) (provider.LanguageModelStream, error) {
	var stream provider.LanguageModelStream
	var lastErr error

	backoff := r.opt.InitialBackoff
	for attempt := 1; attempt <= r.opt.MaxAttempts; attempt++ {
		if attempt > 1 {
			if err := sleepWithContext(ctx, backoff); err != nil {
				return nil, err
			}
			backoff = nextBackoff(backoff, r.opt.MaxBackoff)
		}

		res, err := r.next.Stream(ctx, req)
		if err == nil {
			stream = res
			break
		}
		// Do not retry on context cancellation.
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return nil, err
		}
		if !r.opt.ShouldRetry(err) {
			return nil, err
		}
		lastErr = err
	}

	if stream == nil {
		if lastErr != nil {
			return nil, lastErr
		}
		return nil, errors.New("middleware: retry: exhausted attempts with no stream")
	}

	return stream, nil
}

// sleepWithContext sleeps for the given duration or returns early if
// the context is cancelled.
func sleepWithContext(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// nextBackoff computes the next backoff delay using exponential
// backoff with an optional maximum cap.
func nextBackoff(current, max time.Duration) time.Duration {
	next := current * 2
	if max > 0 && next > max {
		return max
	}
	return next
}

// isTransientError reports whether err looks like a transient network
// error suitable for retry (timeouts or temporary network failures).
func isTransientError(err error) bool {
	var netErr net.Error
	if errors.As(err, &netErr) {
		return netErr.Timeout() || netErr.Temporary()
	}
	return false
}

// LanguageModelCallKind describes the kind of language-model call for
// telemetry purposes.
type LanguageModelCallKind string

const (
	// LanguageModelCallGenerate represents a non-streaming Generate call.
	LanguageModelCallGenerate LanguageModelCallKind = "generate"
	// LanguageModelCallStream represents establishing a streaming call.
	LanguageModelCallStream LanguageModelCallKind = "stream"
)

// LanguageModelCallInfo contains high-level metadata about a
// language-model call that can be used for metrics or tracing.
type LanguageModelCallInfo struct {
	Kind      LanguageModelCallKind
	Model     string
	StartTime time.Time
	EndTime   time.Time
	Err       error
}

// TelemetryHooks defines callbacks that are invoked around language
// model calls. These hooks are intentionally generic so that callers
// can integrate with metrics/tracing systems such as OpenTelemetry
// without this package taking a hard dependency on them.
type TelemetryHooks struct {
	OnLanguageModelCall func(ctx context.Context, info LanguageModelCallInfo)
}

// TelemetryLanguageModel returns a LanguageModelMiddleware that invokes
// the provided telemetry hooks around Generate and Stream calls.
func TelemetryLanguageModel(hooks TelemetryHooks) LanguageModelMiddleware {
	return func(next provider.LanguageModel) provider.LanguageModel {
		return &telemetryLanguageModel{
			next:  next,
			hooks: hooks,
		}
	}
}

type telemetryLanguageModel struct {
	next  provider.LanguageModel
	hooks TelemetryHooks
}

func (t *telemetryLanguageModel) Generate(ctx context.Context, req *provider.LanguageModelRequest) (*provider.LanguageModelResponse, error) {
	start := time.Now()
	res, err := t.next.Generate(ctx, req)
	if t.hooks.OnLanguageModelCall != nil {
		t.hooks.OnLanguageModelCall(ctx, LanguageModelCallInfo{
			Kind:      LanguageModelCallGenerate,
			Model:     req.Model,
			StartTime: start,
			EndTime:   time.Now(),
			Err:       err,
		})
	}
	return res, err
}

func (t *telemetryLanguageModel) Stream(ctx context.Context, req *provider.LanguageModelRequest) (provider.LanguageModelStream, error) {
	start := time.Now()
	stream, err := t.next.Stream(ctx, req)
	if t.hooks.OnLanguageModelCall != nil {
		t.hooks.OnLanguageModelCall(ctx, LanguageModelCallInfo{
			Kind:      LanguageModelCallStream,
			Model:     req.Model,
			StartTime: start,
			EndTime:   time.Now(),
			Err:       err,
		})
	}
	return stream, err
}
