package agent

import (
	"context"
	"encoding/json"
	"fmt"

	ai "github.com/ncecere/ai-sdk"
	"github.com/ncecere/ai-sdk/registry"
)

// EventType describes the kind of agent event.
type EventType string

const (
	EventTypeMessage    EventType = "message"
	EventTypeToolStart  EventType = "tool_start"
	EventTypeToolResult EventType = "tool_result"
	EventTypeError      EventType = "error"
	EventTypeDone       EventType = "done"
)

// Event represents a single step in an agent run that can be streamed
// to callers (for example over Server-Sent Events).
type Event struct {
	Type EventType `json:"type"`
	// Step is the zero-based tool-loop iteration count.
	Step int `json:"step,omitempty"`
	// Role is set for message events (e.g. "assistant" or "tool").
	Role string `json:"role,omitempty"`
	// Content contains message text for message events or an error
	// description for error events.
	Content string `json:"content,omitempty"`
	// Tool is the name of the tool for tool-related events.
	Tool string `json:"tool,omitempty"`
}

// EventEmitter is a callback used to observe agent events.
type EventEmitter func(Event)

// Tool represents a callable tool that can be used by an agent.
//
// Tools are identified by name and expose a JSON-schema description
// of their parameters along with an Execute function that performs the
// actual work when the language model requests the tool.
type Tool struct {
	// Name is the unique tool name used in ToolDefinition and by the model.
	Name string
	// Description is a human-readable description of the tool.
	Description string
	// Parameters is an optional JSON Schema describing the tool input.
	Parameters json.RawMessage
	// Execute is invoked when the model calls this tool. The args
	// parameter contains the raw JSON arguments provided by the model.
	Execute func(ctx context.Context, args json.RawMessage) (any, error)
}

// Config contains the static configuration for an agent run.
type Config struct {
	// Registry is used to resolve the language model by name.
	Registry registry.Registry
	// ModelName is the registry key for the language model the agent
	// should use for reasoning and tool selection.
	ModelName string

	// Tools is a map from tool name to tool implementation. The keys
	// should match the Tool.Name field.
	Tools map[string]Tool

	// MaxSteps controls how many tool-loop iterations the agent may run
	// before returning an error. If zero or negative, a default of 8 is
	// used.
	MaxSteps int
}

// Result represents the outcome of an agent run.
type Result struct {
	// Messages is the full conversation history including user,
	// assistant, and tool messages.
	Messages []ai.Message
	// FinalText is the final assistant message text, if any.
	FinalText string
	// Steps is the number of tool-loop iterations executed.
	Steps int
}

func (c *Config) validate() error {
	if c.Registry == nil {
		return &ai.InvalidArgumentError{Parameter: "Registry", Value: nil, Message: "must not be nil"}
	}
	if c.ModelName == "" {
		return &ai.InvalidArgumentError{Parameter: "ModelName", Value: c.ModelName, Message: "must not be empty"}
	}
	return nil
}

func maxStepsOrDefault(v int) int {
	if v <= 0 {
		return 8
	}
	return v
}

// Run executes a simple tool-loop agent using the provided configuration
// and initial messages.
//
// The agent repeatedly calls the configured language model, passing the
// current message history and tool definitions. If the model returns
// tool calls, the corresponding tools are executed and their JSON
// results are appended as tool messages. The loop continues until the
// model returns no tool calls or MaxSteps is reached.
func Run(ctx context.Context, cfg Config, initialMessages []ai.Message) (*Result, error) {
	return RunWithEvents(ctx, cfg, initialMessages, nil)
}

// RunWithEvents is like Run but invokes the provided EventEmitter for
// each significant step (assistant messages, tool start, tool result,
// and completion). This is useful for driving streaming UIs such as
// Server-Sent Events or CLIs that want incremental updates.
func RunWithEvents(ctx context.Context, cfg Config, initialMessages []ai.Message, emit EventEmitter) (*Result, error) {
	if err := cfg.validate(); err != nil {
		return nil, err
	}

	emitEvent := func(e Event) {
		if emit != nil {
			emit(e)
		}
	}

	messages := append([]ai.Message(nil), initialMessages...)
	steps := 0
	maxSteps := maxStepsOrDefault(cfg.MaxSteps)

	for {
		if steps >= maxSteps {
			err := &ai.UnsupportedFunctionalityError{
				Feature: "agent.maxSteps",
				Message: fmt.Sprintf("maximum steps (%d) exceeded", maxSteps),
			}
			emitEvent(Event{Type: EventTypeError, Step: steps, Content: err.Error()})
			return nil, err
		}

		// Build tool definitions from the configured tools.
		var toolDefs []ai.ToolDefinition
		if len(cfg.Tools) > 0 {
			toolDefs = make([]ai.ToolDefinition, 0, len(cfg.Tools))
			for name, t := range cfg.Tools {
				params := []byte(nil)
				if len(t.Parameters) > 0 {
					params = t.Parameters
				}
				toolDefs = append(toolDefs, ai.ToolDefinition{
					Name:        name,
					Description: t.Description,
					Parameters:  params,
				})
			}
		}

		res, err := ai.GenerateTextWithRegistry(ctx, cfg.Registry, cfg.ModelName, ai.GenerateTextRequest{
			Messages: messages,
			Tools:    toolDefs,
		})
		if err != nil {
			emitEvent(Event{Type: EventTypeError, Step: steps, Content: err.Error()})
			return nil, err
		}

		if res.Text != "" {
			messages = append(messages, ai.Message{
				Role:    ai.RoleAssistant,
				Content: res.Text,
			})
			emitEvent(Event{
				Type:    EventTypeMessage,
				Step:    steps,
				Role:    ai.RoleAssistant,
				Content: res.Text,
			})
		}

		if len(res.ToolCalls) == 0 {
			emitEvent(Event{Type: EventTypeDone, Step: steps})
			return &Result{
				Messages:  messages,
				FinalText: res.Text,
				Steps:     steps,
			}, nil
		}

		for _, tc := range res.ToolCalls {
			tool, ok := cfg.Tools[tc.Name]
			if !ok {
				err := &ai.UnsupportedFunctionalityError{
					Feature: "agent.tool",
					Message: fmt.Sprintf("no tool registered with name %q", tc.Name),
				}
				emitEvent(Event{Type: EventTypeError, Step: steps, Content: err.Error(), Tool: tc.Name})
				return nil, err
			}

			emitEvent(Event{Type: EventTypeToolStart, Step: steps, Tool: tool.Name})

			args := json.RawMessage(tc.RawArguments)
			result, err := tool.Execute(ctx, args)
			if err != nil {
				emitEvent(Event{Type: EventTypeError, Step: steps, Content: err.Error(), Tool: tool.Name})
				return nil, err
			}

			payload := map[string]any{
				"tool":   tool.Name,
				"result": result,
			}
			data, err := json.Marshal(payload)
			if err != nil {
				emitEvent(Event{Type: EventTypeError, Step: steps, Content: err.Error(), Tool: tool.Name})
				return nil, err
			}

			messages = append(messages, ai.Message{
				Role:    ai.RoleTool,
				Content: string(data),
			})
			emitEvent(Event{Type: EventTypeToolResult, Step: steps, Tool: tool.Name})
		}

		steps++
	}
}
