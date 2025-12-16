package ai

// Conversation is a small helper for building chat
// message histories in a convenient, chainable way.
//
// It is purely a convenience type; you can always
// construct []Message directly.
//
// Example:
//
//	conv := NewConversation().
//	    System("You are a helpful assistant.").
//	    User("Hello!")
//	res, err := GenerateText(ctx, GenerateTextRequest{
//	    Model:    model,
//	    Messages: conv.Messages,
//	})
//
// Conversation is safe to reuse by appending more
// messages over time.
type Conversation struct {
	Messages []Message
}

// NewConversation creates an empty Conversation.
func NewConversation() *Conversation {
	return &Conversation{}
}

// System appends a system message and returns the
// Conversation for chaining.
func (c *Conversation) System(content string) *Conversation {
	c.Messages = append(c.Messages, Message{Role: RoleSystem, Content: content})
	return c
}

// User appends a user message and returns the
// Conversation for chaining.
func (c *Conversation) User(content string) *Conversation {
	c.Messages = append(c.Messages, Message{Role: RoleUser, Content: content})
	return c
}

// Assistant appends an assistant message and returns
// the Conversation for chaining.
func (c *Conversation) Assistant(content string) *Conversation {
	c.Messages = append(c.Messages, Message{Role: RoleAssistant, Content: content})
	return c
}
