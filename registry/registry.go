package registry

import (
	"fmt"
	"sync"

	"github.com/ncecere/ai-sdk/provider"
)

// Registry is a simple, provider-agnostic registry for models.
//
// It maps string model identifiers (for example, "gpt-4o" or
// "openai:gpt-4o") to concrete provider implementations. This
// allows application code and higher-level helpers to look up
// models by name without depending directly on specific provider
// packages.
type Registry interface {
	// LanguageModel returns the registered language model for the given name.
	// If no such model exists, a *NoSuchModelError is returned.
	LanguageModel(name string) (provider.LanguageModel, error)

	// EmbeddingModel returns the registered embedding model for the given name.
	// If no such model exists, a *NoSuchModelError is returned.
	EmbeddingModel(name string) (provider.EmbeddingModel, error)

	// CompletionModel returns the registered completion model for the given name.
	// If no such model exists, a *NoSuchModelError is returned.
	CompletionModel(name string) (provider.CompletionModel, error)

	// ImageModel returns the registered image model for the given name.
	// If no such model exists, a *NoSuchModelError is returned.
	ImageModel(name string) (provider.ImageModel, error)

	// SpeechModel returns the registered text-to-speech model for the given name.
	// If no such model exists, a *NoSuchModelError is returned.
	SpeechModel(name string) (provider.SpeechModel, error)

	// TranscriptionModel returns the registered transcription model for the given name.
	// If no such model exists, a *NoSuchModelError is returned.
	TranscriptionModel(name string) (provider.TranscriptionModel, error)

	// RerankModel returns the registered rerank model for the given name.
	// If no such model exists, a *NoSuchModelError is returned.
	RerankModel(name string) (provider.RerankModel, error)

	// RegisterLanguageModel registers or replaces a language model under the given name.
	// Passing a nil model removes any existing registration for that name.
	RegisterLanguageModel(name string, model provider.LanguageModel)

	// RegisterEmbeddingModel registers or replaces an embedding model under the given name.
	// Passing a nil model removes any existing registration for that name.
	RegisterEmbeddingModel(name string, model provider.EmbeddingModel)

	// RegisterCompletionModel registers or replaces a completion model under the given name.
	// Passing a nil model removes any existing registration for that name.
	RegisterCompletionModel(name string, model provider.CompletionModel)

	// RegisterImageModel registers or replaces an image model under the given name.
	// Passing a nil model removes any existing registration for that name.
	RegisterImageModel(name string, model provider.ImageModel)

	// RegisterSpeechModel registers or replaces a speech model under the given name.
	// Passing a nil model removes any existing registration for that name.
	RegisterSpeechModel(name string, model provider.SpeechModel)

	// RegisterTranscriptionModel registers or replaces a transcription model under the given name.
	// Passing a nil model removes any existing registration for that name.
	RegisterTranscriptionModel(name string, model provider.TranscriptionModel)

	// RegisterRerankModel registers or replaces a rerank model under the given name.
	// Passing a nil model removes any existing registration for that name.
	RegisterRerankModel(name string, model provider.RerankModel)
}

// NoSuchModelError indicates that a requested model name was not
// found in the registry.
type NoSuchModelError struct {
	// Name is the model name that was requested.
	Name string
	// Kind is the optional kind of model, such as "language" or "embedding".
	Kind string
}

func (e *NoSuchModelError) Error() string {
	if e == nil {
		return "<nil>"
	}
	if e.Kind == "" {
		return fmt.Sprintf("registry: no such model %q", e.Name)
	}
	return fmt.Sprintf("registry: no such %s model %q", e.Kind, e.Name)
}

// InMemoryRegistry is a concurrency-safe in-memory implementation of Registry.
// It is suitable for typical application startup wiring where models are
// registered once and then used throughout the lifetime of the process.
type InMemoryRegistry struct {
	mu sync.RWMutex

	languageModels      map[string]provider.LanguageModel
	embeddingModels     map[string]provider.EmbeddingModel
	completionModels    map[string]provider.CompletionModel
	imageModels         map[string]provider.ImageModel
	speechModels        map[string]provider.SpeechModel
	transcriptionModels map[string]provider.TranscriptionModel
	rerankModels        map[string]provider.RerankModel
}

// Ensure InMemoryRegistry implements Registry.
var _ Registry = (*InMemoryRegistry)(nil)

// NewInMemoryRegistry creates a new empty in-memory registry.
func NewInMemoryRegistry() *InMemoryRegistry {
	return &InMemoryRegistry{
		languageModels:      make(map[string]provider.LanguageModel),
		embeddingModels:     make(map[string]provider.EmbeddingModel),
		completionModels:    make(map[string]provider.CompletionModel),
		imageModels:         make(map[string]provider.ImageModel),
		speechModels:        make(map[string]provider.SpeechModel),
		transcriptionModels: make(map[string]provider.TranscriptionModel),
		rerankModels:        make(map[string]provider.RerankModel),
	}
}

// LanguageModel implements Registry.LanguageModel.
func (r *InMemoryRegistry) LanguageModel(name string) (provider.LanguageModel, error) {
	r.mu.RLock()
	model, ok := r.languageModels[name]
	r.mu.RUnlock()
	if !ok || model == nil {
		return nil, &NoSuchModelError{Name: name, Kind: "language"}
	}
	return model, nil
}

// EmbeddingModel implements Registry.EmbeddingModel.
func (r *InMemoryRegistry) EmbeddingModel(name string) (provider.EmbeddingModel, error) {
	r.mu.RLock()
	model, ok := r.embeddingModels[name]
	r.mu.RUnlock()
	if !ok || model == nil {
		return nil, &NoSuchModelError{Name: name, Kind: "embedding"}
	}
	return model, nil
}

// CompletionModel implements Registry.CompletionModel.
func (r *InMemoryRegistry) CompletionModel(name string) (provider.CompletionModel, error) {
	r.mu.RLock()
	model, ok := r.completionModels[name]
	r.mu.RUnlock()
	if !ok || model == nil {
		return nil, &NoSuchModelError{Name: name, Kind: "completion"}
	}
	return model, nil
}

// ImageModel implements Registry.ImageModel.
func (r *InMemoryRegistry) ImageModel(name string) (provider.ImageModel, error) {
	r.mu.RLock()
	model, ok := r.imageModels[name]
	r.mu.RUnlock()
	if !ok || model == nil {
		return nil, &NoSuchModelError{Name: name, Kind: "image"}
	}
	return model, nil
}

// SpeechModel implements Registry.SpeechModel.
func (r *InMemoryRegistry) SpeechModel(name string) (provider.SpeechModel, error) {
	r.mu.RLock()
	model, ok := r.speechModels[name]
	r.mu.RUnlock()
	if !ok || model == nil {
		return nil, &NoSuchModelError{Name: name, Kind: "speech"}
	}
	return model, nil
}

// TranscriptionModel implements Registry.TranscriptionModel.
func (r *InMemoryRegistry) TranscriptionModel(name string) (provider.TranscriptionModel, error) {
	r.mu.RLock()
	model, ok := r.transcriptionModels[name]
	r.mu.RUnlock()
	if !ok || model == nil {
		return nil, &NoSuchModelError{Name: name, Kind: "transcription"}
	}
	return model, nil
}

// RerankModel implements Registry.RerankModel.
func (r *InMemoryRegistry) RerankModel(name string) (provider.RerankModel, error) {
	r.mu.RLock()
	model, ok := r.rerankModels[name]
	r.mu.RUnlock()
	if !ok || model == nil {
		return nil, &NoSuchModelError{Name: name, Kind: "rerank"}
	}
	return model, nil
}

// RegisterLanguageModel implements Registry.RegisterLanguageModel.
func (r *InMemoryRegistry) RegisterLanguageModel(name string, model provider.LanguageModel) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if model == nil {
		delete(r.languageModels, name)
		return
	}
	r.languageModels[name] = model
}

// RegisterEmbeddingModel implements Registry.RegisterEmbeddingModel.
func (r *InMemoryRegistry) RegisterEmbeddingModel(name string, model provider.EmbeddingModel) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if model == nil {
		delete(r.embeddingModels, name)
		return
	}
	r.embeddingModels[name] = model
}

// RegisterCompletionModel implements Registry.RegisterCompletionModel.
func (r *InMemoryRegistry) RegisterCompletionModel(name string, model provider.CompletionModel) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if model == nil {
		delete(r.completionModels, name)
		return
	}
	r.completionModels[name] = model
}

// RegisterImageModel implements Registry.RegisterImageModel.
func (r *InMemoryRegistry) RegisterImageModel(name string, model provider.ImageModel) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if model == nil {
		delete(r.imageModels, name)
		return
	}
	r.imageModels[name] = model
}

// RegisterSpeechModel implements Registry.RegisterSpeechModel.
func (r *InMemoryRegistry) RegisterSpeechModel(name string, model provider.SpeechModel) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if model == nil {
		delete(r.speechModels, name)
		return
	}
	r.speechModels[name] = model
}

// RegisterTranscriptionModel implements Registry.RegisterTranscriptionModel.
func (r *InMemoryRegistry) RegisterTranscriptionModel(name string, model provider.TranscriptionModel) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if model == nil {
		delete(r.transcriptionModels, name)
		return
	}
	r.transcriptionModels[name] = model
}

// RegisterRerankModel implements Registry.RegisterRerankModel.
func (r *InMemoryRegistry) RegisterRerankModel(name string, model provider.RerankModel) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if model == nil {
		delete(r.rerankModels, name)
		return
	}
	r.rerankModels[name] = model
}
