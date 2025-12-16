package ai

import "context"

// Embed is a convenience helper for generating an embedding vector for
// a single input string using the given embedding model.
func Embed(ctx context.Context, model EmbeddingModel, input string) ([]float32, error) {
	res, err := GenerateEmbeddings(ctx, EmbeddingRequest{
		Model: model,
		Input: []string{input},
	})
	if err != nil {
		return nil, err
	}
	if len(res.Embeddings) == 0 {
		return nil, ErrNoEmbeddingGenerated
	}
	return res.Embeddings[0], nil
}

// EmbedMany generates embeddings for a batch of input strings using
// the given embedding model.
func EmbedMany(ctx context.Context, model EmbeddingModel, inputs []string) ([][]float32, error) {
	res, err := GenerateEmbeddings(ctx, EmbeddingRequest{
		Model: model,
		Input: inputs,
	})
	if err != nil {
		return nil, err
	}
	if len(res.Embeddings) == 0 {
		return nil, ErrNoEmbeddingGenerated
	}
	return res.Embeddings, nil
}
