# Model Zoo And Language Helpers

The model zoo is a small registry of reproducible local model factories. It
does not download pretrained weights.

```python
import tensorstudio as ts

print(ts.list_models())
model = ts.create_model("tiny_mlp", input_dim=2, hidden_dim=8, output_dim=1)
print(ts.model_info("tiny_mlp"))
```

Built-in factories:

- `tiny_mlp`
- `tiny_cnn`
- `tiny_bigram_language_model`

## Language Helpers

```python
from tensorstudio import nn

vocab = nn.Vocabulary.build(["hello tensor world", "hello small model"])
ids = vocab.encode("hello model")
inputs, targets = nn.make_causal_lm_batch(ids + ids + ids, sequence_length=2)

model = nn.CausalLanguageModel(vocab_size=len(vocab), embedding_dim=8, max_length=8)
logits = model(inputs)
loss = nn.causal_language_model_loss(logits, targets)
```

The language utilities are compact training-loop building blocks, not a
Transformer stack or pretrained language-model runtime.
