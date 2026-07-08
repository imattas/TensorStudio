from __future__ import annotations

import tempfile
from pathlib import Path

import tensorstudio as ts
from tensorstudio import nn


def demo_sparse() -> None:
    sparse = ts.sparse_coo_tensor(
        [[0, 1], [1, 0], [1, 0]],
        [2.0, 3.0, 4.0],
        (2, 2),
        coalesced=False,
    ).coalesce()

    print("sparse dense:", sparse.to_dense().tolist())
    print("sparse matmul:", (sparse @ ts.ones((2, 1))).tolist())


def demo_public_datasets(tmpdir: Path) -> None:
    path = tmpdir / "samples.csv"
    path.write_text("x,y,label\n1,2,0\n3,4,1\n", encoding="utf-8")

    dataset = ts.data.from_csv(path, target_column="label", target_dtype="int64")
    features, label = dataset[0]
    print("csv sample:", features.tolist(), int(label.item()))


def demo_language_model() -> None:
    vocab = nn.Vocabulary.build(["small tensor models", "small language batches"])
    token_ids = vocab.encode("small tensor models")
    inputs, targets = nn.make_causal_lm_batch(token_ids, sequence_length=2)

    model = nn.CausalLanguageModel(vocab_size=len(vocab), embedding_dim=4, max_length=4)
    loss = nn.causal_language_model_loss(model(inputs), targets)
    print("tiny lm loss:", round(loss.item(), 4))


def demo_model_zoo_quantization_and_kernels() -> None:
    model = ts.create_model("tiny_mlp", input_dim=2, hidden_dim=4, output_dim=1)
    output = model(ts.ones((1, 2)))
    print("model zoo output shape:", output.shape)

    quantized = ts.quantization.quantize_tensor(ts.tensor([-1.0, 0.0, 1.0]))
    print("quantized roundtrip:", quantized.dequantize().tolist())

    ts.register_kernel("example_double", lambda x: x * 2.0, overwrite=True)
    try:
        print("kernel call:", ts.call_kernel("example_double", ts.ones((2,))).tolist())
    finally:
        ts.unregister_kernel("example_double")


def demo_distributed_and_onnx() -> None:
    plan = ts.distributed.data_parallel_plan(dataset_size=10, batch_size=4)
    print("data parallel plan:", plan["rank_batches"], plan["global_batch_size"])
    print("onnxruntime available:", ts.onnxruntime_is_available())


def main() -> None:
    ts.manual_seed(7)
    demo_sparse()
    with tempfile.TemporaryDirectory() as directory:
        demo_public_datasets(Path(directory))
    demo_language_model()
    demo_model_zoo_quantization_and_kernels()
    demo_distributed_and_onnx()


if __name__ == "__main__":
    main()
