import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm.gguf import convert_to_gguf
from mlx_lm.utils import load_model
import mlx_lm.gguf as gguf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    model, config = load_model(args.model)
    weights = dict(tree_flatten(model.parameters()))
    original_save = gguf.mx.save_gguf

    # MLX's fused Llama weights already use the attention layout expected by
    # llama.cpp.  The generic MLX exporter otherwise permutes Q/K a second
    # time, which produces a loadable but behaviorally incorrect GGUF.
    gguf.permute_weights = lambda values, *unused: values

    original_prepare_metadata = gguf.prepare_metadata

    def complete_metadata(model_config, vocab):
        metadata = original_prepare_metadata(model_config, vocab)
        tokenizer = json.loads((args.model / "tokenizer.json").read_text())
        merges = tokenizer.get("model", {}).get("merges", [])
        metadata.update(
            {
                "general.basename": "Llama-3.2",
                "general.finetune": "Sacred-Harp",
                "general.size_label": "1B",
                "general.type": "model",
                "general.languages": ["en"],
                "general.tags": ["llama", "sacred-harp"],
                "llama.attention.key_length": mx.array(64, dtype=mx.uint32),
                "llama.attention.value_length": mx.array(64, dtype=mx.uint32),
                "llama.vocab_size": mx.array(128256, dtype=mx.uint32),
                "tokenizer.ggml.pre": "llama-bpe",
                "tokenizer.ggml.add_bos_token": mx.array(True, dtype=mx.bool_),
                "tokenizer.ggml.add_sep_token": mx.array(False, dtype=mx.bool_),
                "tokenizer.ggml.merges": [
                    " ".join(pair) if isinstance(pair, list) else pair
                    for pair in merges
                ],
                "tokenizer.chat_template": (
                    args.model / "chat_template.jinja"
                ).read_text(),
            }
        )
        token_types = metadata["tokenizer.ggml.token_type"]
        metadata["tokenizer.ggml.token_type"] = mx.array(
            [3 if int(value) == 4 else int(value) for value in token_types],
            dtype=mx.uint32,
        )
        metadata.pop("tokenizer.ggml.scores", None)
        metadata.pop("general.alignment", None)
        return metadata

    gguf.prepare_metadata = complete_metadata

    def contiguous_save(path, values, metadata):
        values = {name: mx.contiguous(value) for name, value in values.items()}
        return original_save(path, values, metadata)

    gguf.mx.save_gguf = contiguous_save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    convert_to_gguf(args.model, weights, config, str(args.output))


if __name__ == "__main__":
    main()
