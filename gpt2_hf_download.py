# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""
Download GPT-2 from ModelScope (modelscope.cn, Alibaba Cloud - fast in mainland China)
and convert weights to the format expected by the book's download_and_load_gpt2().
"""

import os
import sys
import json
import numpy as np


def download_and_load_gpt2_hf(model_size="124M", models_dir="gpt2-hf"):
    """
    Download GPT-2 from ModelScope (fast in China) and load it
    in the same (settings, params) format as download_and_load_gpt2().
    """

    import torch

    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Map model_size to ModelScope model ID
    # These are mirrored from HuggingFace by AI-ModelScope community
    size_to_ms = {
        "124M": "AI-ModelScope/gpt2",
        "355M": "AI-ModelScope/gpt2-medium",
        "774M": "AI-ModelScope/gpt2-large",
        "1558M": "AI-ModelScope/gpt2-xl"
    }
    model_id = size_to_ms[model_size]

    model_dir = os.path.join(models_dir, model_size)
    os.makedirs(model_dir, exist_ok=True)

    print(f"📡 下载源: ModelScope (modelscope.cn)")
    print(f"📥 下载 GPT-2 {model_size} 到 {model_dir}...")

    # Check if already fully downloaded
    config_path = os.path.join(model_dir, "config.json")
    model_path = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.exists(config_path) and os.path.exists(model_path):
        print("✅ 模型文件已存在，跳过下载")
    else:
        from modelscope.hub.snapshot_download import snapshot_download
        snapshot_download(
            model_id=model_id,
            local_dir=model_dir,
            revision="master",
        )
        print("✅ 下载完成！")

    # Load config (settings)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    settings = {
        "n_vocab": config["vocab_size"],
        "n_ctx": config["n_positions"],
        "n_embd": config["n_embd"],
        "n_head": config["n_head"],
        "n_layer": config["n_layer"],
    }

    # Load PyTorch state dict and convert to the book's NumPy format
    print("🔄 转换 PyTorch 权重为书籍格式...")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    # Convert PyTorch weights to the NumPy-based format from the book
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Helper: convert torch tensor to numpy
    def tn(t):
        return t.detach().cpu().numpy()

    # Detect key prefix (some models have "transformer." prefix, some don't)
    sample_key = [k for k in state_dict.keys() if "wte" in k][0]
    prefix = "transformer." if sample_key.startswith("transformer.") else ""
    print(f"  Key prefix: '{prefix}' (detected from '{sample_key}')")

    # wte (token embeddings)
    params["wte"] = tn(state_dict[f"{prefix}wte.weight"])
    # wpe (position embeddings)
    params["wpe"] = tn(state_dict[f"{prefix}wpe.weight"])
    # final layer norm — stored at top level as "b"/"g" to match original TF format
    params["b"] = tn(state_dict[f"{prefix}ln_f.bias"])
    params["g"] = tn(state_dict[f"{prefix}ln_f.weight"])

    n_embd = settings["n_embd"]

    # Per-block weights
    for i in range(settings["n_layer"]):
        block = params["blocks"][i]
        p = f"{prefix}h.{i}."

        # Layer norms — use "ln_1"/"ln_2" to match original TF checkpoint format
        block["ln_1"] = {
            "b": tn(state_dict[p + "ln_1.bias"]),
            "g": tn(state_dict[p + "ln_1.weight"]),
        }
        block["ln_2"] = {
            "b": tn(state_dict[p + "ln_2.bias"]),
            "g": tn(state_dict[p + "ln_2.weight"]),
        }

        # Attention c_attn — keep Q,K,V merged (same as original TF format),
        # the load_weights_into_gpt function will split them
        block["attn"] = {
            "c_attn": {
                "w": tn(state_dict[p + "attn.c_attn.weight"]),  # [n_embd, 3*n_embd]
                "b": tn(state_dict[p + "attn.c_attn.bias"]),    # [3*n_embd]
            },
            "c_proj": {
                "w": tn(state_dict[p + "attn.c_proj.weight"]),
                "b": tn(state_dict[p + "attn.c_proj.bias"]),
            },
        }

        # Feed-forward
        block["mlp"] = {
            "c_fc": {
                "w": tn(state_dict[p + "mlp.c_fc.weight"]),
                "b": tn(state_dict[p + "mlp.c_fc.bias"]),
            },
            "c_proj": {
                "w": tn(state_dict[p + "mlp.c_proj.weight"]),
                "b": tn(state_dict[p + "mlp.c_proj.bias"]),
            },
        }

    print(f"✅ GPT-2 {model_size} 加载完成！")
    return settings, params


# Test
if __name__ == "__main__":
    settings, params = download_and_load_gpt2_hf(model_size="355M", models_dir="gpt2-hf")
    print("\nSettings:", json.dumps(settings, indent=2))
    print(f"\nNumber of transformer blocks: {len(params['blocks'])}")
    print(f"Token embedding shape: {params['wte'].shape}")
    print(f"Position embedding shape: {params['wpe'].shape}")
