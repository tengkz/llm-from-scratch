"""
在 Wikipedia 中文语料上预训练 GPT 模型
用法: python train_wiki.py
"""

import torch
import torch.nn as nn
import tiktoken
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import os
import json
import time
import argparse

# ============================================================
# 1. 模型组件（与 train_sgyy.py 一致）
# ============================================================

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(-2, -1)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -float("inf"))

        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, x):
        batch_size, seq_len = x.shape
        tok_embeds = self.tok_emb(x)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=x.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# ============================================================
# 2. 数据处理
# ============================================================

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128,
                         shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader


def load_wikipedia_data(max_tokens=None, data_dir="/home/tengkz/LLM/data/wikipedia-zh-cn"):
    """
    加载 Wikipedia 中文语料，用 <|endoftext|> 拼接所有条目
    max_tokens: 限制最大 token 数（None = 全部加载）
    优先从本地缓存加载，避免网络问题
    """
    print("📖 加载 Wikipedia 中文语料...")
    from datasets import Dataset

    # 优先从本地缓存加载（绕过网络和哈希匹配问题）
    from datasets import concatenate_datasets
    cache_base = os.path.join(
        data_dir,
        "fjcanyue___wikipedia-zh-cn"
    )
    # 找到缓存中的实际目录（哈希名）
    if os.path.isdir(cache_base):
        config_dirs = [
            d for d in os.listdir(cache_base)
            if os.path.isdir(os.path.join(cache_base, d))
        ]
        if config_dirs:
            config_dir = os.path.join(cache_base, config_dirs[0])
            version_dirs = [
                d for d in os.listdir(config_dir)
                if os.path.isdir(os.path.join(config_dir, d))
            ]
            if version_dirs:
                version_dir = os.path.join(config_dir, version_dirs[0])
                hash_dirs = [
                    d for d in os.listdir(version_dir)
                    if os.path.isdir(os.path.join(version_dir, d))
                ]
                if hash_dirs:
                    arrow_dir = os.path.join(version_dir, hash_dirs[0])
                    arrow_files = sorted([
                        os.path.join(arrow_dir, f)
                        for f in os.listdir(arrow_dir)
                        if f.endswith(".arrow")
                    ])
                    if arrow_files:
                        print(f"   从缓存加载: {len(arrow_files)} 个 arrow 文件")
                        parts = [Dataset.from_file(af) for af in arrow_files]
                        ds = concatenate_datasets(parts)
                        print(f"   数据集条目: {len(ds):,}")
                    else:
                        raise FileNotFoundError(f"缓存中没有 arrow 文件: {arrow_dir}")
                else:
                    raise FileNotFoundError(f"未找到数据哈希目录: {version_dir}")
            else:
                raise FileNotFoundError(f"未找到版本目录: {config_dir}")
        else:
            raise FileNotFoundError(f"未找到配置目录: {cache_base}")
    else:
        raise FileNotFoundError(f"缓存目录不存在: {cache_base}\n请先下载数据集")

    # 拼接所有文本，用 <|endoftext|> 分隔
    print("🔗 拼接文本...")
    tokenizer = tiktoken.get_encoding("gpt2")
    full_text = ""
    item_count = 0

    for item in ds:
        text = item["text"].strip()
        if not text:
            continue
        full_text += text + "<|endoftext|>"
        item_count += 1

        # 每 20000 条估算并检查 token 是否达标
        if max_tokens and item_count % 20000 == 0:
            # 粗略估算：中文约 2 token/字符
            if len(full_text) * 2 > max_tokens:
                token_ids = tokenizer.encode(full_text, allowed_special={"<|endoftext|>"})
                if len(token_ids) >= max_tokens:
                    break

    # 确保不超过 max_tokens（精确截断）
    if max_tokens:
        token_ids = tokenizer.encode(full_text, allowed_special={"<|endoftext|>"})
        if len(token_ids) > max_tokens:
            token_ids = token_ids[:max_tokens]
            full_text = tokenizer.decode(token_ids)
            print(f"   ⚠️ 截断到 {max_tokens:,} tokens")

    total_tokens = len(tokenizer.encode(full_text, allowed_special={"<|endoftext|>"}))
    print(f"   总字符数: {len(full_text):,}")
    print(f"   总 token 数: {total_tokens:,}")

    return full_text


# ============================================================
# 3. 训练工具函数
# ============================================================

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # 每个 epoch 结束生成一段样本
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


# ============================================================
# 4. 文本生成
# ============================================================

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=-1)
    return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits,
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


# ============================================================
# 5. 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Wikipedia 中文语料 GPT 训练")
    parser.add_argument("--max_tokens", type=int, default=50_000_000,
                        help="最大训练 token 数 (默认: 5000万, 约 200MB 文本)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="训练 epoch 数 (默认: 3)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (默认: 4)")
    args = parser.parse_args()

    # ---- 配置 ----
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 512,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": True,
    }

    # 训练参数
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 0.1
    EVAL_FREQ = 500
    MAX_TOKENS = args.max_tokens

    # ---- 设备 ----
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"🖥  使用设备: {device}")

    # ---- 加载数据 ----
    print(f"📦 数据量上限: {MAX_TOKENS:,} tokens")
    # 用 HF 镜像，避免国内网络问题
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    text_data = load_wikipedia_data(max_tokens=MAX_TOKENS)

    # 加载 tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    total_tokens = len(tokenizer.encode(text_data, allowed_special={"<|endoftext|>"}))
    print(f"   实际使用 token 数: {total_tokens:,}")

    # 划分训练集 / 验证集（95% / 5%）
    split_idx = int(len(text_data) * 0.95)
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    print(f"   训练集字符: {len(train_data):,}, 验证集字符: {len(val_data):,}")

    # 创建 DataLoader
    torch.manual_seed(123)
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=BATCH_SIZE,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"] // 2,  # 50% 重叠
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=BATCH_SIZE,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],  # 验证集不重叠
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    print(f"   训练 batches: {len(train_loader):,}, 验证 batches: {len(val_loader):,}")

    # ---- 创建模型 ----
    print("🏗️  创建 GPT 模型...")
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   参数量: {total_params:,}")
    print(f"   context_length: {GPT_CONFIG_124M['context_length']}")

    # ---- 优化器 ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ---- 训练 ----
    START_CONTEXT = "人工智能"
    print(f"\n🚀 开始训练 ({NUM_EPOCHS} epochs, {BATCH_SIZE} batch_size)...\n")
    start_time = time.time()

    train_losses, val_losses, tokens_seen = train_model_simple(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS,
        eval_freq=EVAL_FREQ,
        eval_iter=5,
        start_context=START_CONTEXT,
        tokenizer=tokenizer,
    )

    elapsed = time.time() - start_time
    print(f"\n⏱  训练用时: {elapsed // 60:.0f} 分 {elapsed % 60:.0f} 秒")
    print(f"   处理 token 数: {tokens_seen[-1]:,}")

    # ---- 保存模型 ----
    save_dir = "wiki_model"
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"💾 模型已保存至: {model_path}")

    optim_path = os.path.join(save_dir, "model_and_optimizer.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, optim_path)
    print(f"💾 检查点已保存至: {optim_path}")

    # 保存配置
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "model_config": GPT_CONFIG_124M,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "max_tokens": MAX_TOKENS,
            "actual_tokens": total_tokens,
        }, f, indent=2, ensure_ascii=False)
    print(f"💾 配置已保存至: {config_path}")

    # 保存训练曲线数据
    history_path = os.path.join(save_dir, "train_history.json")
    with open(history_path, "w") as f:
        json.dump({
            "train_losses": train_losses,
            "val_losses": val_losses,
            "tokens_seen": tokens_seen,
            "elapsed_seconds": elapsed,
        }, f, indent=2, ensure_ascii=False)
    print(f"💾 训练历史已保存至: {history_path}")

    # ---- 生成最终样本 ----
    print("\n" + "=" * 60)
    print("📝 最终生成样本")
    print("=" * 60)

    model.eval()
    PROMPTS = [
        "人工智能",
        "中国历史",
        "数学是",
        "物理学",
        "地球",
        "人类",
        "计算机",
    ]
    for prompt in PROMPTS:
        input_ids = text_to_token_ids(prompt, tokenizer).to(device)
        with torch.no_grad():
            output_ids = generate(
                model=model,
                idx=input_ids,
                max_new_tokens=100,
                context_size=GPT_CONFIG_124M["context_length"],
                temperature=0.8,
                top_k=40,
            )
        generated = token_ids_to_text(output_ids, tokenizer)
        print(f"\nPrompt: 「{prompt}」")
        print(f"生成:   {generated}")


if __name__ == "__main__":
    main()
