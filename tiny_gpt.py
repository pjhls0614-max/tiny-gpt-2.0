"""
tiny_gpt.py
============
6개 노트북(bigram -> MLP -> sequence model -> attention -> GPT)을 하나로 합친
character-level language model.

각 부품이 어느 노트북에서 왔는지:
  - NextTokenDataset        : Notebook 4 (target도 sequence: x=[t1..tT], y=[t2..t(T+1)])
  - token/position embedding: Notebook 4
  - Head (single attention) : Notebook 5 (masked self-attention)
  - MultiHeadAttention      : Notebook 6
  - FeedForward / Block     : Notebook 6 (residual + LayerNorm)
  - TinyGPT                 : Notebook 6 (block 쌓기)
  - train / sample 골격     : Notebook 1부터 그대로 재사용

기본값은 CPU에서도 몇 분 안에 "학습이 되는지" 확인할 수 있는 작은 설정이고,
교수님 원본(GPU/A100)에 맞추려면 아래 BIG_CONFIG 주석을 참고하세요.
"""

from dataclasses import dataclass
from pathlib import Path
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------------------------------------------
# 0. Config
# --------------------------------------------------------------------------------------
@dataclass
class Config:
    # --- 아래는 CPU에서 ~4분이면 도는 검증된 설정 ---
    dataset: str = "sherlock"      # "sherlock"(기본, tiny Shakespeare 대체), "shakespeare", "names", 또는 파일경로/URL
    block_size: int = 32           # context 길이
    batch_size: int = 32
    emb_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1
    lr: float = 3e-4
    epochs: int = 8
    steps_per_epoch: int = 300     # epoch당 mini-batch 수
    eval_steps: int = 40           # val loss 측정용 batch 수
    seed: int = 1337

# 교수님 원본(GPU/A100)에 맞추려면 — Colab에서 GPU 켜고 아래로:
#   block_size=64, emb_dim=128, num_heads=4, num_layers=4, dropout=0.1,
#   lr=3e-4, epochs=100, steps_per_epoch=300
# 그러면 훨씬 읽을 만한 가짜 셰익스피어가 나옵니다 (val loss ~1.5 부근).


# --------------------------------------------------------------------------------------
# 1. Data  (Notebook 3~6 공통)
# --------------------------------------------------------------------------------------
# 미리 정의된 데이터셋. 여기에 없는 것도 파일 경로나 URL을 직접 넘기면 됩니다.
PRESETS = {
    "sherlock":    "https://raw.githubusercontent.com/dscape/spell/master/test/resources/big.txt",
    "shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "names":       "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt",
}


def load_data(source: str):
    """source는 (1) PRESETS 키, (2) 로컬 파일 경로, (3) http(s) URL 셋 다 가능.
    character-level이라 어떤 텍스트든 그대로 학습할 수 있습니다."""
    import urllib.request

    if source in PRESETS:
        url, path = PRESETS[source], Path(f"data_{source}.txt")
    elif source.startswith(("http://", "https://")):
        url, path = source, Path("data_custom.txt")
    else:  # 로컬 파일 경로
        url, path = None, Path(source)

    if not path.exists():
        if url is None:
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {source}")
        urllib.request.urlretrieve(url, path)

    text = path.read_text(encoding="utf-8", errors="ignore")
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    vocab_size = len(chars)
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    return text, data, stoi, itos, vocab_size


# x = [t1..tT], y = [t2..t(T+1)]  ->  매 위치마다 "다음 토큰" 예측 (Notebook 4)
class NextTokenDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx + 1: idx + self.block_size + 1]
        return x, y


# --------------------------------------------------------------------------------------
# 2. Attention  (Notebook 5 -> 6)
# --------------------------------------------------------------------------------------
class Head(nn.Module):
    """하나의 masked self-attention head. (Notebook 5의 SingleHeadSelfAttention 일반화)"""
    def __init__(self, emb_dim, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.query = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)          # (B,T,T) 유사도
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # 미래 차단 (causal)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v                                                # (B,T,head_size)


class MultiHeadAttention(nn.Module):
    """여러 head를 병렬로 돌리고 합친 뒤 projection. (Notebook 6)"""
    def __init__(self, emb_dim, num_heads, block_size, dropout):
        super().__init__()
        head_size = emb_dim // num_heads
        self.heads = nn.ModuleList(
            [Head(emb_dim, head_size, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


# --------------------------------------------------------------------------------------
# 3. Transformer block  (Notebook 6)
# --------------------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, emb_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.ReLU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """pre-norm transformer block: x = x + sa(ln(x)); x = x + ffwd(ln(x))"""
    def __init__(self, emb_dim, num_heads, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.sa = MultiHeadAttention(emb_dim, num_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.ffwd = FeedForward(emb_dim, dropout)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))     # residual + attention
        x = x + self.ffwd(self.ln2(x))   # residual + feedforward
        return x


# --------------------------------------------------------------------------------------
# 4. TinyGPT  (Notebook 6)
# --------------------------------------------------------------------------------------
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size, emb_dim=128, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(block_size, emb_dim)
        self.blocks = nn.Sequential(
            *[Block(emb_dim, num_heads, block_size, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device)
        tok = self.token_embedding(x)             # (B,T,C)
        pos = self.position_embedding(pos)[None]  # (1,T,C)
        h = tok + pos
        h = self.blocks(h)
        h = self.ln_f(h)
        logits = self.lm_head(h)                  # (B,T,V)
        return logits


# --------------------------------------------------------------------------------------
# 5. Loss / Train / Eval  (Notebook 1의 골격을 sequence용으로)
# --------------------------------------------------------------------------------------
def sequence_cross_entropy(logits, targets):
    # (B,T,V) -> (B,V,T) 로 바꿔 위치별 cross entropy
    return F.cross_entropy(logits.transpose(1, 2), targets)


def train_one_epoch(model, loader, optimizer, device, max_steps):
    model.train()
    total_loss, total_count = 0.0, 0
    for step, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = sequence_cross_entropy(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        total_count += xb.size(0)
        if step + 1 >= max_steps:
            break
    return total_loss / total_count


@torch.no_grad()
def evaluate(model, loader, device, max_steps):
    model.eval()
    total_loss, total_count = 0.0, 0
    for step, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)
        loss = sequence_cross_entropy(model(xb), yb)
        total_loss += loss.item() * xb.size(0)
        total_count += xb.size(0)
        if step + 1 >= max_steps:
            break
    return total_loss / total_count


# --------------------------------------------------------------------------------------
# 6. Sampling  (Notebook 6)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def generate(model, stoi, itos, device, start_text="\n", max_new_tokens=500):
    model.eval()
    block_size = model.block_size
    context = torch.zeros((1, block_size), dtype=torch.long, device=device)
    for ch in start_text:
        if ch in stoi:
            ix = torch.tensor([[stoi[ch]]], device=device)
            context = torch.cat([context[:, 1:], ix], dim=1)
    out = list(start_text)
    for _ in range(max_new_tokens):
        logits = model(context)[:, -1, :]        # 마지막 위치의 다음-토큰 분포
        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1)
        out.append(itos[ix.item()])
        context = torch.cat([context[:, 1:], ix], dim=1)
    return "".join(out)


# --------------------------------------------------------------------------------------
# 7. Main
# --------------------------------------------------------------------------------------
def main(cfg: Config):
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text, data, stoi, itos, vocab_size = load_data(cfg.dataset)
    n = int(0.9 * len(data))
    train_ds = NextTokenDataset(data[:n], cfg.block_size)
    val_ds = NextTokenDataset(data[n:], cfg.block_size)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=True)

    model = TinyGPT(vocab_size, cfg.block_size, cfg.emb_dim,
                    cfg.num_heads, cfg.num_layers, cfg.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    print(f"device={device}  dataset={cfg.dataset}  vocab_size={vocab_size}  "
          f"params={n_params/1e6:.2f}M")
    print(f"train chars={len(data[:n]):,}  val chars={len(data[n:]):,}\n")

    for epoch in range(cfg.epochs):
        t0 = time.time()
        tl = train_one_epoch(model, train_loader, optimizer, device, cfg.steps_per_epoch)
        vl = evaluate(model, val_loader, device, cfg.eval_steps)
        print(f"epoch {epoch:2d} | train {tl:.4f} | val {vl:.4f} | {time.time()-t0:.1f}s")

    print("\n----- sample -----")
    print(generate(model, stoi, itos, device, start_text="\n", max_new_tokens=600))
    return model


if __name__ == "__main__":
    main(Config())
