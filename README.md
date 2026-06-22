# TinyGPT — Character-level Language Model

`bigram → MLP → sequence model → self-attention → GPT`로 이어지는 6개 노트북의 흐름을
하나의 character-level language model로 구현한 과제 결과물입니다.
[karpathy/nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) /
[karpathy/makemore](https://github.com/karpathy/makemore)를 기반으로 합니다.

모델은 token/position embedding + masked multi-head self-attention + feedforward 블록을
쌓은 GPT 구조이며, **데이터셋에 독립적**이라 어떤 텍스트 코퍼스든 학습할 수 있습니다.

---

## 실행 방법

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
python tiny_gpt.py
```

`tiny_gpt.py` 하나가 데이터 다운로드 → 학습 → 샘플 생성까지 모두 수행합니다.
실행하면 epoch별 train/val loss가 출력되고, 마지막에 학습된 모델이 생성한 텍스트가 나옵니다
(CPU 기준 약 3~4분).

### 데이터셋 바꾸기

`load_data`는 **프리셋 이름 / 로컬 파일 / URL**을 모두 받습니다:

```python
from tiny_gpt import Config, main
main(Config(dataset="sherlock"))            # 프리셋: sherlock / shakespeare / names
main(Config(dataset="my_corpus.txt"))       # 로컬 텍스트 파일
main(Config(dataset="https://.../any.txt")) # URL
```

### GitHub Codespaces

`.devcontainer/devcontainer.json`이 포함되어 있어, Codespace를 생성하면 CPU용 PyTorch가
자동 설치됩니다. 빌드가 끝나면 터미널에서 `python tiny_gpt.py`만 실행하면 됩니다.

---

## 결과

> **과제 요구사항**: tiny Shakespeare 이외의 데이터셋으로 훈련하고 결과를 제시.

기본 데이터셋은 tiny Shakespeare 대신
[`big.txt`](https://raw.githubusercontent.com/dscape/spell/master/test/resources/big.txt)
(Sherlock Holmes를 포함한 Project Gutenberg 공개 도메인 영어 산문, 약 648만 글자, vocab 93)
입니다. CPU에서 8 epoch 학습한 loss는 다음과 같습니다.

| epoch | train | val |
|------:|------:|----:|
| 0 | 2.78 | 2.53 |
| 2 | 2.23 | 2.31 |
| 4 | 2.08 | 2.25 |
| 6 | 2.00 | 2.17 |
| 7 | 1.97 | 2.16 |

무작위 출발(loss ≈ 4.2)에서 안정적으로 수렴했고, 생성 샘플은 **따옴표 대화·문단 단위의
산문체**를 학습한 모습입니다:

```
Regiven the in a velve. The morre. GHenown inexawnch of secubled b the
reation conlaffeded Bock it dussones, but the prose hnopersting the
paboylocination.

"Now of accricentsher, on the ceccosiase oh, himst carked add the
tapplatix has ding tupnessed this complere to sadd tnerer is oo blong his was
fecleends all becomentiles."
```

비교로, 같은 모델을 tiny Shakespeare로 학습하면 loss는 train 2.71 → 1.88, val 2.40 → 1.92이고
출력은 희곡 대본의 `화자 이름(대문자+콜론) + 대사` 구조를 학습합니다. **모델·하이퍼파라미터가
같아도 데이터셋만 바꾸면 출력의 장르·문체가 완전히 달라진다**는 점이 핵심입니다.

> 0.6M 파라미터를 CPU에서 약 2,400 step만 학습한 결과라 완전한 단어는 아닙니다.
> GPU에서 더 키우면(아래 *Scaling up*) 훨씬 또렷해집니다.

---

## 구조와 노트북 매핑

전체 코드는 `Dataset → DataLoader → Model → Loss → Train → Sample` 골격을 따릅니다.

| 구성 요소 | 역할 | 출처 |
|---|---|---|
| `load_data` | 프리셋/파일/URL → character 인코딩 | (일반화) |
| `NextTokenDataset` | `x=[t1..tT]`, `y=[t2..t(T+1)]`로 다음 토큰 예측 | Notebook 4 |
| token + position embedding | 토큰 의미 + 위치 정보 | Notebook 4 |
| `Head` | masked self-attention 1개 (causal) | Notebook 5 |
| `MultiHeadAttention` | head 여러 개 병렬 + projection | Notebook 6 |
| `FeedForward`, `Block` | residual + LayerNorm 트랜스포머 블록 | Notebook 6 |
| `TinyGPT` | 블록 N개 + 최종 LayerNorm + lm_head | Notebook 6 |
| train / sample 루프 | epoch 학습, autoregressive 생성 | Notebook 1부터 재사용 |

기본 하이퍼파라미터: `block_size=32`, `emb_dim=128`, `num_heads=4`,
`num_layers=3`, `dropout=0.1`, `lr=3e-4`, AdamW.

---

## Scaling up (GPU)

`tiny_gpt.py`의 `Config`를 교수님 원본 수준으로 키우면 결과가 크게 좋아집니다 (Colab GPU 권장):

```python
block_size=64, emb_dim=128, num_heads=4, num_layers=4,
dropout=0.1, lr=3e-4, epochs=100, steps_per_epoch=300
```