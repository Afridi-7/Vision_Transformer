# Vision Transformer vs ResNet-18 — CIFAR-10

A from-scratch implementation and comparison of Vision Transformer (ViT) and ResNet-18 on CIFAR-10, trained with identical augmentation and training recipes for a fair head-to-head evaluation.

---

## Results

| Model | Accuracy | Parameters | Training Time | Epochs |
|---|---|---|---|---|
| **ResNet-18** | **95.67%** | 11.2M | ~100 min | 100 |
| ViT (DeiT-Tiny) | 89.11% | 5.7M | ~249 min | 200 |

Both models trained from scratch — no pretrained weights used.

---

## Key Findings

**ResNet-18 wins on CIFAR-10** by 6.56 percentage points while being half the size and training 2.5× faster. This is expected and well-understood: CNNs have translation equivariance and locality built into their architecture, giving them a structural advantage on small datasets. ViT has to learn these spatial priors from data alone, which requires significantly more samples than CIFAR-10's 50K images.

**ViT is more parameter-efficient** — 89.11% with only 5.7M parameters vs ResNet-18's 11.2M for 95.67%. At scale (100M+ training samples), this efficiency advantage flips the comparison in ViT's favour.

**The hardest class for both models is cat** — ResNet-18 scores 89.5%, ViT scores 75.8%. The cat↔dog confusion is visible in the ViT confusion matrix (109 cats misclassified as dog, 108 dogs misclassified as cat), reflecting genuine visual similarity that both architectures struggle with.

---

## Architecture

### Vision Transformer (ViT)
Inspired by DeiT-Tiny (Touvron et al., 2020), adapted for 32×32 CIFAR images.

```
Image (32×32×3)
  → PatchEmbeddings   — Conv2d(patch_size=4) → 64 patches
  → + [CLS] token + learnable position embeddings
  → Encoder (12 × Block)
       Pre-LayerNorm → Multi-Head Attention (3 heads, 64 dim/head)
       Pre-LayerNorm → MLP (192 → 768 → 192, GELU)
       Stochastic Depth (DropPath, rate 0→0.1)
  → [CLS] output → Linear(192, 10)
```

| Hyperparameter | Value |
|---|---|
| Hidden size | 192 |
| Layers | 12 |
| Heads | 3 |
| FFN dim | 768 |
| Patch size | 4×4 |
| Drop path rate | 0.1 |

### ResNet-18
Standard ResNet-18 adapted for 32×32 images. Two modifications from the original ImageNet architecture:

- First conv: `7×7 stride 2` → `3×3 stride 1` (preserves spatial resolution)
- MaxPool after first conv: removed (replaced with Identity)

Without these changes the 32×32 input is downsampled to 8×8 before any residual learning begins.

---

## Training Recipe

Both models use an identical training setup for a fair comparison.

| Setting | Value |
|---|---|
| Optimizer | AdamW (β₁=0.9, β₂=0.999) |
| Learning rate | 1e-3 with linear warmup (5 ep) → cosine decay |
| Weight decay | 0.05 |
| Batch size | 128 |
| Label smoothing | 0.1 |
| Gradient clipping | 1.0 |
| Early stopping | Patience 20 |

### Augmentation (identical for both)
```python
RandomCrop(32, padding=4)
RandomHorizontalFlip()
RandAugment(num_ops=2, magnitude=9)
RandomErasing(p=0.25)
MixUp(alpha=0.4)      # 50% probability per batch
CutMix(alpha=1.0)     # 50% probability per batch
```

> Note: Train accuracy appears lower than test accuracy (~55% vs 89% for ViT) because MixUp/CutMix blend images and score accuracy against the original hard labels. This is expected behaviour, not a bug.

---

## Visualisations

### ResNet-18
**Learning curves** — smooth convergence to 95.67% with no overfitting.

**Per-class accuracy** — all classes above 89%, automobile and ship above 98%.

### ViT
**Training curves** — 200-epoch run showing slow but steady improvement. Classic MixUp signature: low train accuracy, high test accuracy.

**Per-class accuracy** — frog (93.2%), horse (92.7%), ship (93.8%) are strongest. Cat (75.8%) and dog (82.5%) are hardest.

**Confusion matrix** — strong diagonal with visible cat↔dog confusion, reflecting genuine visual similarity between these classes.

### Comparison
Side-by-side bar charts comparing accuracy, model size, and training time across both models.

---

## Repository Structure

```
Vision_Transformer/
├── ResNet.ipynb          # ResNet-18 training + evaluation
├── vit_90_colab.py       # ViT training + evaluation
└── README.md
```

---

## How to Run

Both notebooks run on a free Google Colab T4 GPU.

**ResNet-18** (~100 min):
Open `ResNet.ipynb` in Colab, set runtime to T4 GPU, run all cells.

**ViT** (~250 min):
Open `vit_90_colab.py` in Colab, set runtime to T4 GPU, run all cells.

CIFAR-10 downloads automatically (~170MB) on first run.

---

## References

- Dosovitskiy et al. (2020) — [An Image is Worth 16×16 Words](https://arxiv.org/abs/2010.11929)
- Touvron et al. (2020) — [Training data-efficient image transformers (DeiT)](https://arxiv.org/abs/2012.12877)
- He et al. (2015) — [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- Huang et al. (2016) — [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382)
- Yun et al. (2019) — [CutMix](https://arxiv.org/abs/1905.04899)
- Zhang et al. (2017) — [MixUp](https://arxiv.org/abs/1710.09412)
- tintn/vision-transformer-from-scratch — base ViT reference implementation

---

## License

MIT
