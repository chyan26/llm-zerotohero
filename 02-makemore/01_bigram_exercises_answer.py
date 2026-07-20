"""
makemore — Exercises E01–E06
Companion code for "The spelled-out intro to language modeling: building makemore"
(Andrej Karpathy, Zero to Hero series)

Dataset: names.txt (32,033 US names), same as used in the video.
Run: python3 exercises.py
"""

import random
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Setup: vocabulary + 80/10/10 train/dev/test split (E02)
# ---------------------------------------------------------------------------
words = open('./names.txt', 'r').read().splitlines()
chars = sorted(list(set(''.join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}
V = len(stoi)  # 27

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
train_words, dev_words, test_words = words[:n1], words[n1:n2], words[n2:]
print(f"split sizes -> train: {len(train_words)}, dev: {len(dev_words)}, test: {len(test_words)}\n")


# ---------------------------------------------------------------------------
# E01 + E02: Trigram counting model, evaluated on train/dev/test
# ---------------------------------------------------------------------------
def build_trigram_counts(words_list):
    T = torch.zeros((V * V, V), dtype=torch.int32)
    for w in words_list:
        chs = ['.', '.'] + list(w) + ['.']
        for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
            ix1, ix2, ix3 = stoi[ch1], stoi[ch2], stoi[ch3]
            T[ix1 * V + ix2, ix3] += 1
    return T


def build_bigram_counts(words_list):
    N = torch.zeros((V, V), dtype=torch.int32)
    for w in words_list:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            N[stoi[ch1], stoi[ch2]] += 1
    return N


def bigram_loss(words_list, N, smoothing):
    P = (N + smoothing).float()
    P /= P.sum(1, keepdim=True)
    log_likelihood, n = 0.0, 0
    for w in words_list:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            log_likelihood += torch.log(P[stoi[ch1], stoi[ch2]])
            n += 1
    return (-log_likelihood / n).item()


def trigram_loss(words_list, T, smoothing):
    P = (T + smoothing).float()
    P /= P.sum(1, keepdim=True)
    log_likelihood, n = 0.0, 0
    for w in words_list:
        chs = ['.', '.'] + list(w) + ['.']
        for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
            ix1, ix2, ix3 = stoi[ch1], stoi[ch2], stoi[ch3]
            log_likelihood += torch.log(P[ix1 * V + ix2, ix3])
            n += 1
    return (-log_likelihood / n).item()


print("=" * 70)
print("E01: Trigram counting model vs. bigram baseline")
print("=" * 70)
N_bigram = build_bigram_counts(train_words)
T_trigram = build_trigram_counts(train_words)

print("Bigram  (smoothing=1) -> train: %.4f  dev: %.4f  test: %.4f" % (
    bigram_loss(train_words, N_bigram, 1),
    bigram_loss(dev_words, N_bigram, 1),
    bigram_loss(test_words, N_bigram, 1),
))
print("Trigram (smoothing=1) -> train: %.4f  dev: %.4f  test: %.4f" % (
    trigram_loss(train_words, T_trigram, 1),
    trigram_loss(dev_words, T_trigram, 1),
    trigram_loss(test_words, T_trigram, 1),
))
print("-> Trigram improves over bigram: more context = better prediction.\n")


# ---------------------------------------------------------------------------
# E03: Tune smoothing strength for the trigram model using the DEV set
# ---------------------------------------------------------------------------
print("=" * 70)
print("E03: Tuning trigram smoothing strength on the dev set")
print("=" * 70)
candidates = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
best_s, best_dev = None, float('inf')
for s in candidates:
    tr = trigram_loss(train_words, T_trigram, s)
    dv = trigram_loss(dev_words, T_trigram, s)
    marker = ""
    if dv < best_dev:
        best_dev, best_s = dv, s
        marker = "  <- best so far"
    print(f"  smoothing={s:>7}: train={tr:.4f}  dev={dv:.4f}{marker}")

test_at_best = trigram_loss(test_words, T_trigram, best_s)
print(f"\nBest smoothing (chosen on dev) = {best_s} -> TEST loss = {test_at_best:.4f}")
print("""
Pattern observed:
  - Low smoothing -> train loss keeps dropping (overfits rare/unseen trigrams,
    which get near-zero probability and blow up dev/test loss).
  - High smoothing -> both train and dev loss rise together (model washed out
    toward uniform distribution, underfitting).
  - There's a sweet spot in between where dev loss is minimized -- classic
    bias/variance trade-off, same idea as L2 regularization strength.
""")


# ---------------------------------------------------------------------------
# E04 + E05: Neural net bigram model
#   E04 - index rows of W directly instead of F.one_hot(x) @ W
#   E05 - use F.cross_entropy instead of manual softmax + NLL
# ---------------------------------------------------------------------------
print("=" * 70)
print("E04 + E05: Neural net bigram -- indexing instead of one_hot,")
print("           F.cross_entropy instead of manual softmax+NLL")
print("=" * 70)


def build_bigram_xy(words_list):
    xs, ys = [], []
    for w in words_list:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            xs.append(stoi[ch1]); ys.append(stoi[ch2])
    return torch.tensor(xs), torch.tensor(ys)


xs, ys = build_bigram_xy(train_words)
xs_dev, ys_dev = build_bigram_xy(dev_words)

g = torch.Generator().manual_seed(2147483647)
W = torch.randn((V, V), generator=g, requires_grad=True)

for k in range(100):
    logits = W[xs]                                    # E04: direct row-indexing, no one_hot
    loss = F.cross_entropy(logits, ys) + 0.01 * (W ** 2).mean()  # E05: F.cross_entropy
    W.grad = None
    loss.backward()
    W.data += -50 * W.grad

with torch.no_grad():
    dev_loss = F.cross_entropy(W[xs_dev], ys_dev)
print(f"final train loss: {loss.item():.4f}   dev loss: {dev_loss.item():.4f}")
print("""
Why F.cross_entropy over manual softmax + NLL (E05):
  1. Numerical stability -- it subtracts the max logit internally (the
     "logit trick") before exponentiating, avoiding exp() overflow that a
     naive softmax can hit with large logits.
  2. Speed & memory -- it fuses softmax + log + NLL into one kernel instead
     of materializing several intermediate tensors.
  3. Fewer bugs -- no risk of forgetting keepdim, the wrong reduction axis,
     or numerically-unsafe log(0).
""")


# ---------------------------------------------------------------------------
# E06: meta-exercise -- a "fun" one.
#   Neural trigram with a small learned character EMBEDDING table instead of
#   a full 729x27 lookup. This previews the next video (MLP / embeddings) and
#   shows the parameter-count vs. expressiveness trade-off directly.
# ---------------------------------------------------------------------------
print("=" * 70)
print("E06 (fun): embedding-based neural trigram (previews the MLP video)")
print("=" * 70)


def build_trigram_xy(words_list):
    x1s, x2s, ys = [], [], []
    for w in words_list:
        chs = ['.', '.'] + list(w) + ['.']
        for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
            x1s.append(stoi[ch1]); x2s.append(stoi[ch2]); ys.append(stoi[ch3])
    return torch.tensor(x1s), torch.tensor(x2s), torch.tensor(ys)


x1, x2, y = build_trigram_xy(train_words)
x1d, x2d, yd = build_trigram_xy(dev_words)

g = torch.Generator().manual_seed(2147483647)
d = 10  # embedding dimension per character
C = (torch.randn((V, d), generator=g) * 0.1).requires_grad_()
Wt = (torch.randn((2 * d, V), generator=g) * 0.1).requires_grad_()
bt = torch.zeros(V, requires_grad=True)
params = [C, Wt, bt]

for k in range(2000):
    emb = torch.cat([C[x1], C[x2]], dim=1)          # (N, 2d) -- concat two learned embeddings
    logits = emb @ Wt + bt
    loss = F.cross_entropy(logits, y)
    for p in params:
        p.grad = None
    loss.backward()
    lr = 2.0 if k < 1000 else 0.5
    for p in params:
        p.data += -lr * p.grad

with torch.no_grad():
    emb_dev = torch.cat([C[x1d], C[x2d]], dim=1)
    dev_loss_emb = F.cross_entropy(emb_dev @ Wt + bt, yd)

n_params_emb = sum(p.nelement() for p in params)
n_params_table = V * V * V
print(f"embedding-trigram  -> train: {loss.item():.4f}  dev: {dev_loss_emb.item():.4f}")
print(f"counting-trigram   -> dev:   {trigram_loss(dev_words, T_trigram, best_s):.4f}  (from E03, best smoothing)")
print(f"params: embedding model = {n_params_emb}   vs.   full lookup table = {n_params_table}")
print("""
Take-away: with only ~4% of the parameters, the embedding+linear trigram
gets CLOSE to but does not beat the full count-based trigram table. A linear
model on top of concatenated embeddings can only represent a limited
(low-rank / bilinear) family of interactions between the two context
characters. Closing that gap needs a NON-LINEARITY (a hidden layer) between
the embeddings and the output -- exactly the MLP architecture in Karpathy's
next video.
""")
