import math


# Binary Cross-Entropy: Loss = -(y*log(p) + (1-y)*log(1-p)), kep p trong [eps, 1-eps]
def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    y_pred = min(max(y_pred, eps), 1 - eps)
    return -(y_true * math.log(y_pred) + (1 - y_true) * math.log(1 - y_pred))


# Laplace Smoothing: P = (count_word + alpha) / (count_class + alpha * |V|)
def laplace_smoothing(count_word, count_class, vocab_size, alpha=1.0):
    return (count_word + alpha) / (count_class + alpha * vocab_size)


if __name__ == "__main__":
    print(f"Loss khi du doan chuan (y=1, p=0.9): {binary_cross_entropy(1, 0.9):.4f}")  # ~0.1054
    print(f"Loss khi du doan sai (y=1, p=0.1): {binary_cross_entropy(1, 0.1):.4f}")    # ~2.3026
    print(f"Loss khi p = 0 tuyet doi (y=1, p=0.0): {binary_cross_entropy(1, 0.0):.4f} (khong loi log(0))")

    p_smoothed = laplace_smoothing(count_word=0, count_class=100, vocab_size=500)
    print(f"Xac suat sau lam muot: {p_smoothed:.4f}")  # ~0.0017 (khac 0)
