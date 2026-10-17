import math


# Shannon Entropy: H(D) = - sum(p_i * log2(p_i)), don vi bit
def compute_entropy(labels):
    if not labels:
        return 0.0

    total_samples = len(labels)

    # 1. Dem tan suat nhan
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1

    # 2. Tinh Entropy
    entropy = 0.0
    for count in label_counts.values():
        p = count / total_samples
        if p > 0:
            entropy -= p * math.log2(p)

    return round(entropy, 4)


if __name__ == "__main__":
    labels_mixed = ['Yes', 'Yes', 'No', 'Yes', 'No']  # 3 Yes, 2 No
    labels_pure = ['Yes', 'Yes', 'Yes']                # Tinh khiet

    print(f"Entropy tap hon loan (3 Yes, 2 No): {compute_entropy(labels_mixed)} bits")
    print(f"Entropy tap tinh khiet (100% Yes): {compute_entropy(labels_pure)} bits")
    # Ket qua ky vong: 0.971 bits va 0.0 bits
