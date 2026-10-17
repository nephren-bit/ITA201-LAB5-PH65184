import math


def compute_entropy(labels):
    if not labels:
        return 0.0
    total_samples = len(labels)
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    entropy = 0.0
    for count in label_counts.values():
        p = count / total_samples
        if p > 0:
            entropy -= p * math.log2(p)
    return round(entropy, 4)


# Information Gain: IG(D, A) = H(D) - H(D|A)
def compute_information_gain(parent_labels, subsets_labels):
    # 1. Entropy tap cha
    h_parent = compute_entropy(parent_labels)
    total_len = len(parent_labels)

    # 2. Entropy co dieu kien sau chia (trung binh co trong so)
    h_children = 0.0
    for sub in subsets_labels:
        if len(sub) > 0:
            weight = len(sub) / total_len
            h_children += weight * compute_entropy(sub)

    # 3. Information Gain
    ig = h_parent - h_children
    return round(ig, 4)


if __name__ == "__main__":
    # Chia [Yes, Yes, No, No] thanh 2 tap tinh khiet
    parent = ['Yes', 'Yes', 'No', 'No']
    sub1 = ['Yes', 'Yes']
    sub2 = ['No', 'No']
    print(f"Information Gain (chia tinh khiet): {compute_information_gain(parent, [sub1, sub2])} bits")  # 1.0

    # Chia khong mang thong tin: moi tap con van 50/50
    print(f"Information Gain (chia vo ich): "
          f"{compute_information_gain(parent, [['Yes', 'No'], ['Yes', 'No']])} bits")  # 0.0
