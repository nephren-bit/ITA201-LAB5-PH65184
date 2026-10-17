# Naive Bayes voi dac trung nhi phan: Score(Y) = P(Y) * P(x1|Y) * P(x2|Y)
def naive_bayes_predict(dataset, x_new):
    n_samples = len(dataset)
    n_features = len(x_new)
    classes = sorted(set(y for _, y in dataset))

    scores = {}
    for c in classes:
        # 1. Xac suat tien nghiem P(Y=c)
        samples_c = [x for x, y in dataset if y == c]
        prior = len(samples_c) / n_samples

        # 2. Likelihood P(x_j = x_new[j] | Y=c), gia dinh cac dac trung doc lap
        likelihood = 1.0
        for j in range(n_features):
            match = sum(1 for x in samples_c if x[j] == x_new[j])
            likelihood *= match / len(samples_c)

        # 3. Diem hau nghiem (khong chia P(X) vi giong nhau cho moi lop)
        scores[c] = prior * likelihood

    prediction = max(scores, key=scores.get)
    return prediction, scores


if __name__ == "__main__":
    # x1: chua 'Khuyen mai', x2: chua 'Trung thuong'; Y: 1 = Spam, 0 = Thuong
    dataset = [
        ([1, 1], 1),
        ([1, 0], 1),
        ([1, 1], 1),
        ([0, 1], 0),
        ([0, 0], 0),
        ([0, 0], 0)
    ]

    x_new = [1, 0]
    label, scores = naive_bayes_predict(dataset, x_new)
    print(f"Score(Y=1) = {scores[1]:.4f}, Score(Y=0) = {scores[0]:.4f}")
    print(f"Du doan cho x_new = {x_new}: {label} ({'Spam' if label == 1 else 'Thuong'})")  # 1 (Spam)
