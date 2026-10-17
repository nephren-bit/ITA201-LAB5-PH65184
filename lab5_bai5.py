import math


# Tu dien V: tap tat ca tu duy nhat trong tap huan luyen
def extract_vocabulary(emails):
    vocab = set()
    for words, _ in emails:
        vocab.update(words)
    return vocab


# Naive Bayes da thuc + Laplace Smoothing
def train_naive_bayes(train_emails, alpha=1.0):
    vocab = extract_vocabulary(train_emails)
    n_total = len(train_emails)

    model = {'vocab': vocab, 'prior': {}, 'likelihood': {}}
    for c in (1, 0):
        docs_c = [words for words, y in train_emails if y == c]
        model['prior'][c] = len(docs_c) / n_total

        # Dem so lan xuat hien cua tung tu trong lop c
        word_counts = {w: 0 for w in vocab}
        for words in docs_c:
            for w in words:
                word_counts[w] += 1
        total_words_c = sum(word_counts.values())

        # P(w|c) = (count(w, c) + alpha) / (tong so tu cua lop c + alpha * |V|)
        model['likelihood'][c] = {
            w: (word_counts[w] + alpha) / (total_words_c + alpha * len(vocab))
            for w in vocab
        }
    return model


# Du doan bang Log-Likelihood: log P(Y) + sum log P(w_i|Y); bo qua tu ngoai tu dien
def predict_email(email_words, model):
    log_scores = {}
    for c in (1, 0):
        score = math.log(model['prior'][c])
        for w in email_words:
            if w in model['vocab']:
                score += math.log(model['likelihood'][c][w])
        log_scores[c] = score
    return max(log_scores, key=log_scores.get), log_scores


# Ma tran nham lan cho bai toan nhi phan (1 = Spam la lop duong)
def confusion_matrix(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    return tp, tn, fp, fn


if __name__ == "__main__":
    train_emails = [
        (["free", "money", "offer"], 1),
        (["free", "ticket", "win"], 1),
        (["money", "bonus", "free"], 1),
        (["meeting", "project", "deadline"], 0),
        (["project", "report", "review"], 0),
        (["schedule", "meeting", "room"], 0)
    ]

    vocab = extract_vocabulary(train_emails)
    print(f"Tu dien V ({len(vocab)} tu): {sorted(vocab)}")

    model = train_naive_bayes(train_emails)
    print(f"P(Spam) = {model['prior'][1]:.4f}, P(Ham) = {model['prior'][0]:.4f}")
    print(f"{'Tu':<10}{'P(w|Spam)':>12}{'P(w|Ham)':>12}")
    for w in sorted(vocab):
        print(f"{w:<10}{model['likelihood'][1][w]:>12.4f}{model['likelihood'][0][w]:>12.4f}")

    test_email = ['free', 'project', 'bonus']
    label, log_scores = predict_email(test_email, model)
    print(f"\nEmail moi {test_email}:")
    print(f"  log Score(Spam) = {log_scores[1]:.4f}, log Score(Ham) = {log_scores[0]:.4f}")
    print(f"  -> Du doan: {label} ({'Spam' if label == 1 else 'Ham'})")

    # Danh gia tren chinh tap huan luyen bang ma tran nham lan
    y_true = [y for _, y in train_emails]
    y_pred = [predict_email(words, model)[0] for words, _ in train_emails]
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    print("\nMa tran nham lan (tap huan luyen):")
    print("              Pred Spam  Pred Ham")
    print(f"  True Spam   {tp:^9}  {fn:^8}")
    print(f"  True Ham    {fp:^9}  {tn:^8}")
    print(f"  Accuracy = {(tp + tn) / len(y_true):.2f}")

# Vi sao dung Log-Likelihood thay cho tich xac suat?
# - Moi P(w_i|Y) la so rat nho (vd 0.001). Van ban dai 1000 tu -> tich ~ 10^-3000, nho hon
#   so float64 duong nho nhat (~10^-308) nen bi lam tron ve 0.0 (Underflow). Khi do moi lop
#   deu co Score = 0 va mo hinh khong phan biet duoc nua.
# - Ham log don dieu tang nen argmax cua tich = argmax cua tong log -> nhan du doan khong doi.
# - log(a * b) = log(a) + log(b): bien phep nhan nhieu so nho thanh phep cong cac so am vua
#   phai (vd 1000 * log(0.001) = -6907.8), hoan toan nam trong mien bieu dien cua float.
# - Phep cong cung on dinh so hoc hon va nhanh hon phep nhan lien tiep.
