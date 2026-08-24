import re, math
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Documents
docs = [
    "Machine learning is a powerful technology for data analysis",
    "Machine learning is a powerful technology used for data analysis",
    "Deep learning uses neural networks for image classification",
    "Natural language processing helps computers understand human language"
]

names = ["Assignment 1", "Assignment 2", "Research Paper 1", "Research Paper 2"]

# Tokenization
tokenize = lambda x: re.findall(r'\b\w+\b', x.lower())
tokens = [tokenize(d) for d in docs]

# ---------------- BoW FROM SCRATCH ----------------
vocab = sorted(set(w for d in tokens for w in d))

bow = np.array([
    [Counter(d).get(w, 0) for w in vocab]
    for d in tokens
])

print("BoW Vocabulary:\n", vocab)
print("\nBoW Matrix:\n", bow)

cv = CountVectorizer()
sk_bow = cv.fit_transform(docs).toarray()

print("\nSklearn BoW:\n", sk_bow)

N = len(docs)

idf = {
    w: math.log(N / sum(w in d for d in tokens))
    for w in vocab
}

tfidf = np.array([
    [
        (Counter(d).get(w, 0) / len(d)) * idf[w]
        for w in vocab
    ]
    for d in tokens
])

np.set_printoptions(precision=4, suppress=True)

print("\nTF-IDF From Scratch:\n", tfidf)

tv = TfidfVectorizer(norm=None, smooth_idf=False)
sk_tfidf = tv.fit_transform(docs).toarray()

print("\nSklearn TF-IDF:\n", sk_tfidf)

sim = cosine_similarity(tfidf)

print("\nCosine Similarity:\n", np.round(sim, 4))

print("\nPlagiarism Detection:")

for i in range(len(docs)):
    for j in range(i + 1, len(docs)):
        score = sim[i, j]
        result = "PLAGIARISM DETECTED" if score > 0.85 else "No plagiarism"
        print(f"{names[i]} vs {names[j]}: {score:.4f} -> {result}")

N = 5

print("\nTop Keywords:")

for i, name in enumerate(names):
    top = np.argsort(tfidf[i])[-N:][::-1]
    print(name, [(vocab[j], round(tfidf[i, j], 4)) for j in top])

i = 0
top = np.argsort(tfidf[i])[-N:][::-1]

plt.bar([vocab[j] for j in top], tfidf[i, top])
plt.xlabel("Keywords")
plt.ylabel("TF-IDF Score")
plt.title(f"Top {N} Keywords - {names[i]}")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()