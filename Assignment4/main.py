import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# 1. E-commerce corpus
data = [
    ("Samsung Smartphone", "premium smartphone powerful processor camera amoled display"),
    ("iPhone", "advanced smartphone powerful processor excellent camera display performance"),
    ("HP Laptop", "business laptop powerful processor large memory ssd storage display"),
    ("MacBook", "premium laptop powerful processor retina display fast storage performance"),
    ("Sony Headphones", "wireless headphones noise cancellation deep bass premium sound"),
    ("Boat Headphones", "wireless headphones powerful bass comfortable bluetooth connectivity"),
    ("JBL Headphones", "bluetooth headphones deep bass excellent sound wireless connectivity"),
    ("Canon Camera", "digital camera high resolution sensor professional photography optical zoom"),
    ("Nikon Camera", "professional camera high resolution sensor optical zoom photography"),
    ("Samsung Smartwatch", "smartwatch fitness tracking health monitoring display wireless connectivity"),
    ("iPad Tablet", "tablet high resolution display powerful processor long battery life"),
    ("Samsung Tablet", "android tablet large display powerful processor long battery life")
]

# 2. Train Word2Vec
sentences = [desc.lower().split() for _, desc in data]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1,
    epochs=100
)

print("Word2Vec model trained successfully!")

# 3. Similar words
print("\nSimilar to smartphone:")
for word, score in model.wv.most_similar("smartphone", topn=5):
    print(word, round(score, 3))


# 4. Product recommendation
def product_vector(text):
    words = [w for w in text.lower().split() if w in model.wv]
    return np.mean([model.wv[w] for w in words], axis=0)

product_vectors = np.array([product_vector(desc) for _, desc in data])

def recommend(query, n=5):
    q = product_vector(query)
    sim = cosine_similarity([q], product_vectors)[0]
    for i in np.argsort(sim)[::-1][:n]:
        print(data[i][0], "->", round(sim[i], 3))

print("\nRecommendations for 'smartphone camera':")
recommend("smartphone camera")

print("\nRecommendations for 'wireless headphones bass':")
recommend("wireless headphones bass")

words = ["smartphone", "processor", "camera", "laptop",
         "headphones", "wireless", "display", "battery",
         "tablet", "smartwatch", "bass", "storage"]

words = [w for w in words if w in model.wv]
vectors = np.array([model.wv[w] for w in words])

def plot_embeddings(method, title):
    result = method.fit_transform(vectors)
    plt.figure(figsize=(9, 6))
    plt.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(words):
        plt.annotate(word, result[i])
    plt.title(title)
    plt.grid()
    plt.show()

plot_embeddings(PCA(n_components=2), "Word2Vec - PCA")
plot_embeddings(TSNE(n_components=2, perplexity=5, random_state=42),
                "Word2Vec - t-SNE")