import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from wordcloud import WordCloud


def main():

    data = [
        "billing amount is incorrect",
        "charged twice for my bill",
        "network outage in my area",
        "no mobile network signal",
        "internet speed is very slow",
        "mobile data is not working",
        "SIM card is not activating",
        "new SIM card is not working",
        "recharge payment failed",
        "recharge balance is missing",
        "phone cannot connect to network",
        "unexpected charge on bill"
    ]

    df = pd.DataFrame({"ticket": data})

    # ---------------- LSA ----------------
    tfidf = TfidfVectorizer(stop_words="english")
    X = tfidf.fit_transform(df["ticket"])

    lsa = TruncatedSVD(n_components=3, random_state=42)
    lsa.fit(X)

    print("\nLSA Topics:")
    words = tfidf.get_feature_names_out()

    for i, topic in enumerate(lsa.components_):
        print(
            "Topic", i + 1,
            [words[j] for j in topic.argsort()[-5:]]
        )

    # ---------------- LDA ----------------
    texts = [x.lower().split() for x in data]

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=3,
        random_state=42,
        passes=10
    )

    print("\nLDA Topics:")

    for topic in lda.print_topics():
        print(topic)

    # ---------------- Coherence ----------------
    coherence = CoherenceModel(
        model=lda,
        texts=texts,
        dictionary=dictionary,
        coherence="c_v",
        processes=1
    ).get_coherence()

    print("\nCoherence Score:", round(coherence, 3))

    # ---------------- Word Clouds ----------------
    for i in range(3):

        words = dict(lda.show_topic(i, 10))

        wc = WordCloud(
            width=700,
            height=400,
            background_color="white"
        ).generate_from_frequencies(words)

        plt.figure(figsize=(8, 4))
        plt.imshow(wc)
        plt.axis("off")
        plt.title("Topic " + str(i + 1))
        plt.show()


if __name__ == "__main__":
    main()