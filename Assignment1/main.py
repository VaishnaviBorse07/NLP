import pandas as pd
import re
import emoji
import nltk
import spacy

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk import pos_tag
from transformers import BertTokenizer

# -----------------------
# Download NLTK Resources
# -----------------------
resources = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng"
]

for resource in resources:
    try:
        nltk.download(resource, quiet=True)
    except:
        pass

# -----------------------
# Load spaCy Model
# -----------------------
nlp = spacy.load("en_core_web_sm")

# -----------------------
# Load BERT Tokenizer
# -----------------------
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# -----------------------
# Load Dataset
# -----------------------
df = pd.read_csv("Articles.csv", encoding="latin1")

# Process only first 300 articles
df = df.head(300)

# -----------------------
# Clean Missing Values
# -----------------------
df["Article"] = df["Article"].fillna("")
df["Heading"] = df["Heading"].fillna("")

# Combine heading + article
df["Text"] = df["Heading"] + " " + df["Article"]

# -----------------------
# Cleaning Function
# -----------------------
def clean_text(text):
    text = str(text)

    text = text.lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)

    text = emoji.replace_emoji(text, replace="")

    text = re.sub(r"[^\w\s]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text

# -----------------------
# Tokenizers
# -----------------------
def whitespace_tokenizer(text):
    return text.split()

def nltk_tokenizer(text):
    return word_tokenize(text)

def bert_subword(text):
    # Limit to avoid long sequence warning
    return bert_tokenizer.tokenize(text)[:512]

def spacy_tokenizer(text):
    return [token.text for token in nlp(text)]

# -----------------------
# Stopword Removal
# -----------------------
stop_words = set(stopwords.words("english"))

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

# -----------------------
# Porter Stemmer
# -----------------------
porter = PorterStemmer()

def porter_stemming(tokens):
    return [porter.stem(word) for word in tokens]

# -----------------------
# Snowball Stemmer
# -----------------------
snowball = SnowballStemmer("english")

def snowball_stemming(tokens):
    return [snowball.stem(word) for word in tokens]

# -----------------------
# Lemmatization
# -----------------------
lemmatizer = WordNetLemmatizer()

def lemmatization(tokens):
    return [lemmatizer.lemmatize(word) for word in tokens]

# -----------------------
# POS Tagging
# -----------------------
def nltk_pos(tokens):
    try:
        return pos_tag(tokens)
    except:
        return []

def spacy_pos(text):
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]

# -----------------------
# Apply Pipeline
# -----------------------
print("Cleaning text...")

df["Clean_Text"] = df["Text"].apply(clean_text)

print("Whitespace Tokenization...")
df["Whitespace"] = df["Clean_Text"].apply(whitespace_tokenizer)

print("NLTK Tokenization...")
df["NLTK"] = df["Clean_Text"].apply(nltk_tokenizer)

print("spaCy Tokenization...")
df["spaCy"] = df["Clean_Text"].apply(spacy_tokenizer)

print("Subword Tokenization...")
df["Subword"] = df["Clean_Text"].apply(bert_subword)

print("Removing Stopwords...")
df["Stopwords_Removed"] = df["NLTK"].apply(remove_stopwords)

print("Porter Stemming...")
df["Porter"] = df["Stopwords_Removed"].apply(porter_stemming)

print("Snowball Stemming...")
df["Snowball"] = df["Stopwords_Removed"].apply(snowball_stemming)

print("Lemmatization...")
df["Lemma"] = df["Stopwords_Removed"].apply(lemmatization)

print("NLTK POS Tagging...")
df["NLTK_POS"] = df["Lemma"].apply(nltk_pos)

print("spaCy POS Tagging...")
df["spaCy_POS"] = df["Clean_Text"].apply(spacy_pos)

# -----------------------
# Save Output
# -----------------------
df.to_csv("processed_articles.csv", index=False)

print("\nProcessing Completed Successfully!")
print("Output saved as processed_articles.csv")

# -----------------------
# Compare NLTK vs spaCy
# -----------------------
print("\n==============================")
print("NLTK vs spaCy Comparison")
print("==============================")

sample = df.iloc[0]

print("\nOriginal Text:\n")
print(sample["Text"][:500])

print("\nNLTK Tokens:\n")
print(sample["NLTK"][:30])

print("\nspaCy Tokens:\n")
print(sample["spaCy"][:30])
