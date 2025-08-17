import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import clean_text

# -------------------------------
# Load and sample dataset
# -------------------------------
df = pd.read_csv("data/news.csv")
print("Dataset shape before cleaning:", df.shape)

df.dropna(subset=['text', 'label'], inplace=True)
df['text'] = df['text'].astype(str)

# Take smaller sample for faster search
df = df.sample(5000, random_state=42)
print("Dataset shape after sampling:", df.shape)

# -------------------------------
# Cache cleaned text
# -------------------------------
if "cleaned" not in df.columns:
    print("Cleaning text... (first time may take a while)")
    df['cleaned'] = df['text'].apply(clean_text)
    df.to_csv("data/news_cleaned.csv", index=False)
else:
    print("✅ Using pre-cleaned dataset")

# -------------------------------
# Build TF-IDF
# -------------------------------
vectorizer = TfidfVectorizer(max_features=3000)
doc_vectors = vectorizer.fit_transform(df['cleaned'])
print("✅ TF-IDF ready:", doc_vectors.shape)


def search(query, top_k=5):
    query_clean = clean_text(query)
    query_vec = vectorizer.transform([query_clean])
    sims = cosine_similarity(query_vec, doc_vectors).flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    return [(df.iloc[i]['text'], df.iloc[i]['label'], sims[i]) for i in top_idx]


if __name__ == "__main__":
    query = "AI research and new technology"
    results = search(query, top_k=5)

    print(f"\nTop {len(results)} results for query: {query}\n")
    for text, label, score in results:
        print(f"[{label}] ({score:.2f}) {text}")
