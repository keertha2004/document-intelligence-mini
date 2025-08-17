import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from preprocess import clean_text

# Load dataset
df = pd.read_csv("data/news.csv")

# Clean dataset
df.dropna(subset=['text', 'label'], inplace=True)

# (Optional) Use subset for faster training
df = df.sample(20000, random_state=42)

# Clean text
df['cleaned'] = df['text'].apply(clean_text)

# Features and labels
X = df['cleaned']
y = df['label']

# Vectorization
vectorizer = TfidfVectorizer(max_features=3000)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "doc_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ Model and vectorizer saved.")
