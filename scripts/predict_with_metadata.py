import joblib
from preprocess import clean_text, extract_metadata

# Load model + vectorizer
model = joblib.load("doc_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_doc(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    metadata = extract_metadata(text)
    return prediction, metadata

# Test
sample = "India wins cricket world cup 2025 in Mumbai"
pred, meta = predict_doc(sample)
print("Text:", sample)
print("Predicted Label:", pred)
print("Metadata:", meta)
