import joblib
from preprocess import clean_text

# Load saved model & vectorizer
model = joblib.load("doc_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

sample = "India wins the cricket world cup final"
cleaned = clean_text(sample)
vec = vectorizer.transform([cleaned])

prediction = model.predict(vec)[0]
print("Text:", sample)
print("Predicted Label:", prediction)
