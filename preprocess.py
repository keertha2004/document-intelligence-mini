import spacy

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

def clean_text(text: str) -> str:
    """
    Cleans text: lowercases, removes stopwords/punctuation, lemmatizes.
    Returns a cleaned string.
    """
    doc = nlp(text)
    tokens = [t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop]
    return " ".join(tokens)

def extract_metadata(text: str):
    """
    Extracts entities (ORG, DATE, MONEY, etc.) using spaCy NER.
    """
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]
