# app.py
import streamlit as st
import joblib
import pandas as pd
from preprocess import clean_text, extract_metadata
from utils import read_pdf, read_txt
from search import search

# --- Load model + vectorizer ---
model = joblib.load("doc_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Document Intelligence (Mini)",
    page_icon="🧠",
    layout="wide"
)

# --- Custom CSS Styling ---
st.markdown(
    """
    <style>
    body { background-color: #FAFAFA; }
    .title { font-size:36px !important; font-weight:800; color:#154360; }
    .subtitle { font-size:18px !important; color:#1F618D; }
    .card {
        background-color:#FDFEFE;
        padding:20px;
        border-radius:12px;
        margin:15px 0;
        border:1px solid #D5DBDB;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .highlight {
        background: #EBF5FB;
        padding: 10px 15px;
        border-radius: 8px;
        font-weight: 600;
        color: #1A5276;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True
)

# --- Sidebar Navigation ---
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", ["📤 Upload & Classify", "🔍 Search Dataset", "ℹ️ About"])

# --- Header ---
st.markdown('<p class="title">🧠 Document Intelligence (Mini)</p>', unsafe_allow_html=True)
st.caption("Smart document analysis: classify → extract metadata → semantic search")

# --- Upload & Classify Page ---
if page == "📤 Upload & Classify":
    st.subheader("📄 Upload a Document")
    uploaded_file = st.file_uploader("Choose a TXT or PDF file", type=["txt", "pdf"])

    if uploaded_file:
        # Extract text
        text = read_pdf(uploaded_file) if uploaded_file.type == "application/pdf" else read_txt(uploaded_file)

        # Display text (collapsible)
        with st.expander("🔎 Extracted Text (click to expand)"):
            st.write(text if len(text) < 2000 else text[:2000] + "...")

        # Classification
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]

        st.markdown("### 📌 Classification")
        st.markdown(f"<div class='highlight'>✅ Classified as: <b>{prediction}</b></div>", unsafe_allow_html=True)

        # Metadata Extraction
        st.markdown("### 🏷️ Metadata Tags")
        metadata = extract_metadata(text)
        if metadata:
            cols = st.columns(2)
            for i, (ent, label) in enumerate(metadata):
                cols[i % 2].markdown(f"<div class='card'>🔹 <b>{ent}</b> <br><small>{label}</small></div>", unsafe_allow_html=True)
        else:
            st.info("No entities found in this document.")

# --- Search Page ---
elif page == "🔍 Search Dataset":
    st.subheader("🔍 Search in News Dataset")
    query = st.text_input("Enter a query (e.g., 'AI research in healthcare')")

    if query:
        results = search(query, top_k=5)
        st.markdown(f"### 📊 Top results for: **{query}**")
        for text, label, score in results:
            st.markdown(
                f"<div class='card'><b>[{label}]</b> ({score:.2f})<br>{text}</div>",
                unsafe_allow_html=True
            )

# --- About Page ---
elif page == "ℹ️ About":
    st.subheader("ℹ️ About this Project")
    st.write("""
    **Document Intelligence (Mini)** is a machine learning powered app that can:
    - 📌 Classify documents into categories (Politics, Sports, Tech, etc.)
    - 🏷️ Extract metadata like organizations, names, dates, money
    - 🔍 Perform intelligent search across a large dataset
    
    **Tech Stack**: Python, scikit-learn, spaCy, Streamlit, pandas, joblib  
    **Dataset**: Kaggle News Category Dataset  
    
    🚀 Future improvements:  
    - Replace TF-IDF with transformer embeddings (e.g., BERT, SBERT)  
    - Use vector databases (FAISS, Pinecone) for faster search  
    - Improve metadata extraction with domain-specific NER  
    """)
