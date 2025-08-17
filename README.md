
# 🧠 Document Intelligence (Mini)

**Document Intelligence (Mini)** is a **Streamlit-based AI application** that allows users to:

- 📌 Classify documents into categories (Politics, Sports, Tech, etc.)
- 🏷️ Extract metadata like organizations, names, dates, money
- 🔍 Perform semantic search across a dataset of news articles

This project demonstrates **NLP, machine learning, and web app deployment** skills in a compact, resume-ready project.

---

## ⚡ Features

1. **Upload & Classify**
   - Upload TXT or PDF documents.
   - Model predicts the category of the document using **TF-IDF + Logistic Regression**.

2. **Metadata Extraction**
   - Extract entities using **spaCy Named Entity Recognition (NER)**.
   - Displays entities like ORG, DATE, MONEY, PERSON.

3. **Search Dataset**
   - Search through a news dataset using **TF-IDF + Cosine Similarity**.
   - Returns the top-k relevant articles for a given query.

4. **Interactive GUI**
   - Built with **Streamlit** for a clean and intuitive interface.

---

## 🛠️ Tech Stack

- Python 3.x
- **Streamlit** – Web app framework
- **pandas** – Data manipulation
- **scikit-learn** – ML models (TF-IDF, Logistic Regression)
- **spaCy** – NLP preprocessing & NER
- **PyPDF2** – PDF text extraction
- **joblib** – Save/load ML models

---

## 📂 Dataset

- Original dataset: [Kaggle News Category Dataset](https://www.kaggle.com/rmisra/news-category-dataset)
- Preprocessed CSV (`news.csv`) contains:
  - `text` → Article headline
  - `label` → Category (top 5 categories only)

---

## ⚡ Installation

1. **Clone the repo**
```bash
git clone https://github.com/yourusername/document-intelligence-mini.git
cd document-intelligence-mini
````

2. **Create virtual environment**

```bash
python -m venv .venv
```

3. **Activate environment**

* Windows:

```bash
.venv\Scripts\activate
```

* macOS/Linux:

```bash
source .venv/bin/activate
```

4. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the App

```bash
streamlit run app.py
```

* Upload a TXT/PDF file to classify and extract metadata.
* Try the search feature using queries like `"AI research in healthcare"`.

---

## 🧩 Project Structure

```
Document-Intelligence-Mini/
│
├─ app.py                  # Streamlit frontend
├─ preprocess.py           # Text cleaning & NER
├─ utils.py                # PDF/TXT reading functions
├─ search.py               # TF-IDF + cosine similarity search
├─ models.py               # ML model training & saving
├─ convert_dataset.py      # Preprocess raw dataset to CSV
├─ requirements.txt      
│
├─ data/                
│  ├─ news.csv
│  └─ News_Category_Dataset_v3.json
│
├─ scripts/           
│  └─ predict_with_metadata.py
│
├─ tests/               
│  ├─ test_model.py
│  ├─ test_preprocessor.py
│  └─ test_metadata.py
│
└─ README.md
```

---

## 📈 Future Improvements

* Replace TF-IDF with **BERT / SBERT embeddings** for better semantic search.
* Use **vector databases** (FAISS, Pinecone) for faster and scalable search.
* Enhance metadata extraction with **domain-specific NER** models.
* Extend classification to **more categories**.

---

## 📄 License

MIT License © 2025


