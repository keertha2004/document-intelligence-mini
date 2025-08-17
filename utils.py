from PyPDF2 import PdfReader

def read_pdf(file_bytes) -> str:
    reader = PdfReader(file_bytes)
    return " ".join([p.extract_text() or "" for p in reader.pages])

def read_txt(file_bytes) -> str:
    return file_bytes.read().decode("utf-8", errors="ignore")
