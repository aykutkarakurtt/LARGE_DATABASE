import os

import pymupdf
from docx import Document

from config import SUPPORTED_EXTENSIONS


def load_document(file_path):
    extension = os.path.splitext(file_path)[1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        return None
    if extension == ".pdf":
        return _load_pdf(file_path)
    if extension == ".txt":
        return _load_txt(file_path)
    if extension == ".docx":
        return _load_docx(file_path)
    return None


def _load_pdf(file_path):
    pages = []
    document = pymupdf.open(file_path)
    try:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text()
            if text.strip():
                pages.append({"page": page_number, "text": text})
    finally:
        document.close()
    return {"source": os.path.basename(file_path), "pages": pages}


def _load_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        text = file.read()
    return {
        "source": os.path.basename(file_path),
        "pages": [{"page": 1, "text": text}],
    }


def _load_docx(file_path):
    document = Document(file_path)
    pages = []
    for page_number, paragraph in enumerate(document.paragraphs, start=1):
        if paragraph.text.strip():
            pages.append({"page": page_number, "text": paragraph.text})
    return {"source": os.path.basename(file_path), "pages": pages}


def location_kind(source):
    if os.path.splitext(source)[1].lower() == ".docx":
        return "Paragraf"
    return "Sayfa"


def discover_documents(data_dir):
    if not os.path.isdir(data_dir):
        return []

    documents = []
    for root, directories, files in os.walk(data_dir):
        directories.sort()
        for filename in sorted(files):
            extension = os.path.splitext(filename)[1].lower()
            if extension in SUPPORTED_EXTENSIONS:
                documents.append(os.path.join(root, filename))
    return documents
