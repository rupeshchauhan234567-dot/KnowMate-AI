from io import BytesIO
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader
from langchain_core.documents import Document


def load_uploaded_documents(uploaded_files):
    """Extract text from Streamlit uploads into LangChain Document objects."""
    documents = []
    names = []

    for uploaded in uploaded_files:
        filename = Path(uploaded.name).name
        suffix = Path(filename).suffix.lower()
        raw = uploaded.getvalue()
        names.append(filename)

        if suffix == ".pdf":
            reader = PdfReader(BytesIO(raw))
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": filename, "page": page_number},
                    ))

        elif suffix == ".docx":
            doc = DocxDocument(BytesIO(raw))
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={"source": filename, "page": "N/A"},
                ))

        elif suffix == ".txt":
            text = raw.decode("utf-8", errors="ignore")
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={"source": filename, "page": "N/A"},
                ))

    return documents, names
