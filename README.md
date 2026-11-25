# KnowMate AI — RAG Document Assistant

KnowMate AI is a local Retrieval-Augmented Generation (RAG) app built with Python and Streamlit. Upload PDF, DOCX, and TXT files, then ask questions grounded in their contents.

## Features
- Multiple PDF, DOCX, and TXT uploads
- PDF page metadata for source references
- Text chunking with overlap
- Local HuggingFace sentence embeddings
- ChromaDB vector storage and semantic retrieval
- Groq-powered answers grounded in retrieved context
- Source list for each answer
- Streamlit chat interface and workspace overview

## Requirements
- Python 3.10 or newer
- A Groq API key
- Internet access for installing packages, downloading the embedding model, and calling Groq

## Setup (Windows PowerShell)

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and add your Groq API key:

```env
GROQ_API_KEY=your_actual_key
```

Run the app:

```powershell
streamlit run app.py
```

You can also paste your Groq API key into the sidebar instead of using `.env`.

## Notes
- The HuggingFace embedding model downloads the first time the app processes documents. This can take a few minutes.
- The first run may be slower because of model download and initialization.
- ChromaDB data is stored locally under `data/chroma_db/`.
- This is a local MVP, not a production multi-user service. Do not expose it publicly without adding authentication, access controls, and secure deployment configuration.
- Scanned/image-only PDFs may need OCR; this MVP extracts selectable PDF text.
- DOCX page numbers are not available in this simple extractor, so their source page is shown as N/A.
