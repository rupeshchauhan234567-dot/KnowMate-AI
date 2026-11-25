
from pathlib import Path

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_rag_chain(docs, session_id):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("No text chunks could be created from these files.")

    embeddings = get_embeddings()

    collection_name = f"knowmate_{session_id}"

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(CHROMA_DIR),
    )


def get_llm(api_key, model_name):
    if not api_key:
        raise ValueError("Please enter your Groq API key.")

    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0.2,
    )


def get_document_context(vector_store, query, k=8):
    retrieved_docs = vector_store.similarity_search(query, k=k)

    if not retrieved_docs:
        return "", []

    context_parts = []
    sources = []
    seen_sources = set()

    for doc in retrieved_docs:

        source = doc.metadata.get("source", "Unknown document")
        page = doc.metadata.get("page", "N/A")

        context_parts.append(
            f"[Source: {source}, Page: {page}]\n{doc.page_content}"
        )

        source_label = f"{source} — Page {page}"

        if source_label not in seen_sources:
            sources.append(source_label)
            seen_sources.add(source_label)

    return "\n\n---\n\n".join(context_parts), sources


def ask_question(vector_store, question, api_key, model_name):

    context, sources = get_document_context(
        vector_store,
        question,
        k=4,
    )

    if not context:
        return (
            "I couldn't find relevant information in the uploaded documents.",
            [],
        )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are KnowMate AI, a helpful document question-answering assistant.

Answer using ONLY the supplied document context.

If the answer is not present in the context, clearly say you could not find it
in the uploaded documents.

Do not invent facts. Keep the answer clear and useful.
When useful, mention the source filename and page number.""",
        ),
        (
            "human",
            "DOCUMENT CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}",
        ),
    ])

    llm = get_llm(api_key, model_name)

    response = (prompt | llm).invoke({
        "context": context,
        "question": question,
    })

    return response.content, sources


def generate_document_summary(vector_store, api_key, model_name):

    context, sources = get_document_context(
        vector_store,
        "Main topics, important information, key concepts, facts and conclusions",
        k=10,
    )

    if not context:
        raise ValueError("No relevant document content was found.")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are KnowMate AI, an expert document summarizer.

Create a clear and useful summary using ONLY the supplied context.

Include:
1. Overview
2. Main topics
3. Important points
4. Key facts and findings
5. Conclusion

Use headings and bullet points.
Do not invent information or add outside knowledge.""",
        ),
        (
            "human",
            "DOCUMENT CONTENT:\n{context}",
        ),
    ])

    llm = get_llm(api_key, model_name)

    response = (prompt | llm).invoke({
        "context": context,
    })

    return response.content


def generate_quiz(
    vector_store,
    api_key,
    model_name,
    number_of_questions=5,
):

    context, sources = get_document_context(
        vector_store,
        "Important concepts, definitions, facts and key topics for a quiz",
        k=10,
    )

    if not context:
        raise ValueError("No relevant document content was found.")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are KnowMate AI, an educational quiz generator.

Create multiple-choice questions using ONLY the supplied document context.

Requirements:
- Create exactly the requested number of questions.
- Give four options: A, B, C and D.
- Only one option should be correct.
- Mark the correct answer clearly.
- Add a short explanation for each answer.
- Do not invent facts outside the document.

Use clean Markdown formatting.""",
        ),
        (
            "human",
            """Create {number_of_questions} questions.

DOCUMENT CONTENT:
{context}""",
        ),
    ])

    llm = get_llm(api_key, model_name)

    response = (prompt | llm).invoke({
        "context": context,
        "number_of_questions": number_of_questions,
    })

    return response.content
