import os
import sys
import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from config import get_llm, CHROMA_DIR



COLLECTION_NAME = "edu_documents"
EMBED_MODEL     = "sentence-transformers/all-mpnet-base-v2"

# Cached objects — created once, reused on every call
_chroma_client  = None
_raw_collection = None
_vectorstore    = None
_embeddings     = None


def get_embeddings():
    """Load the HuggingFace embedding model (only once)."""
    global _embeddings
    if _embeddings is None:
        print("Loading embedding model...", file=sys.stderr)
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        print("Embedding model loaded.", file=sys.stderr)
    return _embeddings


def get_chroma_client():
    """Get (or create) the persistent ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _chroma_client


def get_raw_collection():
    """
    Get the ChromaDB collection WITHOUT registering an embedding function.

    Passing embedding_function=None tells ChromaDB:
    'I will handle embeddings myself, don't touch them.'
    This avoids the conflict error completely.
    """
    global _raw_collection
    if _raw_collection is None:
        _raw_collection = get_chroma_client().get_or_create_collection(
            name=COLLECTION_NAME
            # No embedding_function here — this is the fix!
        )
    return _raw_collection


def get_vectorstore():
    """
    Get the LangChain Chroma vectorstore (used for similarity search).
    This uses HuggingFaceEmbeddings — same model as before, just one wrapper.
    """
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            client=get_chroma_client(),
            collection_name=COLLECTION_NAME,
            embedding_function=get_embeddings()
        )
    return _vectorstore


def is_store_empty() -> bool:
    """Returns True if no documents have been uploaded yet."""
    return get_raw_collection().count() == 0


def is_pdf_already_uploaded(filename: str) -> bool:
    """Check if a PDF with this filename is already in the database."""
    results = get_raw_collection().get(where={"source_file": filename})
    return len(results["ids"]) > 0


def upload_pdf(pdf_path: str) -> dict:
    """
    Upload a PDF file: read it, split into chunks, embed and store in ChromaDB.

    Returns a dict with success/failure info.
    """
    filename = os.path.basename(pdf_path)

    # Don't upload the same file twice
    if is_pdf_already_uploaded(filename):
        return {
            "success": False,
            "reason":  "duplicate",
            "message": f"'{filename}' is already uploaded."
        }

    # Read the PDF
    try:
        loader = PyPDFLoader(pdf_path)
        pages  = loader.load()
    except Exception as e:
        return {
            "success": False,
            "reason":  "read_error",
            "message": f"Failed to read PDF: {str(e)}"
        }

    if not pages:
        return {
            "success": False,
            "reason":  "empty_pdf",
            "message": f"'{filename}' has no readable text."
        }

    # Split into small chunks so search works well
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(pages)

    if not chunks:
        return {
            "success": False,
            "reason":  "no_chunks",
            "message": f"Could not split '{filename}' into chunks."
        }

    # Tag each chunk with the source filename
    for chunk in chunks:
        chunk.metadata["source_file"] = filename

    # Embed the chunks using our HuggingFace model
    try:
        embeddings_model = get_embeddings()
        texts      = [chunk.page_content for chunk in chunks]
        embeddings = embeddings_model.embed_documents(texts)   # list of vectors

        collection = get_raw_collection()
        collection.add(
            documents=texts,
            embeddings=embeddings,   # we provide embeddings ourselves
            ids=[f"{filename}_chunk_{i}" for i in range(len(chunks))],
            metadatas=[
                {
                    "source_file": filename,
                    "page": chunk.metadata.get("page", 0)
                }
                for chunk in chunks
            ]
        )
    except Exception as e:
        return {
            "success": False,
            "reason":  "store_error",
            "message": f"Failed to store chunks: {str(e)}"
        }

    return {
        "success":      True,
        "filename":     filename,
        "chunks_added": len(chunks),
        "pages_read":   len(pages)
    }


def get_context_for_topic(topic: str) -> str:
    """Search the vectorstore for content related to a topic."""
    if is_store_empty():
        return ""
    retriever = get_vectorstore().as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.8}
    )
    docs = retriever.invoke(topic)
    return "\n\n".join(doc.page_content for doc in docs)


def run_rag_agent(query: str) -> str:
    """Answer a student's question using the uploaded PDF notes."""
    if is_store_empty():
        return "No study materials uploaded yet. Please upload PDF notes first."

    retriever = get_vectorstore().as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.8}
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful study assistant for students.
Answer the question using ONLY the information from the study notes provided below.

Rules:
- If the answer is clearly in the notes, answer directly and concisely
- If the answer is partially in the notes, answer what you can and mention what is missing
- If the answer is not in the notes at all, say exactly: "This topic is not covered in the uploaded notes."
- Never use outside knowledge
- Keep answers clear and easy for a student to understand

Study Notes:
{context}"""),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )

    try:
        return chain.invoke(query)
    except Exception as e:
        return f"Failed to search documents: {str(e)}"