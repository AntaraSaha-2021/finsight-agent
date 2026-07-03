from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings

#---------------Constants--------------------
MAX_PDF_SIZE_MB = 50
VECTORSTORE_PATH = Path(settings.VECTORSTORE_DIR)
INDEX_FILE = VECTORSTORE_PATH / "index.faiss"
METADATA_FILE = VECTORSTORE_PATH / "metadata.pkl"

#---------------Embedding Model-----------------------
print("[RAG] Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name = settings.EMBED_MODEL,
    model_kwargs ={"device": "cpu"},
    encode_kwargs ={"normalize_embeddings": True}
)
print("[RAG] Embedding Model ready.")

#--------------Test Splitter---------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " ", ""]
)

#------------------PDF Validation------------------------
def validate_pdf(file_path: str) -> tuple[bool, str]:
    #returns (is_valid, error_message)
    path = Path(file_path)

    if not path.exists():
        return False, f"File not found: {file_path}"
    
    if path.suffix.lower() != ".pdf":
        return False, "Only .pdf files are accepted"
    
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb>MAX_PDF_SIZE_MB:
        return False, f"PDF too large: {size_mb:.1f}MB (max{MAX_PDF_SIZE_MB}MB)"
    
    return True,""

#----------------Ingestion--------------------------
def ingest_pdf(file_path: str) -> dict:
    #ingests pdf into FAISS vector store
    is_valid, error = validate_pdf(file_path)

    if not is_valid:
        return {"success": False, "error": error}
    
    #Load PDF
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
    except Exception as e:
        return {"success": False, "error": f"Failed to read PDF: {str(e)}"}

    if not pages:
        return {
            "success": False,
            "error": "PDF appears to be empty or image-based (no extractable text)"
        }
    
    #Chunk
    chunks = text_splitter.split_documents(pages)

    if not chunks:
        return {"success": False, "error": "No text chunks produced from PDF"}
    
    #Embed and store
    try:
        VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)

        if INDEX_FILE.exists():
            #Load existing index and add new documents
            vectorstore = FAISS.load_local(
                str(VECTORSTORE_PATH),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            vectorstore.add_documents(chunks)
        else:
            #Create new index
            vectorstore = FAISS.from_documents(chunks, embeddings)
        # Persist to disk
        vectorstore.save_local(str(VECTORSTORE_PATH))
    except Exception as e:
        return {"success": False, "error": f"Indexing failed: {str(e)}"}

    return {
        "success": True,
        "pages": len(pages),
        "chunks": len(chunks),
        "filename": Path(file_path).name,
    }

#------------------Retrieval---------------------------------
def retrieve(query: str) -> list[str]:
    #Retrieves top k relevant chunks for a query and returns a list of strings.
    print(f"[RAG DEBUG] INDEX_FILE path: {INDEX_FILE}")
    print(f"[RAG DEBUG] INDEX_FILE exists: {INDEX_FILE.exists()}")
    print(f"[RAG DEBUG] VECTORSTORE_PATH contents: {list(VECTORSTORE_PATH.iterdir()) if VECTORSTORE_PATH.exists() else 'directory missing'}")

    if not INDEX_FILE.exists():
        return []
    
    try:
        vectorstore = FAISS.load_local(
            str(VECTORSTORE_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        docs = vectorstore.similarity_search(
            query,
            k=settings.TOP_K_RETRIEVAL,
        )
        return [doc.page_content for doc in docs]
    except Exception as e:
        #Log but don't crash - agent continues without RAG context
        print(f"[RAG] Retrieval error: {e}")
        return []
    
#---------------------Index status---------------------------
def get_index_status() -> dict:
    return {
        "index_exists": INDEX_FILE.exists(),
        "index_path": str(VECTORSTORE_PATH),
    }