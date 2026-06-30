"""
Central configuration for FinSight.
"""

import os
from dotenv import load_dotenv

load_dotenv()

#LLM Provider - switch between 'groq' and 'ollama'
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

#Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL")

#LLM
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")

#Embeddings
EMBED_MODEL = os.getenv("EMBED_MODEL")

#RAG
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL"))

#Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")

API_SECRET_KEY = os.getenv("API_SECRET_KEY")