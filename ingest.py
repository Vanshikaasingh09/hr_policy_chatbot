import os
import ssl
import nltk

from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# SSL fix (Windows)
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

nltk.download("punkt")

DOCS_DIR = "data/policies"
VECTOR_DIR = "vector_store"

print("📂 Checking documents directory:", DOCS_DIR)

if not os.path.exists(DOCS_DIR):
    raise RuntimeError("❌ data/policies folder does NOT exist")

files = os.listdir(DOCS_DIR)
print("📄 Files found:", files)

if not files:
    raise RuntimeError("❌ data/policies is EMPTY. Add PDFs.")

# Load PDFs
loader = DirectoryLoader(
    path=DOCS_DIR,
    glob="**/*.pdf",
    loader_cls=UnstructuredFileLoader
)

documents = loader.load()
print(f"✅ Loaded {len(documents)} documents")

if not documents:
    raise RuntimeError("❌ No documents loaded from PDFs")

# Split documents
text_splitter = CharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300
)

chunks = text_splitter.split_documents(documents)
print(f"✂️ Created {len(chunks)} text chunks")

if not chunks:
    raise RuntimeError("❌ No chunks created")

# Embeddings (FAST & LIGHT)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# FAISS index
vectorstore = FAISS.from_documents(chunks, embeddings)

vectorstore.save_local(VECTOR_DIR)

print("🎉 SUCCESS: FAISS vector_store created")  