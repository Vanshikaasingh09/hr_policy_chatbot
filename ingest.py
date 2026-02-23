import os
import ssl
import nltk

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# SSL fix (Windows)
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
except Exception:
    pass

DOCS_DIR = "data/policies"
VECTOR_DIR = "vector_store_1"

print(" Scanning documents in:", DOCS_DIR)

if not os.path.exists(DOCS_DIR):
    raise RuntimeError(" data/policies folder does NOT exist")

pdf_files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")]
print(f" Found {len(pdf_files)} PDFs: {pdf_files}")

if not pdf_files:
    raise RuntimeError(" No PDFs found in data/policies")

# Load each PDF individually to attach rich metadata
all_chunks = []
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " "]
)

for filename in sorted(pdf_files):
    filepath = os.path.join(DOCS_DIR, filename)
    try:
        loader = PyPDFLoader(filepath)
        pages = loader.load()

        # Friendly document name from filename
        doc_name = filename.replace("_", " ").replace(".pdf", "").title()

        # Attach metadata to every page
        for page in pages:
            page.metadata["source_file"] = filename
            page.metadata["document_name"] = doc_name
            page.metadata["page_number"] = page.metadata.get("page", 0) + 1

        chunks = splitter.split_documents(pages)

        # Ensure metadata flows into chunks
        for chunk in chunks:
            chunk.metadata["source_file"] = filename
            chunk.metadata["document_name"] = doc_name

        all_chunks.extend(chunks)
        print(f"   {doc_name}: {len(pages)} pages → {len(chunks)} chunks")

    except Exception as e:
        print(f"    Skipped {filename}: {e}")

print(f"\n Total chunks: {len(all_chunks)}")

if not all_chunks:
    raise RuntimeError(" No chunks created — check your PDFs")

# Build & save FAISS index
vectorstore = FAISS.from_documents(all_chunks, embeddings)
vectorstore.save_local(VECTOR_DIR)
print(f"\n SUCCESS — FAISS vector store saved to '{VECTOR_DIR}'")
print(f"   Documents indexed: {len(pdf_files)}")
print(f"   Total chunks:      {len(all_chunks)}")