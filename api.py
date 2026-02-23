import os
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

VECTOR_DIR = "vector_store_1"
DOCS_DIR   = "data/policies"

app = FastAPI(title="HR Policy Assistant API")

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def load_vectorstore():
    return FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)

vectorstore = load_vectorstore()
retriever   = vectorstore.as_retriever(search_kwargs={"k": 4})

# LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# Prompt — instructs the LLM to cite the source doc inline
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a knowledgeable and helpful HR Policy Assistant.

Use ONLY the policy context below to answer the question.
- Be concise and accurate.
- At the end of each fact, cite the source in parentheses like: (Document Name | Page X)
- If the answer is NOT in the context, say exactly: "I don't know based on the available policy documents."
- Do NOT fabricate or guess.

Policy Context:
{context}

Question: {question}

Answer:"""
)

def build_chain(ret):
    return (
        {
            "context": ret | (lambda docs: "\n\n".join(
                f"[{d.metadata.get('document_name','Document')} | "
                f"Page {d.metadata.get('page_number','?')}]\n{d.page_content}"
                for d in docs
            )),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

chain = build_chain(retriever)

STOP_WORDS = {"the","a","an","is","are","be","to","of","and","in","that","it",
              "for","on","with","as","was","at","by","from","or","but","not","this","have"}

def get_sources(question: str, answer: str = ""):
    """Retrieve docs and only return sources whose content actually appears in the answer."""
    docs = retriever.invoke(question)
    seen = set()
    sources = []
    answer_lower = answer.lower()
    for d in docs:
        name = d.metadata.get("document_name", "Company Policy")
        page = d.metadata.get("page_number", "?")
        key  = f"{name}|{page}"
        if key in seen:
            continue
        # Check how many meaningful words from this chunk appear in the answer
        chunk_words = set(d.page_content.lower().split())
        meaningful  = [w for w in chunk_words if len(w) > 4 and w not in STOP_WORDS]
        matches     = sum(1 for w in meaningful if w in answer_lower)
        if matches >= 3:
            seen.add(key)
            sources.append({"document": name, "page": page})
    return docs, sources

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(req: QueryRequest):
    try:
        docs = retriever.invoke(req.question)
        if not docs:
            return {"answer": "I don't know based on the available policy documents.", "sources": []}

        answer = chain.invoke(req.question).strip()

        if not answer or answer.lower().startswith("i don't know"):
            return {"answer": answer or "I don't know based on the available policy documents.", "sources": []}

        # Now filter sources against the actual answer
        _, sources = get_sources(req.question, answer)

        # Fallback: if filter was too strict, use the top doc at least
        if not sources and docs:
            top = docs[0]
            sources = [{
                "document": top.metadata.get("document_name", "Company Policy"),
                "page": top.metadata.get("page_number", "?")
            }]

        return {"answer": answer, "sources": sources}

    except Exception as e:
        print("Error:", e)
        return {"answer": f"Error: {str(e)}", "sources": []}


@app.post("/admin/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return {"status": "error", "message": "Only PDF files are supported."}
    save_path = os.path.join(DOCS_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    background_tasks.add_task(reindex_document, save_path, file.filename)
    return {"status": "success", "message": f"'{file.filename}' uploaded. Indexing in progress..."}


def reindex_document(filepath: str, filename: str):
    global vectorstore, retriever, chain
    try:
        doc_name = filename.replace("_", " ").replace(".pdf", "").title()
        loader   = PyPDFLoader(filepath)
        pages    = loader.load()
        for page in pages:
            page.metadata["source_file"]   = filename
            page.metadata["document_name"] = doc_name
            page.metadata["page_number"]   = page.metadata.get("page", 0) + 1
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks   = splitter.split_documents(pages)
        for c in chunks:
            c.metadata["source_file"]   = filename
            c.metadata["document_name"] = doc_name
        vectorstore.add_documents(chunks)
        vectorstore.save_local(VECTOR_DIR)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        chain     = build_chain(retriever)
        print(f"✅ Re-indexed: {filename} ({len(chunks)} chunks)")
    except Exception as e:
        print(f"❌ Re-index failed for {filename}: {e}")


@app.get("/admin/documents")
def list_documents():
    if not os.path.exists(DOCS_DIR):
        return {"documents": []}
    files = sorted([f for f in os.listdir(DOCS_DIR) if f.endswith(".pdf")])
    return {
        "documents": [
            {"filename": f, "name": f.replace("_", " ").replace(".pdf", "").title()}
            for f in files
        ],
        "count": len(files)
    }

@app.get("/health")
def health():
    return {"status": "ok"}