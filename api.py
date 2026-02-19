import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

VECTOR_DIR = "vector_store"

app = FastAPI(title="Company Policy AI Assistant")

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS
vectorstore = FAISS.load_local(
    VECTOR_DIR,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Prompt
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a company policy assistant.

Rules:
- Use ONLY the policy context below.
- Answer factual or summary questions.
- If the answer is not present in the context, reply exactly:
"I don't know"

Context:
{context}

Question:
{question}

Answer:
"""
)

# ✅ MANUAL RAG CHAIN (THIS IS THE FIX)
chain = (
    {
        "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Request model
class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_policy_question(req: QueryRequest):
    try:
        docs = retriever.get_relevant_documents(req.question)
        print("Retrieved docs:", len(docs))

        if not docs:
            return {"answer": "I don't know", "sources": []}

        answer = chain.invoke(req.question).strip()

        if not answer:
            return {"answer": "I don't know", "sources": []}

        return {
            "answer": answer,
            "sources": ["Company Policy Document"]
        }

    except Exception as e:
        print("Error:", e)
        return {"answer": "I don't know", "sources": []}
