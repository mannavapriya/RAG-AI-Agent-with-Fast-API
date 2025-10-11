# ----------------------------
# Imports
# ----------------------------
import os
from time import sleep
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryMemory
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PDFPlumberLoader

# ----------------------------
# Config
# ----------------------------
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY. Set it in environment variables.")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY. Set it in environment variables.")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

INDEX_NAME = "kb-index"
PDF_PATH = os.path.join(os.getcwd(), "KB.pdf")

# ----------------------------
# FastAPI App
# ----------------------------
rag_api = FastAPI()

class QueryRequest(BaseModel):
    input: str

# ----------------------------
# Pinecone Setup
# ----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index only if it doesn't exist
if INDEX_NAME not in [i["name"] for i in pc.list_indexes()]:
    print(f"Creating Pinecone index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    sleep(10)

index = pc.Index(INDEX_NAME)

# ----------------------------
# Helper to check if index has data
# ----------------------------
def index_is_empty(index) -> bool:
    stats = index.describe_index_stats()
    return stats.get("total_vector_count", 0) == 0

# ----------------------------
# Load PDF to Pinecone (only once)
# ----------------------------
def load_pdf_to_pinecone(pdf_path: str):
    pdf_loader = PDFPlumberLoader(pdf_path)
    pdf_docs = pdf_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(pdf_docs)

    embed_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # Store directly in Pinecone
    PineconeVectorStore.from_documents(
        documents=split_docs,
        embedding=embed_model,
        index_name=INDEX_NAME
    )
    print("✅ KB.pdf successfully embedded into Pinecone.")

# ----------------------------
# Initialize Vector Store
# ----------------------------
embed_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

if index_is_empty(index):
    print("📘 Index is empty — embedding KB.pdf...")
    load_pdf_to_pinecone(PDF_PATH)
else:
    print("✅ Existing index found — skipping reindexing.")

vector_store = PineconeVectorStore.from_existing_index(
    embedding=embed_model,
    index_name=INDEX_NAME
)

# ----------------------------
# LLM + Memory + Prompt
# ----------------------------
llm = GoogleGenerativeAI(model="gemini-2.5-flash")

memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    output_key="answer",
    return_messages=True
)

qa_prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template="""
You are Nomi, a helpful assistant. You MUST answer questions using ONLY the information from the provided PDF notes.

Do NOT provide any information that is not in the PDF. If the answer is not in the PDF, respond exactly:
"The information is not in the notes."

Context from notes:
{context}

Chat History:
{chat_history}

Question:
{question}

Answer:
"""
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": qa_prompt},
    return_source_documents=True
)

# ----------------------------
# FastAPI Endpoint
# ----------------------------
@rag_api.post("/ask")
async def ask_question(req: QueryRequest):
    try:
        result = qa_chain.invoke({"question": req.input})
        answer = result.get("answer") if isinstance(result, dict) else str(result)
        return {"answer": answer or "Sorry, no response generated."}
    except Exception as e:
        print("RAG chain error:", e)
        return {"answer": "Sorry, I couldn't process your request."}
