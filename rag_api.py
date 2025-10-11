# ----------------------------
# Imports
# ----------------------------
import os
import re
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
from langchain.document_loaders import TextLoader

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
KB_PATH = os.path.join(os.getcwd(), "KB.txt")

# ----------------------------
# FastAPI App
# ----------------------------
rag_api = FastAPI()

class QueryRequest(BaseModel):
    input: str

# ----------------------------
# Initialize Pinecone Index
# ----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

if not pc.has_index(INDEX_NAME):
    pc.create_index(
        name=INDEX_NAME,
        dimension=3072,  # Gemini embedding dimension
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# ----------------------------
# Load PDF & Store in Pinecone
# ----------------------------
def load_doc_to_pinecone(kb_path: str):
    # Load PDF
    txt_loader = TextLoader(kb_path, encoding="utf-8")
    docs = txt_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?"]
    )

    split_docs = text_splitter.split_documents(docs)

    # Generate embeddings
    embed_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    embeddings_list = []
    for doc in split_docs:
        vector = embed_model.embed_query(doc.page_content)
        embeddings_list.append({
            "text": doc.page_content,
            "embedding": vector,
            "metadata": doc.metadata
        })
    # Convert to LangChain documents
    study_docs = []

    for item in embeddings_list:
        study_docs.append(
            Document(
                page_content=item["text"],
                metadata=item["metadata"]
            )
        )

    # Store in Pinecone
    vector_store = PineconeVectorStore.from_documents(
        documents=study_docs,
        embedding=embed_model,
        index_name=INDEX_NAME
    )

    return vector_store

# ----------------------------
# Check if index already has data
# ----------------------------
stats = index.describe_index_stats()
if stats.get("total_vector_count", 0) == 0:
    print("Pinecone index is empty — loading KB.pdf into Pinecone...")
    vector_store = load_doc_to_pinecone(KB_PATH)
else:
    print("Existing data found in Pinecone — skipping reindexing.")
    embed_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = PineconeVectorStore.from_existing_index(
        embedding=embed_model,
        index_name=INDEX_NAME
    )

# ----------------------------
# LLM & RAG Chain
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
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
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
