# ----------------------------
# Imports
# ----------------------------
import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
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
def load_pdf_to_pinecone(pdf_path: str):
    # Load PDF
    pdf_loader = PDFPlumberLoader(pdf_path)
    pdf_docs = pdf_loader.load()

    # Split documents intelligently
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?"]
    )
    split_docs = text_splitter.split_documents(pdf_docs)

    # Generate embeddings
    embed_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # Convert to LangChain documents
    study_docs = [Document(page_content=doc.page_content, metadata=doc.metadata)
                  for doc in split_docs]

    # Store in Pinecone
    vector_store = PineconeVectorStore.from_documents(
        documents=study_docs,
        embedding=embed_model,
        index_name=INDEX_NAME
    )

    return vector_store

vector_store = load_pdf_to_pinecone(PDF_PATH)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# ----------------------------
# LLM & RAG Chain
# ----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    output_key="answer",
    return_messages=True
)

qa_prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template="""
You are Nomi, a helpful assistant answering questions based ONLY on the provided knowledge base.

Context from notes:
{context}

Chat History:
{chat_history}

Question:
{question}

Instructions:
- Answer using ONLY the information in the context above.
- If the answer is not contained in the context, respond exactly:
  "The information is not in the knowledge base."
- Provide a clear, concise answer.
"""
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
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
