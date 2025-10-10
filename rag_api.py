import os
import time
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

# --- Config ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
CHROMA_DIR = "./chroma_db"
if not GOOGLE_API_KEY:
    raise ValueError("Set GOOGLE_API_KEY environment variable.")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# --- FastAPI App ---
app = FastAPI()

class QueryRequest(BaseModel):
    input: str

# --- Initialize Vectorstore ---
def get_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    
    if os.path.exists(CHROMA_DIR) and len(os.listdir(CHROMA_DIR)) > 0:
        print("🔹 Loading existing Chroma index...")
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    print("🆕 Creating new Chroma index...")
    csv_path = os.path.join(os.getcwd(), "knowledge_base.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    df = pd.read_csv(csv_path)
    docs = [
        Document(page_content=row["Answer"], metadata={"question": row["Question"]})
        for _, row in df.iterrows()
    ]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    vectorstore.persist()
    return vectorstore

vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# --- LLM and RAG Chain ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are Nomi, a travel assistant. "
     "Answer the user's question using ONLY the context below. "
     "If the answer is not in the context, say: "
     "'I'm sorry, I can only provide information available in my travel knowledge base.'\n\n"
     "Context:\n{context}"),
    ("human", "{input}")
])

qa_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# --- Retry Helper ---
def invoke_with_retry(chain, input_text, retries=3, delay=2):
    for attempt in range(retries):
        try:
            return chain.invoke({"input": input_text})
        except Exception as e:
            print(f"Attempt {attempt+1} failed:", e)
            time.sleep(delay * (attempt + 1))
    # Fallback: return retrieved context if LLM fails
    docs = retriever.get_relevant_documents(input_text)
    if docs:
        return {"answer": docs[0].page_content}
    return {"answer": "Sorry, I couldn't process your request."}

# --- FastAPI Endpoint ---
@app.post("/ask")
async def ask_question(req: QueryRequest):
    result = invoke_with_retry(rag_chain, req.input)
    if isinstance(result, dict):
        answer = result.get("answer") or str(result)
    else:
        answer = str(result)
    return {"answer": answer or "Sorry, no response generated."}
