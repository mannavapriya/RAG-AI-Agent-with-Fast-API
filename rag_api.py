import os
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
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY. Set it in environment variables.")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
CHROMA_DIR = os.path.join(os.getcwd(), "chroma_db")

# --- FastAPI App ---
app = FastAPI()

class QueryRequest(BaseModel):
    input: str

# --- Initialize Vectorstore ---
def get_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    if os.path.exists(CHROMA_DIR) and len(os.listdir(CHROMA_DIR)) > 0:
        print("Loading existing Chroma index...")
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    print("Creating new Chroma index...")
    csv_path = os.path.join(os.getcwd(), "knowledge_base.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    df = pd.read_csv(csv_path)
    docs = [Document(page_content=row["Answer"], metadata={"question": row["Question"]})
            for _, row in df.iterrows()]

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
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- LLM & RAG Chain ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are Nomi, a travel assistant. "
     "Answer the user's question using ONLY the context below. "
     "Do not make up answers. "
     "If the question is outside travel, respond politely: "
     "'I'm sorry, I can only provide travel-related information.'\n\n"
     "Context:\n{context}"),
    ("human", "{input}")
])

qa_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# --- FastAPI Endpoint ---
@app.post("/ask")
async def ask_question(req: QueryRequest):
    try:
        result = rag_chain.invoke({"input": req.input})

        if isinstance(result, dict):
            answer = result.get("answer") or str(result)
        else:
            answer = str(result)

        return {"answer": answer or "Sorry, no response generated."}

    except Exception as e:
        print("RAG chain error:", e)
        return {"answer": "Sorry, I couldn't process your request."}
