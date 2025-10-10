import os
import time
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.combine_documents import StuffDocumentsChain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain

# ---------------------------
# FastAPI setup
# ---------------------------
app = FastAPI()
conversational_rag_chain = None

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY. Set it in Heroku config vars.")

CHROMA_DIR = os.path.join(os.getcwd(), "chroma_db")

class QueryRequest(BaseModel):
    session_id: str
    input: str

# ---------------------------
# Helper: Create / load Chroma
# ---------------------------
def get_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    
    # If index exists, load it
    if os.path.exists(CHROMA_DIR) and len(os.listdir(CHROMA_DIR)) > 0:
        print("🔹 Loading existing Chroma index...")
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    
    # Otherwise, create new index from CSV
    print("Creating new Chroma index...")
    csv_path = os.path.join(os.getcwd(), "knowledge_base.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    docs = [Document(page_content=f"Q: {row['Question']}\nA: {row['Answer']}") for _, row in df.iterrows()]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=CHROMA_DIR)
    vectorstore.persist()
    return vectorstore

# ---------------------------
# Build conversational RAG chain
# ---------------------------
async def get_conversational_chain():
    global conversational_rag_chain
    if conversational_rag_chain is None:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        vectorstore = get_vectorstore()
        
        # Use top 3 relevant docs
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Strict QA prompt: only use retrieved docs
        qa_prompt_str = """
            You are Nomi, a travel assistant. Use ONLY the information provided below to answer the question.
            Do not include any information that is not present in the context.
            If the answer is not in the context, say:
            "I'm sorry, I can only provide information from the travel knowledge base."

            Context:
            {context}

            Question: {question}
            Answer:
        """
        qa_prompt = PromptTemplate(template=qa_prompt_str, input_variables=["context", "question"])
        
        # Stuff chain strictly uses retrieved docs
        question_answer_chain = StuffDocumentsChain(llm=llm, prompt=qa_prompt, document_variable_name="context")
        
        # Wrap retriever to respect chat history
        contextualize_q_prompt = PromptTemplate(
            template="Use the following context to answer the user's question. {context}",
            input_variables=["context"]
        )
        
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        # Session history storage
        store = {}
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in store:
                store[session_id] = ChatMessageHistory()
            return store[session_id]
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
    return conversational_rag_chain

# ---------------------------
# FastAPI endpoint
# ---------------------------
@app.post("/ask")
async def ask_question(req: QueryRequest):
    try:       
        chain = await get_conversational_chain()
        temp_session_id = f"{req.session_id}_{os.urandom(4).hex()}"
        
        response = chain.invoke(
            {"input": req.input},
            config={"configurable": {"session_id": temp_session_id}}
        )
        
        if isinstance(response, dict):
            answer = response.get("answer") or str(response)
        else:
            answer = str(response)
        
        return {"answer": answer or "Sorry, no response generated."}
    
    except Exception as e:
        print("RAG chain error:", e)
        return {"answer": "Sorry, I couldn't process your request."}
