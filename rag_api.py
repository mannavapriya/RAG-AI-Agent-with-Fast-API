import os
import pandas as pd
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.docstore.document import Document
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

rag_api = FastAPI()

# -------------------------------
# 1️⃣ Define request model
# -------------------------------
class QueryRequest(BaseModel):
    session_id: str
    input: str

# -------------------------------
# 2️⃣ Session management
# -------------------------------
conversational_rag_chain = None
session_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

# -------------------------------
# 3️⃣ Build Conversational Chain
# -------------------------------
async def get_conversational_chain():
    global conversational_rag_chain
    if conversational_rag_chain is not None:
        return conversational_rag_chain

    # ✅ Load free Hugging Face model locally
    model_name = "facebook/blenderbot-400M-distill"  # Small, no agreement, runs on CPU
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=512)
    llm = HuggingFacePipeline(pipeline=pipe)

    # ✅ Load knowledge base
    csv_path = os.path.join(os.getcwd(), "knowledge_base.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    df = pd.read_csv(csv_path)
    docs_csv = [
        Document(page_content=f"Q: {row['Question']}\nA: {row['Answer']}")
        for _, row in df.iterrows()
    ]

    # ✅ Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs_csv)

    # ✅ Use local embeddings (no API, no quota)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # ✅ Use FAISS for local vector store
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # ✅ Contextualization prompt
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are Nomi, a travel assistant. "
             "Your goal is to restate the user's question clearly for searching the knowledge base. "
             "Resolve pronouns like 'it' or 'this app' only if the context clearly indicates them. "
             "Do NOT answer the question here, only clarify it."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # ✅ QA prompt
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are Nomi, a travel assistant. "
             "Answer only using the provided knowledge base context.\n\n"
             "{context}\n\n"
             "Rules:\n"
             "1. If the answer is in the context, return it accurately.\n"
             "2. If not, reply exactly with: \"I'm sorry, I don't know.\"\n"
             "3. Do NOT guess or include outside knowledge."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )
    return conversational_rag_chain

# -------------------------------
# 4️⃣ API Endpoint
# -------------------------------
@rag_api.post("/ask")
async def ask_question(req: QueryRequest):
    try:
        chain = await get_conversational_chain()

        response = await asyncio.to_thread(
            lambda: chain.invoke(
                {"input": req.input},
                config={"configurable": {"session_id": req.session_id}}
            )
        )

        print("Full RAG response:", response)

        if isinstance(response, dict):
            answer = response.get("answer") or response.get("output_text") or str(response)
        else:
            answer = str(response)

        return {"answer": answer or "Sorry, no response generated."}

    except Exception as e:
        import traceback
        print("RAG chain error:", e)
        traceback.print_exc()
        return {"answer": "Sorry, I couldn't process your request."}
