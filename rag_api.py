import os
import pandas as pd
from langchain.docstore.document import Document
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from fastapi import FastAPI
from pydantic import BaseModel

rag_api = FastAPI()

conversational_rag_chain = None

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY. Set it in Heroku config vars.")

class QueryRequest(BaseModel):
    session_id: str
    input: str

# Lazy-load function
async def get_conversational_chain():
    global conversational_rag_chain
    if conversational_rag_chain is None:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

        csv_path = os.path.join(os.getcwd(), "knowledge_base.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at {csv_path}")
        df = pd.read_csv(csv_path)
        docs_csv = [Document(page_content=f"Q: {row['Question']}\nA: {row['Answer']}") for _, row in df.iterrows()]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs_csv)

        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "You are Nomi, a travel assistant. "
                 "You only answer questions related to travel, tourism, local spots, restaurants, hotels, events, and activities. "
                 "If the question is outside travel, respond politely: 'I'm sorry, I can only provide travel-related information.'\n\n{context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are Nomi, a travel assistant.\n\n{context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        store = {}
        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in store:
                store[session_id] = ChatMessageHistory()
            return store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    return conversational_rag_chain

@rag_api.post("/ask")
async def ask_question(req: QueryRequest):
    try:
        chain = await get_conversational_chain()
        temp_session_id = f"{req.session_id}_{os.urandom(4).hex()}"

        response = await chain.ainvoke(
            {"input": req.input},
            config={"configurable": {"session_id": temp_session_id}}
        )

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



