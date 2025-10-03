import os
import bs4
import pandas as pd
from langchain.docstore.document import Document
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from fastapi import FastAPI
from pydantic import BaseModel

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY. Set it in Hugging Face 'Settings > Secrets'.")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Load website docs
loader = WebBaseLoader(
    web_paths=("https://www.cosmo-millennial.com/",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_="content")),
)
web_docs = loader.load()

csv_path = os.path.join(os.getcwd(), "knowledge_base.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found at {csv_path}")

df = pd.read_csv(csv_path)

docs_csv = [Document(page_content=f"Q: {row['Question']}\nA: {row['Answer']}") for _, row in df.iterrows()]

all_docs = web_docs + docs_csv

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(all_docs)

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

# RAG chain setup
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
    [("system", "You are Nomi, a travel assistant.\n\n{context}"),
     MessagesPlaceholder("chat_history"),
     ("human", "{input}")]
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

rag_api = FastAPI()

class QueryRequest(BaseModel):
    session_id: str
    input: str

@rag_api.post("/ask")
async def ask_question(req: QueryRequest):
    response = conversational_rag_chain.invoke(
        {"input": req.input},
        config={"configurable": {"session_id": req.session_id}}
    )
    return {"answer": response["answer"]}
