# Conversational RAG with PDF Upload and Chat History

import streamlit as st
import os
import tempfile

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory

st.title("Conversational RAG with PDF Uploads and Chat History")
st.write("Upload PDF files and chat with their content")

# ---- GROQ API ----
api_key = st.text_input("Enter your GROQ API Key:", type="password")

if not api_key:
    st.warning("Please enter the GROQ API Key")
    st.stop()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=api_key,
    temperature=0)

# ---- Session ID ----
session_id = st.text_input("Session ID", value="default_session")

if "store" not in st.session_state:
    st.session_state.store = {}

# ---- Embedding Model ----
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ---- File Upload ----
uploaded_files = st.file_uploader(
    "Choose PDF files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files and "vectorstore" not in st.session_state:

    documents = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            temp_path = tmp.name

        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        documents.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )

    splits = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(splits, embedding)
    st.session_state.vectorstore = vectorstore

    st.success("Documents processed and vectorstore created!")

# ---- Build RAG Pipeline ----
if "vectorstore" in st.session_state:

    retriever = st.session_state.vectorstore.as_retriever()

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, "
         "reformulate it into a standalone question. Do NOT answer it."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_prompt
    )

    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the retrieved context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentences maximum.\n\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    # ---- Session History ----
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # ---- User Query ----
    user_input = st.text_input("Your Question:")

    if user_input:
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )

        st.success(response["answer"])

        # Display Chat History
        session_history = get_session_history(session_id)

        with st.expander("Chat History"):
            for msg in session_history.messages:
                role = "User" if msg.type == "human" else "Assistant"

                st.write(f"**{role}:** {msg.content}")
