import streamlit as st
import os
import tempfile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.title("Document Q & A Chatbot")

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=groq_api_key
)

if "chat" not in st.session_state:
    st.session_state.chat = []
if "db" not in st.session_state:
    st.session_state.db = None

file = st.file_uploader("Upload PDF", type=["pdf"])

if file:
    with st.spinner("Reading file..."):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            path = tmp.name

        loader = PyPDFLoader(path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

        emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.from_documents(chunks, emb)

        st.session_state.db = db
        st.success("File ready!")

def answer_question(q):
    if st.session_state.db is None:
        return "Upload a document first."
    
    try:
        retriever = st.session_state.db.as_retriever()
        docs = retriever.invoke(q)

        if len(docs) == 0:
            return "No relevant info found."

        context = "\n".join([d.page_content for d in docs])

        prompt = f"""
Answer from the context only.

Context:
{context}

Question:
{q}
"""

        res = llm.invoke(prompt)
        return res.content

    except Exception as e:
        return f"Error: {str(e)}"

user_q = st.chat_input("Ask something")

if user_q:
    st.session_state.chat.append(("user", user_q))
    reply = answer_question(user_q)
    st.session_state.chat.append(("assistant", reply))

for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.write(msg)