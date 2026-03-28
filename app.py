import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os

st.set_page_config(page_title="LegalGuard AI", layout="wide")
st.title("⚖️ LegalGuard: Enterprise Contract Intelligence")

# 1. Sidebar for API Key & Upload
with st.sidebar:
    # This checks Streamlit Secrets first, then falls back to user input
    openai_key = st.text_input("Enter OpenAI API Key", type="password") or st.secrets.get("OPENAI_API_KEY")
    uploaded_file = st.file_uploader("Upload Legal Contract (PDF)", type="pdf")

# 2. Cached Ingestion (This prevents re-running the heavy work)
@st.cache_resource(show_spinner="Analyzing document...")
def process_pdf(file_content, key):
    with open("temp.pdf", "wb") as f:
        f.write(file_content)
    
    loader = PyPDFLoader("temp.pdf")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)
    
    embeddings = OpenAIEmbeddings(openai_api_key=key)
    return FAISS.from_documents(chunks, embeddings)

if uploaded_file and openai_key:
    # Process the file once and store it in cache
    vector_db = process_pdf(uploaded_file.getbuffer(), openai_key)
    
    # 3. Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if query := st.chat_input("Ask a question about the contract..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=openai_key)
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=vector_db.as_retriever())
            response = qa_chain.invoke(query)
            answer = response["result"]
            st.write(answer)
            
            # Show Sources
            with st.expander("View Source Material"):
                docs = vector_db.similarity_search(query)
                for i, doc in enumerate(docs[:2]):
                    st.markdown(f"**Source {i+1}:**\n{doc.page_content}...")
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
