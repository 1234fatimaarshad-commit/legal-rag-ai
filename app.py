import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# 1. Page Configuration
st.set_page_config(page_title="LegalGuard AI", layout="wide")
st.title("⚖️ LegalGuard: Enterprise Contract Intelligence")

# 2. Access the Secret Key
# This pulls the key you just saved in the Streamlit "Secrets" menu
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except:
    st.error("Please add your GROQ_API_KEY to Streamlit Secrets.")
    st.stop()

# 3. Sidebar for File Upload
with st.sidebar:
    st.header("Upload Center")
    uploaded_file = st.file_uploader("Upload Legal Contract (PDF)", type="pdf")

# 4. Processing Logic (Cached for Speed)
@st.cache_resource(show_spinner="Analyzing document...")
def process_pdf(file_content):
    # Save uploaded file to a temporary location
    with open("temp.pdf", "wb") as f:
        f.write(file_content)
    
    # Load and split the PDF into small chunks
    loader = PyPDFLoader("temp.pdf")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)
    
    # Create Embeddings for free using HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Store chunks in a searchable vector database
    return FAISS.from_documents(chunks, embeddings)

# 5. The RAG Chat Interface
if uploaded_file:
    # Build the database
    vector_db = process_pdf(uploaded_file.getbuffer())
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if query := st.chat_input("Ask a question about the contract..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            # Link to Groq Llama 3
            llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")
            
            # Create the "Search + Answer" chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm, 
                retriever=vector_db.as_retriever(search_kwargs={"k": 3})
            )
            
            # Execute the RAG process
            response = qa_chain.invoke(query)
            answer = response["result"]
            st.write(answer)
            
            # Show the exact source for transparency
            with st.expander("View Source Material"):
                docs = vector_db.similarity_search(query)
                for i, doc in enumerate(docs[:2]):
                    st.markdown(f"**Source {i+1}:**\n{doc.page_content}...")

        st.session_state.messages.append({"role": "assistant", "content": answer})
