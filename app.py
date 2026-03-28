import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="LegalGuard AI", page_icon="⚖️", layout="wide")
st.title("⚖️ LegalGuard AI (High-Speed)")

# 2. Access Secret Key with Error Handling
try:
    # Ensure this matches exactly what you typed in Streamlit Secrets
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    st.error("⚠️ API Key Error: Please check your Streamlit Secrets.")
    st.stop()

# 3. Load the Embedding Model (Free & Local)
@st.cache_resource
def load_embed_model():
    # This model is small, fast, and perfect for legal/academic text
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_embed_model()

# 4. PDF Processing Logic
def process_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
            
    if not text.strip():
        return None, None
    
    # Split text into chunks (1000 characters with 200 character overlap)
    chunks = [text[i:i+1000] for i in range(0, len(text), 800)]
    
    # Convert chunks to numbers (embeddings)
    embeddings = embed_model.encode(chunks)
    
    # Build a Search Index using FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    return chunks, index

# 5. Sidebar & Upload
with st.sidebar:
    st.header("Upload Center")
    uploaded_file = st.file_uploader("Upload Document (PDF)", type="pdf")
    st.markdown("---")
    st.caption("Powered by Groq Llama 3.1 & FAISS")

# 6. Chat Logic
if uploaded_file:
    chunks, index = process_pdf(uploaded_file)
    
    if chunks is None:
        st.error("The uploaded PDF seems to be empty or an image. Please upload a text-based PDF.")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display Chat History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if query := st.chat_input("Ask about the document..."):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                # A. Search the PDF for the most relevant context
                query_embedding = embed_model.encode([query])
                D, I = index.search(np.array(query_embedding).astype('float32'), k=3)
                context = " ".join([chunks[i] for i in I[0] if i < len(chunks)])

                # B. Generate Answer using Groq
                try:
                    response = client.chat.completions.create(
                        messages=[
                            {
                                "role": "system", 
                                "content": f"You are a helpful assistant. Answer the user's question based ONLY on the following context: {context}. If the answer isn't in the context, say you don't know."
                            },
                            {"role": "user", "content": query}
                        ],
                        # Updated to the most stable Groq model
                        model="llama-3.1-8b-instant",
                        temperature=0.2 # Lower temperature for higher accuracy
                    )
                    
                    answer = response.choices[0].message.content
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    st.error(f"Groq API Error: {str(e)}")
else:
    st.info("Please upload a PDF document in the sidebar to begin.")
