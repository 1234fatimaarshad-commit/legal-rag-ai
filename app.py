import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader
import numpy as np

st.set_page_config(page_title="LegalGuard Lite", layout="wide")
st.title("⚖️ LegalGuard AI (High-Speed)")

# 1. Access Secret Key
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except:
    st.error("Missing GROQ_API_KEY in Secrets!")
    st.stop()

# 2. Load the Embedding Model (Free & Local)
@st.cache_resource
def load_embed_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_embed_model()

# 3. PDF Processing Logic
def process_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    
    # Split text into chunks manually
    chunks = [text[i:i+1000] for i in range(0, len(text), 800)]
    
    # Convert chunks to numbers (embeddings)
    embeddings = embed_model.encode(chunks)
    
    # Build a Search Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    return chunks, index

# 4. Sidebar & Upload
with st.sidebar:
    uploaded_file = st.file_uploader("Upload Contract", type="pdf")

if uploaded_file:
    chunks, index = process_pdf(uploaded_file)
    
    if query := st.chat_input("Ask about the legal terms..."):
        # A. Search the PDF for the answer
        query_embedding = embed_model.encode([query])
        D, I = index.search(np.array(query_embedding).astype('float32'), k=2)
        context = " ".join([chunks[i] for i in I[0]])

        # B. Ask Groq using the context
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": f"Answer based ONLY on this text: {context}"},
                {"role": "user", "content": query}
            ],
            model="llama3-8b-8192",
        )
        
        st.chat_message("user").write(query)
        st.chat_message("assistant").write(response.choices[0].message.content)
