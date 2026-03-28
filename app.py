import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="LegalGuard AI", page_icon="⚖️", layout="wide")
st.title("⚖️ Legal RAG AI ")

# 2. Access Secret Key
try:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception as e:
    st.error("⚠️ API Key Error: Check Streamlit Secrets.")
    st.stop()

# 3. Load Embedding Model
@st.cache_resource
def load_embed_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_embed_model()

# 4. PDF Processing
def process_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content: text += content
    if not text.strip(): return None, None
    
    chunks = [text[i:i+1000] for i in range(0, len(text), 800)]
    embeddings = embed_model.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return chunks, index

# 5. Sidebar
with st.sidebar:
    st.header("Upload Center")
    uploaded_file = st.file_uploader("Upload Document (PDF)", type="pdf")
    if st.button("Clear Chat Memory"):
        st.session_state.messages = []
        st.rerun()

# 6. Chat Logic with Memory
if uploaded_file:
    chunks, index = process_pdf(uploaded_file)
    
    if chunks is None:
        st.error("PDF is empty or unreadable.")
    else:
        # Initialize Memory
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if query := st.chat_input("Ask about the document..."):
            # A. Find Context from PDF
            query_embedding = embed_model.encode([query])
            D, I = index.search(np.array(query_embedding).astype('float32'), k=3)
            context = " ".join([chunks[i] for i in I[0] if i < len(chunks)])

            # B. Construct the Memory-Aware Prompt
            # We start with a System message that includes the PDF context
            api_messages = [
                {
                    "role": "system", 
                    "content": f"You are a helpful assistant. Use this PDF context for facts: {context}. Always remember previous parts of this conversation."
                }
            ]
            
            # Add all previous chat messages to the API call
            for msg in st.session_state.messages:
                api_messages.append({"role": msg["role"], "content": msg["content"]})
            
            # Add the current user query
            api_messages.append({"role": "user", "content": query})

            # Display User Message
            with st.chat_message("user"):
                st.markdown(query)
            st.session_state.messages.append({"role": "user", "content": query})

            # C. Generate Response from Groq
            with st.chat_message("assistant"):
                try:
                    response = client.chat.completions.create(
                        messages=api_messages,
                        model="llama-3.1-8b-instant",
                        temperature=0.2
                    )
                    answer = response.choices[0].message.content
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {str(e)}")
else:
    st.info("Upload a PDF to start.")
