import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import requests

# Load HuggingFace sentence transformer
model = SentenceTransformer('all-MiniLM-L6-v2')

# Gemini API config
GEMINI_API_KEY = "AIzaSyBrSJaVpiBFH0sn-HexjmFcjceyRRJ--vE"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"

# Functions
def get_embeddings(texts):
    return model.encode(texts)

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def ask_gemini(prompt):
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    response = requests.post(GEMINI_API_URL, headers=headers, json=data)
    result = response.json()
    try:
        return result['candidates'][0]['content']['parts'][0]['text']
    except:
        return "Error: Unable to generate response. Please check your API key or input."

def split_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

# UI
st.title("📖 Talk with Quran (Gemini API)")
uploaded_pdf = st.file_uploader("Upload your Quran PDF", type="pdf")

if uploaded_pdf:
    pdf = PdfReader(uploaded_pdf)
    full_text = ""
    for page in pdf.pages:
        content = page.extract_text()
        if content:
            full_text += content

    st.success("✅ PDF loaded successfully!")

    # Chunking
    chunks = split_text(full_text)
    st.write(f"Total Chunks: {len(chunks)}")

    # Embedding and Indexing
    with st.spinner("Indexing text..."):
        embeddings = get_embeddings(chunks)
        faiss_index = build_faiss_index(np.array(embeddings))

    query = st.text_input("Ask any question from Quran PDF:")

    if query:
        query_embedding = model.encode([query])
        _, indices = faiss_index.search(np.array(query_embedding), k=2)
        matched_chunks = [chunks[i] for i in indices[0]]

        context = "\n\n".join(matched_chunks)
        prompt = f"Answer the following question based on the context below:\n\nContext:\n{context}\n\nQuestion: {query}"

        response = ask_gemini(prompt)
        st.subheader("Answer:")
        st.write(response)

        with st.expander("📄 Reference Chunks"):
            for idx, chunk in enumerate(matched_chunks, 1):
                st.markdown(f"**Chunk {idx}:** {chunk}")
