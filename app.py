import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

st.header("📖 Talk with Quran (Gemini API)")

# Initialize session state variables
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

pdf = st.file_uploader("Upload your Quran PDF", type="pdf")

if pdf is not None:
    st.write("Loading PDF...")
    try:
        # Read the PDF file
        pdf_reader = PdfReader(pdf)
        full_text = ""
        for page in pdf_reader.pages[:50]:  # Limiting to first 50 pages
            text = page.extract_text()
            if text:  # Only add if text was extracted
                full_text += text

        if not full_text.strip():
            st.error("No text could be extracted from the PDF.")
            st.stop()

        st.success("✅ PDF loaded successfully!")

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        chunks = text_splitter.split_text(full_text)

        # Initialize Google Gemini Embeddings
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            st.error("Please set the GOOGLE_API_KEY environment variable.")
            st.stop()
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        st.session_state.vectorstore = FAISS.from_texts(chunks, embeddings)
        st.success("✅ Text chunked and embedded successfully!")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.stop()

query = st.text_input("Ask any question from Quran PDF:")
if query and st.session_state.vectorstore is not None:
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            st.error("Please set the GOOGLE_API_KEY environment variable.")
            st.stop()

        similar_chunks = st.session_state.vectorstore.similarity_search(query, k=2)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",  # Updated model name
            google_api_key=google_api_key,
            temperature=0.1
        )
        chain = load_qa_chain(llm=llm, chain_type="stuff")

        with st.spinner("Generating answer..."):
            response = chain.run(input_documents=similar_chunks, question=query)
            
        st.subheader("Answer:")
        st.write(response)
        
        st.subheader("Reference Verses:")
        for i, doc in enumerate(similar_chunks, 1):
            st.markdown(f"**Reference {i}:**")
            st.write(doc.page_content)
            st.write("---")

    except Exception as e:
        st.error(f"An error occurred while generating the answer: {str(e)}")

