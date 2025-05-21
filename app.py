import streamlit as st
import pandas as pd
import os
import json
from docx import Document
from PyPDF2 import PdfReader
from rag_chatbot import RAGChatbot

# Set page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the chatbot
@st.cache_resource
def get_chatbot():
    return RAGChatbot()

chatbot = get_chatbot()

# Sidebar
with st.sidebar:
    st.title("About")
    st.markdown("""
    This is a RAG-based chatbot that uses:
    - LangChain for RAG implementation
    - Grok's Llama model for generation
    - HuggingFace embeddings for semantic search
    - Streamlit for the web interface
    """)
    
    # File upload section
    st.markdown("### Upload Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload files to add to the knowledge base",
        type=['txt', 'feather', 'docx', 'pdf', 'csv', 'json', 'xlsx'],
        accept_multiple_files=True,
        help="Upload text files, feather files, Word documents, PDFs, CSVs, JSON files, or Excel files to add to the chatbot's knowledge base"
    )
    
    if uploaded_files:
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        for uploaded_file in uploaded_files:
            # Save the uploaded file
            file_path = os.path.join("data", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Show preview based on file type
            if uploaded_file.name.endswith('.feather'):
                df = pd.read_feather(file_path)
                st.markdown(f"### Preview: {uploaded_file.name}")
                st.dataframe(df.head())
            elif uploaded_file.name.endswith('.docx'):
                doc = Document(file_path)
                preview_text = "\n".join([paragraph.text for paragraph in doc.paragraphs[:5]])
                st.markdown(f"### Preview: {uploaded_file.name}")
                st.text(preview_text)
            elif uploaded_file.name.endswith('.pdf'):
                pdf = PdfReader(file_path)
                preview_text = ""
                for page in pdf.pages[:2]:  # Show first 2 pages
                    preview_text += page.extract_text() + "\n"
                st.markdown(f"### Preview: {uploaded_file.name}")
                st.text(preview_text[:1000] + "..." if len(preview_text) > 1000 else preview_text)
            elif uploaded_file.name.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    preview_text = f.read()[:500] + "..." if len(f.read()) > 500 else f.read()
                st.markdown(f"### Preview: {uploaded_file.name}")
                st.text(preview_text)
            elif uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(file_path)
                st.markdown(f"### Preview: {uploaded_file.name}")
                st.dataframe(df.head())
            elif uploaded_file.name.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                st.markdown(f"### Preview: {uploaded_file.name}")
                st.json(data)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(file_path)
                st.markdown(f"### Preview: {uploaded_file.name}")
                st.dataframe(df.head())
            
            st.success(f"File {uploaded_file.name} uploaded successfully!")
        
        st.info("The chatbot will automatically use the new knowledge base in the next interaction.")
    
    st.markdown("### Sample Questions")
    st.markdown("""
    - What is the company's mission?
    - Who founded the company?
    - What services do you offer?
    """)

# Main chat interface
st.title("🤖 RAG Chatbot")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get chatbot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chatbot.get_response(prompt)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response}) 