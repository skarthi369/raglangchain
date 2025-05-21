import streamlit as st
from rag_chatbot import RAGChatbot
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
    st.title("🤖 RAG Chatbot")
    st.markdown("""
    This chatbot uses RAG (Retrieval-Augmented Generation) to provide intelligent responses based on your documents.
    
    ### Features:
    - Process multiple file formats (PDF, DOCX, TXT, CSV, JSON, Excel)
    - Smart document chunking and retrieval
    - Context-aware responses
    - Conversation memory
    
    ### Sample Questions:
    - What is Machine Learning?
    - Explain Deep Learning
    - How does RAG work?
    """)
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=['txt', 'pdf', 'docx', 'csv', 'json', 'xlsx', 'feather'],
        accept_multiple_files=True,
        help="Upload files to add to the knowledge base"
    )
    
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} files")
        for file in uploaded_files:
            st.write(f"- {file.name}")

# Main chat interface
st.title("Chat with your documents")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get chatbot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chatbot.get_response(prompt)
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response}) 