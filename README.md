# RAG Chatbot with LangChain and Grok
![image](https://github.com/user-attachments/assets/b0d1d23f-a01a-42d9-a1d6-1d41cc6627a5)

![image](https://github.com/user-attachments/assets/1a20ea6b-5ab7-4882-8dd9-95bd6d490885)



A powerful RAG (Retrieval-Augmented Generation) chatbot that uses LangChain for document processing and Grok's Llama model for generating responses. The chatbot can process various file formats and provide intelligent responses based on the content.

## Features

- **Multiple File Format Support**: Process PDF, DOCX, TXT, CSV, JSON, Excel, and Feather files
- **RAG Implementation**: Uses LangChain for efficient document retrieval and processing
- **Grok Integration**: Powered by Grok's Llama model for high-quality responses
- **Modern UI**: Streamlit-based web interface for easy interaction
- **Document Processing**: Automatic chunking and embedding of documents
- **Conversation Memory**: Maintains context across multiple interactions

## Prerequisites

- Python 3.8 or higher
- Grok API key
- Sufficient disk space for model downloads

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd langchainproject
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your Grok API key:
```
GROQ_API_KEY=your_api_key_here
```

## Project Structure

```
langchainproject/
├── app.py                 # Streamlit web interface
├── streamlit_app.py       # Main entry point for Streamlit Cloud
├── rag_chatbot.py         # Core RAG implementation
├── requirements.txt       # Project dependencies
├── packages.txt          # System dependencies for Streamlit Cloud
├── .env                  # Environment variables
└── data/                 # Knowledge base directory
    ├── ai_concepts.txt   # Sample knowledge base
    └── ...              # Other documents
```

## Usage

### Local Development

1. Start the Streamlit web interface:
```bash
streamlit run app.py
```

2. Access the web interface at `http://localhost:8501`

3. Upload documents to the knowledge base using the file uploader in the sidebar

4. Start chatting with the bot about the uploaded documents

### Streamlit Cloud Deployment

1. Create a GitHub repository and push your code:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-github-repo-url>
git push -u origin main
```

2. Go to [Streamlit Cloud](https://streamlit.io/cloud)

3. Sign in with your GitHub account

4. Click "New app"

5. Select your repository, branch, and main file path:
   - Repository: your-github-username/your-repo-name
   - Branch: main
   - Main file path: streamlit_app.py

6. Add your secrets:
   - Click "Advanced settings"
   - Add your Grok API key:
     - Key: GROQ_API_KEY
     - Value: your-api-key-here

7. Click "Deploy"

Your app will be deployed and accessible via a public URL.

## Supported File Types

- Text files (.txt)
- PDF documents (.pdf)
- Word documents (.docx)
- CSV files (.csv)
- JSON files (.json)
- Excel files (.xlsx)
- Feather files (.feather)

## Features in Detail

### Document Processing
- Automatic text extraction from various file formats
- Smart chunking with configurable size and overlap
- Metadata preservation for better context

### RAG Pipeline
- Document loading and preprocessing
- Text splitting and chunking
- Embedding generation using HuggingFace models
- Vector storage using Chroma
- Semantic search for relevant context

### Chat Interface
- Real-time response generation
- Conversation history tracking
- File upload and preview
- Responsive design

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- LangChain for the RAG implementation
- Grok for the language model
- Streamlit for the web interface
- HuggingFace for the embeddings model 
