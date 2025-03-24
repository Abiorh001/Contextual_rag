# Contextualized Hybrid RAG System

A sophisticated hybrid retrieval system that combines multiple retrieval strategies and reranking approaches, inspired by Anthropic's Contextual Retrieval RAG announcement.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

- **Hybrid Retrieval System**: Combines vector and term-based search for robust document retrieval
- **Contextual Enhancement**: Adds semantic context to each document chunk
- **Multi-Stage Reranking**: Implements sophisticated reranking pipeline
- **Dual Storage System**: Leverages both ChromaDB and Elasticsearch
- **Deduplication**: Ensures unique results across retrieval methods

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Docker and Docker Compose
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clonehttps://github.com/Abiorh001/Contextual_rag.git
cd Contextual_rag
```

2. Set up the environment:
```bash
uv venv OR python -m venv venv
source venv/bin/activate 
uv sync
```

3. Start the required services:
```bash
docker-compose up -d
```

4. Set up your environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

## 🏗️ System Architecture

### 1. Document Processing Pipeline
- **Text Splitting**: RecursiveTextSplitter implementation
  - Chunk size: 400 tokens
  - Overlap: 200 tokens
  - Strategy: ChromaDB-optimized chunking

### 2. Dual Storage System
- **ChromaDB (Vector Store)**
  - Docker-hosted vector database
  - Stores embeddings from text-embedding-ada-002
  - Maintains both original and contextualized chunks
  - Enables semantic search capabilities

- **Elasticsearch (Term-Based Search)**
  - BM25 algorithm implementation
  - Stores original and contextualized chunks
  - Enables term-based retrieval with IDF scoring

### 3. Contextual Generation
- **Model**: GPT-4o-mini
- **Process**: Generates contextual information for each chunk
- **Storage**: Maintains contextualized versions alongside originals

### 4. Hybrid Retrieval System
- **Vector Retrieval**:
  - Query embedding via text-embedding-ada-002
  - Similarity search in ChromaDB

- **Term-Based Retrieval**:
  - BM25 scoring in Elasticsearch
  - Traditional keyword matching

### 5. Reranking Pipeline
- **Primary Reranking**: Cohere rerank model
- **Secondary Reranking**: Cross-encoder/ms-marco-MiniLM-L-6-v2
- **Deduplication**: Set-based removal of duplicate results


### Basic Usage

Here's a complete example of how to use the system:

```python
from contextual_rag import ContextualizedRAG

# Initialize the RAG system
rag = ContextualizedRAG(collection_name="test")

# Step 1: Document Chunking
document_chunking = rag.document_chunking(file_path="Data/test.csv")

# Step 2: Contextualize Documents
contextualize_documents = rag.contextualize_documents(
    documents=document_chunking, 
    save_file_path="Data/contextualized_documents.csv"
)

# Step 3: Load Documents to Elasticsearch
contextualize_documents = rag.read_contextualized_csv(
    contextualized_file_path="Data/contextualized_documents.csv"
)
rag.load_documents_to_esbm25(documents=contextualize_documents)

# Step 4: Create Embeddings Store
rag.create_embeddings_store_temp(
    documents=contextualize_documents, 
    embeddings_file_path="Data/embeddings.pkl"
)

# Step 5: Load Embeddings and Save to ChromaDB
rag.load_embeddings_save_chromadb(
    documents=contextualize_documents, 
    embeddings_file_path="Data/embeddings.pkl"
)

# Step 6: Process Hybrid Search
query = "what is RAG"
hybrid_search_results = rag.process_hybrid_search(query=query)

# Step 7: Custom Reranking (Optional)
custom_reranking_results = rag.custom_reranking(
    query=query, 
    documents=hybrid_search_results
)

# Step 8: Cohere Reranking (Optional)
cohere_reranking_results = rag.cohere_reranking(
    query=query, 
    documents=hybrid_search_results
)

# Step 9: Deduplication (Optional, recommended when using both reranking methods)
documents = custom_reranking_results + cohere_reranking_results
final_results = rag.deduplication(documents=documents)
```

### Important Notes

1. **Reranking Options**:
   - You can use either custom reranking or Cohere reranking independently
   - Cohere reranking generally provides better results than custom reranking
   - Deduplication is recommended only when using both reranking methods

2. **Data Format**:
   - Input data should be in CSV format with columns: url, title, content
   - Example data is provided in `Data/test.csv`

3. **Storage**:
   - Elasticsearch is used for term-based search
   - ChromaDB is used for vector storage
   - Both services must be running via docker-compose


## 📊 Performance Metrics
# IN PROGRESS

## 🔧 Configuration

The system can be configured through environment variables or a configuration file:

```env
OPENAI_API_KEY=your_api_key
COHERE_API_KEY=your_api_key
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by Anthropic's Contextual Retrieval RAG announcement
- Built with LangChain and other open-source tools
- Special thanks to all contributors

## 📞 Support

For support, please open an issue in the GitHub repository or contact the maintainers.

---

Made with ❤️ by [Abiola Adeshina]

