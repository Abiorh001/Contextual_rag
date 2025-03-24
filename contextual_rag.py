from dotenv import load_dotenv
import os
import chromadb
import openai
import voyageai
import csv
import time
from typing import List, Dict, Any
import backoff
import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import logging
import pickle
import openai
import json
import cohere
from datetime import datetime
from elasticsearch_bm25 import ElasticsearchBM25
from sentence_transformers import CrossEncoder
from tqdm import tqdm
logging.basicConfig(level=logging.INFO)
load_dotenv()

voyage_api_key = os.getenv("VOYAGE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
voyage_client = voyageai.Client(api_key=voyage_api_key)
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY)
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

DOCUMENT_CONTEXT_PROMPT = """
<document>
{doc_content}
</document>
"""

CHUNK_CONTEXT_PROMPT = """
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
"""
class ContextualizedRAG:
    """
    The ContextualizedRAG class is responsible for implementing a Retrieval-Augmented Generation (RAG) pipeline 
    with contextualized document chunking, hybrid search, and reranking using multiple AI models.

    """
    def __init__(self, collection_name: str):
        """
        Initialize the ContextualizedRAG class.
        """
        self.collection_name = collection_name
        self.chromadb_client = chromadb.HttpClient(host="localhost", port=8000)
        self.openai_client = openai_client
        self.voyage_client = voyage_client
        self.cohere_client = cohere_client
        self.cross_encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.esbm25 = ElasticsearchBM25(index_name=collection_name)
    
    def document_chunking(self, file_path: str, chunk_size: int = 400, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
        """
        Chunk the documents into smaller chunks of 400 characters with 200 characters overlap.
        """
        documents = []
        with open(file_path, 'r', encoding='utf-8') as f:
            doc_id = 0
            reader = csv.reader(f)
            # skip the header
            next(reader)
            for row in reader:
                if len(row) >= 3:
                    textsplitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    texts = textsplitter.split_text(row[2])
                    chunk_id = 0
                    for text in texts:
                        doc = {
                            "doc_id": doc_id,
                            "title": row[1],
                            "chunked_content": text,
                            "full_content": row[2],
                            "chunk_id": chunk_id
                        }
                        documents.append(doc)
                        doc_id += 1
                        chunk_id += 1
        return documents
        
        
    def generate_response(
        self,
        prompt1: str,
        prompt2: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0,
        max_tokens: int = 2000,
    ):
        while True:
            try:
                messages = [
                    {
                        "role": "user",
                        "content": prompt1
                    },
                    {
                        "role": "user",
                        "content": prompt2
                    }
                ]
                response = openai_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=1,
                    stop=None,
                )
                return response.choices[0].message.content.strip(), response
            except Exception as e:
                print(f"Error: {e}")
                print("Unable to connect to internal LLM API")



    def contextualize_documents(self, documents: List[Dict[str, Any]], save_file_path: str):
        for doc in documents:
            llm_response, _ = self.generate_response(
                prompt1=DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc["full_content"]),
                prompt2=CHUNK_CONTEXT_PROMPT.format(chunk_content=doc["chunked_content"]),
                model="gpt-4o-mini",
                temperature=0.5,
                max_tokens=400
            )
            doc["contextualized_content"] = f"{llm_response}\n\n{doc['chunked_content']}"
            del doc["full_content"]

        # save the documents to a new csv
        with open(save_file_path, "w", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["doc_id", "title", "chunked_content", "contextualized_content", "chunk_id"])
            for doc in documents:
                writer.writerow([doc["doc_id"], doc["title"], doc["chunked_content"], doc["contextualized_content"], doc["chunk_id"]])




    def read_contextualized_csv(self, contextualized_file_path: str) -> List[Dict[str, Any]]:
        """
        Read the new csv file with the contextualized content.
        """
        documents = []
        with open(contextualized_file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                chunked_content = row[2]
                doc_id = row[0]
                title = row[1]
                contextualized_content = row[3]
                
                documents.append({
                    "doc_id": doc_id,
                    "title": title,
                    "chunked_content": chunked_content,
                    "contextualized_content": contextualized_content
                })
        return documents
    def load_documents_to_esbm25(self, documents: List[Dict[str, Any]]):
        """
        Load documents to Elasticsearch BM25.
        """
        self.esbm25.index_documents(documents)
        
        
    def create_embeddings_store_temp(self, documents: List[Dict[str, Any]], embeddings_file_path: str):
        """
        Create embeddings with OpenAI and store them in a pickle file.
        """
        embeddings_results = []
        for doc in tqdm(documents, desc="Creating embeddings"):
            response = openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=doc["contextualized_content"]
            )
            
            embedding = response.data[0].embedding
            embeddings_results.append(embedding)

        # Save final embeddings   
        with open(embeddings_file_path, "wb") as f:
            pickle.dump(embeddings_results, f)
        print(f"\nCompleted embedding creation for {len(embeddings_results)} chunks")
    

    def load_embeddings_save_chromadb(self, documents: List[Dict[str, Any]], embeddings_file_path: str):
        """
        Load embeddings from a pickle file and store them in ChromaDB.
        """
        with open(embeddings_file_path, "rb") as f:
            embeddings = pickle.load(f)
            
        doc_ids = [doc["doc_id"] for doc in documents]
        titles = [doc["title"] for doc in documents]
        chunked_contents = [doc["chunked_content"] for doc in documents]
        contextualized_contents = [doc["contextualized_content"] for doc in documents]
        metadata = [{"title": title, "chunked_content": chunked_content, "contextualized_content": contextualized_content} 
                for title, chunked_content, contextualized_content in zip(titles, chunked_contents, contextualized_contents)]


        # Delete existing collection if it exists
        try:
            self.chromadb_client.delete_collection(name=self.collection_name)
            print(f"Deleted existing collection: {self.collection_name}")
        except Exception as e:
            print(f"No existing collection to delete: {e}")

        # Create new collection with proper settings for openai-3 model (1536 dimensions)
        collection = self.chromadb_client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",  # Use cosine similarity for better matching
                "dimension": 1536
            }
        )
        print(f"Created new collection: {self.collection_name} with dimension 1536 (openai-3 model)")
    
        
        try:
            # Add the batch to ChromaDB
            collection.add(
                ids=doc_ids,
                documents=contextualized_contents,
                metadatas=metadata,
                embeddings=embeddings
            )
            print(f"Successfully added batch to ChromaDB with {len(doc_ids)} items")
        except Exception as e:
            print(e)
            raise e


    def process_hybrid_search(self, query: str, top_vector_results: int = 10, top_bm25_results: int = 10):

        # Generate embedding for the query
        response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=query
        )

        query_embedding = response.data[0].embedding

        # Query ChromaDB with the correct embedding
        collection = self.chromadb_client.get_or_create_collection(
        name=self.collection_name,
        metadata={"hnsw:space": "cosine", "dimension": 1536}
        )
        vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_vector_results
        )
        vector_documents = vector_results.get("documents")[0]
        vector_documents = [doc for doc in vector_documents]

        bm25_results = self.esbm25.search(query, top_k=top_bm25_results)
        bm25_sorted_results = self.esbm25.rerank_search_results(bm25_results)

        hybrid_results = {result["contextualized_content"] for result in bm25_sorted_results}
        hybrid_results.update(vector_documents)

        hybrid_results = list(enumerate(hybrid_results))
       
        seen = set()
        unique_hybrid_results = []
        for idx, content in hybrid_results:
            if content not in seen:
                seen.add(content)
                unique_hybrid_results.append((idx, content))

        return unique_hybrid_results

    def custom_reranking(self, query: str, documents: List[Dict[str, Any]], top_n: int = 5):
        documents_list = [doc[1] for doc in documents]
        if documents_list:
            # Get scores for all query-document pairs
            scores = self.cross_encoder_model.predict([(query, doc) for doc in documents_list])
            
            # Pair scores with their indices and sort
            scored_docs = list(enumerate(scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Get top 8 results
            top_indices = [idx for idx, _ in scored_docs[:top_n]]
            custom_reranked_results = [documents[idx][1] for idx in top_indices]
        else:
            custom_reranked_results = []

        return custom_reranked_results

    def cohere_reranking(self, query: str, documents: List[Dict[str, Any]], top_n: int = 5):
        documents_list = [doc[1] for doc in documents]
        if documents_list:
            time.sleep(3)
            
        response = self.cohere_client.rerank(
        model="rerank-v3.5",
        query=query,
        documents=documents_list,
        top_n=top_n,
        )

        cohere_reranked_results = [documents[item.index][1] for item in response.results]
        return cohere_reranked_results if cohere_reranked_results else []
        

    def deduplication(self, documents: List[Dict[str, Any]]):
       result = set()
       for doc in documents:
           if doc not in result:
               result.add(doc)
       return list(result)
