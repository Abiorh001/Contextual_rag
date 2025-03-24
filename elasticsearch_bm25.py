import elasticsearch
from elasticsearch import Elasticsearch, helpers
import time
from typing import List, Dict, Any
import backoff
import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import logging
logging.basicConfig(level=logging.INFO)


class ElasticsearchBM25:
    """
    A class for Elasticsearch BM25. This class is responsible for creating the index, deleting the index, and indexing the documents.
    BM25 uses TF-IDF to score the documents.
    """
    def __init__(self, index_name: str):
        """
        Initialize the ElasticsearchBM25 class.
        """
        self.index_name = index_name
        # Add retry attempts and timeout settings
        self.es = Elasticsearch(
            ["http://localhost:9200"],
            retry_on_timeout=True,
            max_retries=3,
            timeout=30
        )
        
    @backoff.on_exception(
        backoff.expo,
        (elasticsearch.ConnectionError, elasticsearch.ConnectionTimeout),
        max_tries=5
    )
    def delete_index(self) -> None:
        """Delete index with retry logic"""
        logging.info(f"Deleting index {self.index_name}")
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)

    def create_index(self) -> None:
        """Create index with retry logic"""
        logging.info(f"Creating index {self.index_name}")
        if not self.es.indices.exists(index=self.index_name):
            settings = {
                "settings": {
                    # "index": {
                    #     "number_of_shards": 1,
                    #     "number_of_replicas": 0
                    # },
                    "analysis": {"analyzer": {"default": {"type": "english"}}},
                    "similarity": {
                        "default": {
                            "type": "BM25"
                        }
                    },
                    "index.queries.cache.enabled": False # Disable query caching
                },
                "mappings": {
                    "properties": {
                        
                        "title": {"type": "keyword", "index": False},
                        "chunked_content": {"type": "text", "analyzer": "english"},
                        "contextualized_content": {"type": "text", "analyzer": "english"},
                       "doc_id": {"type": "keyword", "index": False},
                    }
                }
            }
            self.es.indices.create(index=self.index_name, body=settings)
            logging.info(f"Index {self.index_name} created")
    

    @backoff.on_exception(
        backoff.expo,
        (elasticsearch.ConnectionError, elasticsearch.ConnectionTimeout),
        max_tries=5
    )
    def index_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Index documents with retry logic"""
        try:
            # First create the index
            self.create_index()
            
            # Using bulk indexing with progress tracking
            actions = [
                {
                    "_index": self.index_name,
                    "_source": {

                        "doc_id": doc["doc_id"],
                        "title": doc["title"],
                        "chunked_content": doc["chunked_content"],
                        "contextualized_content": doc["contextualized_content"]
                    }
                }
                for doc in documents
            ]
            
            success, _ = helpers.bulk(self.es, actions)
            self.es.indices.refresh(index=self.index_name)
            logging.info(f"Index {self.index_name} refreshed")
            return success
                
        except Exception as e:
            print(f"Error indexing documents: {str(e)}")
            # Delete index if it exists and indexing failed
            if self.es.indices.exists(index=self.index_name):
                self.es.indices.delete(index=self.index_name)
            raise

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Search with retry logic"""
        try:
            self.es.indices.refresh(index=self.index_name)
            search_body={
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["contextualized_content", "chunked_content"]
                        }
                    },
                    "size": top_k
                }
            response = self.es.search(
                index=self.index_name,
                body=search_body
            )
            return [
                {
                    "doc_id": hit["_source"]["doc_id"],
                    "contextualized_content": hit["_source"]["contextualized_content"],
                    #"chunked_content": hit["_source"]["chunked_content"],
                    "score": hit["_score"]
                }
                for hit in response["hits"]["hits"]
            ]
        except Exception as e:
            print(f"Search error: {str(e)}")
            return []
    
    def rerank_search_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rearnk the results based on the doc_id and chunk_id"""
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        # remove the score from the results
        results = [{"contextualized_content": result["contextualized_content"]} for result in results]
        return results
