from contextual_rag import ContextualizedRAG

rag = ContextualizedRAG(collection_name="test")

# step 1: document_chunking
# document_chunking = rag.document_chunking(file_path="Data/test.csv")

# step 2: contextualize_documents
# contextualize_documents = rag.contextualize_documents(documents=document_chunking, save_file_path="Data/contextualized_documents.csv")

# step 3: load_documents_to_esbm25
# contextualize_documents = rag.read_contextualized_csv(contextualized_file_path="Data/contextualized_documents.csv")
# rag.load_documents_to_esbm25(documents=contextualize_documents)

# step 4: create embeddings store
# rag.create_embeddings_store_temp(documents=contextualize_documents, embeddings_file_path="Data/embeddings.pkl")

# step 5: load_embeddings_save_chromadb
# rag.load_embeddings_save_chromadb(documents=contextualize_documents, embeddings_file_path="Data/embeddings.pkl")

# step 6: process_hybrid_search
# query = "what is RAG"
# hybrid_search_results = rag.process_hybrid_search(query=query)
#print(hybrid_search_results)

# step 7: custom_reranking
# custom_reranking_results = rag.custom_reranking(query=query, documents=hybrid_search_results)
# print(custom_reranking_results)

# step 8: cohere_reranking
# cohere_reranking_results = rag.cohere_reranking(query=query, documents=hybrid_search_results)

# NOTE: yOU CAN USE EITHER OF THE RERANKING METHODS DIRECTLY WITHOUT NEED TO DEDUPLICATE
# RECOMMENDED TO USE DEDUPLICATION IF YOU ARE USING BOTH THE METHODS, COHERE IS BETTER THAN CUSTOM RERANKING
# step 9: deduplication
# documents = custom_reranking_results + cohere_reranking_results
# print(len(documents))
# final_results = rag.deduplication(documents=documents)
# print(len(final_results))



