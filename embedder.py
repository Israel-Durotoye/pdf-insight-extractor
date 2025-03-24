import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
# import faiss
import numpy as np
import os
import pickle

chroma_client = chromadb.PersistentClient(path = "chroma_db")

chroma_client.delete_collection(name = "rag_docs")
collection = chroma_client.get_or_create_collection(name = "rag_docs", metadata = {"hnsw:space": "cosine"})

# embedding_model = OpenAIEmbeddings(model = "text-embedding-3-large", openai_api_key = OPENAI_API_KEY)
embedding_model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")

def chunk_text(text, chunk_size=2048, overlap=512):
    """Chunk text into meaningful segments for retrieval."""
    print(f"📄 Original Text Length: {len(text)} characters")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = splitter.split_text(text)

    print(f"🔹 Total Chunks Created: {len(chunks)}")
    for i, chunk in enumerate(chunks[:5]):  # Print first 5 chunks
        print(f"📄 Chunk {i}: {chunk[:150]}...")  # Print first 150 characters for debugging

    return chunks

def store_chunks_and_embeddings(chunks):
    """Store document chunks in ChromaDB."""
    embeddings = embedding_model.encode(chunks)

    print(f"✅ Storing {len(chunks)} chunks in ChromaDB...")

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        print(f"📄 Chunk {i}: {chunk[:100]}")  # Print first 100 characters
        collection.add(
            ids=[f"doc_{i}"],
            embeddings=[embedding.tolist()],
            documents=[chunk]
        )

    print("✅ All chunks stored successfully!")

reranker = FlagReranker("BAAI/bge-reranker-large")

def rerank_results(query, retrieved_chunks):
    if not retrieved_chunks:
        return ["No relevant information found."]
    
    scores = reranker.compute_score([(query, chunk) for chunk in retrieved_chunks])
    ranked_chunks = [chunk for _, chunk in sorted(zip(scores, retrieved_chunks), reverse = True)]

    print("Ranked Chunks:", ranked_chunks)
    return ranked_chunks

import yake

def extract_keywords(text, max_keywords = 5):
    # doc = nlp(text)
    kw_extractor = yake.KeywordExtractor(n = 1, top = max_keywords)
    keywords = kw_extractor.extract_keywords(text)
    # return [token.lemma_ for token in doc if token.pos_ in ["NOUN", "PROPN", "VERB"]]
    return [kw[0] for kw in keywords]

def hybrid_search(query, top_k = 5):
    """Retrieve and rerank relevant chunks."""
    query_embedding = embedding_model.encode([query])[0]

    # ✅ Retrieve relevant chunks
    results = collection.query(
        query_embeddings = [query_embedding.tolist()],
        n_results = top_k
    )

    retrieved_chunks = results["documents"][0] if results["documents"] else []

    print(f"🔍 Query: {query}")
    print(f"📌 Retrieved Chunks (Before Reranking): {retrieved_chunks}")

    # ✅ Fix: Prevent Empty List Issue
    if not retrieved_chunks:
        return ["No relevant information found."]

    # ✅ Format input for reranker
    rerank_inputs = [(query, chunk) for chunk in retrieved_chunks]
    
    # ✅ Fix: Ensure reranker does not receive an empty list
    if not rerank_inputs:
        return ["No relevant information found."]

    scores = reranker.compute_score(rerank_inputs)
    ranked_chunks = [chunk for _, chunk in sorted(zip(scores, retrieved_chunks), reverse=True)]

    print(f"🔍 Ranked Chunks: {ranked_chunks}")

    return ranked_chunks[:3] if ranked_chunks else ["No relevant information found."]

def check_chromadb_content():
    """Check if ChromaDB contains stored chunks."""
    stored_docs = collection.get()
    print("📌 Stored Document IDs:", stored_docs.get("ids", []))
    print("📄 Stored Documents (First 3):", stored_docs.get("documents", [])[:3]) 

if __name__ == "__main__":
    sample_text = """
                    But sadly, when self-awareness researchers finally had the chance to catch up,
                    they made many of the same mistakes the Mayan archeologists did, spending
                    years focused on surprisingly myopic details at the expense of bigger, more
                    important questions. The result? Piles of disjointed, often peripheral research that
                    no one even bothered trying to stitch together. So when I set out to summarize the
                    current state of scientific knowledge on self-awareness, I initially came up with
                    more questions than answers, starting with the most central question: What was
                    self-awareness, exactly?
                    """
    chunks = chunk_text(sample_text)
    store_chunks_and_embeddings(chunks)
    check_chromadb_content()
    results = hybrid_search("What is the document about?")
    print(f"Top matches: {results}")