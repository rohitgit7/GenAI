#%pip install -q openai numpy faiss-cpu --upgrade
import os
from getpass import getpass
import re
import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types


# API Key Setup - uses environment variable or prompts for input
if not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = getpass("Enter your GEMINI API key: ")
    
# Client and Model Configuration
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
#MODEL = "gemini-1.5-pro"
EMBEDDING_MODEL = "gemini-embedding-001"

def chunk_text(text, window_size=5, stride=2):
    """Split text into overlapping chunks using a sliding window over sentences."""
    sentences = re.split('(?<=[.!?]) +', text.strip()) # Split text into sentences based on punctuation
    chunks = []
    for i in range(0, len(sentences) - window_size + 1, stride): # Loop through the sentences and create chunks
        chunks.append(' '.join(sentences[i:i + window_size])) # Join the sentences into a chunk
    return chunks

# Load the article from a file instead of pasting it inline
with open("data/batman_history.md") as f:
    text = f.read()

chunks = chunk_text(text, window_size=4, stride=1)
print(f"Created {len(chunks)} chunks from the article\n")

# Preview the first chunk
print("Example chunk:")
print(chunks[0])

def get_embedding(text: str) -> np.ndarray:
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,  # ✅ use current model
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT"
        )
    )

    vector = response.embeddings[0].values
    return np.array(vector, dtype="float32")

# Embed all chunks
vectors = np.vstack([get_embedding(chunk) for chunk in chunks])
print(f"Embedded {vectors.shape[0]} chunks into {vectors.shape[1]}-dimensional vectors")

# Build a FAISS index for fast similarity search
faiss.normalize_L2(vectors)
index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)
print(f"FAISS index built with {index.ntotal} vectors")

print(chunks)

# 1. Embed the query
# 2. Search the index and return the top k results

def vector_search(query_text: str, chunks: list[str], k: int = 3):
    query_vector = get_embedding(query_text).reshape(1, -1)
    faiss.normalize_L2(query_vector)

    distances, indices = index.search(query_vector, k)

    relevant_chunks = "\n".join([chunks[i] for i in indices[0]])

    return distances, indices, relevant_chunks

print(vector_search("batman", chunks, k=3))

def search_and_chat(question, chunks, k=3):
    distances, indices, relevant_chunks = vector_search(question, chunks, k)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""
Answer the user's question based only on the context below.
If the context does not contain the answer, say "I don't know".

Context:
{relevant_chunks}

Question:
{question}
"""
    )

    answer = response.text

    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")

    return answer

# search_and_chat("Who is the actor who played Batman in the 2022 movie?", chunks, k=3)
search_and_chat("Who is James Phoenix?", chunks, k=3)
search_and_chat("What high tech equipment used by Batman?", chunks, k=3)











