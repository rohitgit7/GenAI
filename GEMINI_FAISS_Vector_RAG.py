import os
import numpy as np
import faiss
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize client
client = genai.Client(api_key=GOOGLE_API_KEY)
EMBEDDING_MODEL = "gemini-embedding-001"


def embed_text(text: str):
    """
    Generate embedding using Google GenAI SDK
    """
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT"
        )
    )

    #Handle both possible SDK response formats
    if hasattr(response, "embeddings"):
        vector = response.embeddings[0].values
    else:
        vector = response.embedding.values

    vector = np.array(vector, dtype="float32")

    # Normalize for cosine similarity
    faiss.normalize_L2(vector.reshape(1, -1))
    print (vector)

    return vector


# Sample documents
documents = [
    "FAISS is a library for efficient similarity search.",
    "Google Gemini provides embedding APIs.",
    "Vector databases are useful for semantic search.",
    "Machine learning enables AI applications."
]

# Generate embeddings
embeddings = np.vstack([embed_text(doc) for doc in documents])

#Use cosine similarity (Inner Product + normalized vectors)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print(f"Stored {index.ntotal} vectors in FAISS.")

# Query
query = "How does semantic search work?"
query_embedding = embed_text(query).reshape(1, -1)

k = 2
scores, indices = index.search(query_embedding, k)

print("\nQuery:", query)
print("\nTop matches:")
for score, i in zip(scores[0], indices[0]):
    print(f"- {documents[i]} (score: {score:.4f})")