import os
from openai import OpenAI

# Set your GEMINI_API_KEY environment variable first
client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Text to embedls env:GEMII_KEY
input_text = "apple"

try:
    # Call the embeddings API using the "gemini-embedding-001" model
    response = client.embeddings.create(
        model="gemini-embedding-001",
        input=input_text
    )

    # Extract the embedding vector
    embedding = response.data[0].embedding
    print(f"Embedding length: {len(embedding)}")
    print(f"Embedding (first 50 values): {embedding[:50]}...")

except Exception as e:
    print(f"An error occurred: {e}")
