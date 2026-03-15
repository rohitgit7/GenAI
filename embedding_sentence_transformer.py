'''
import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="hf-inference",
    api_key=os.getenv("HF_FIRST_API_KEY")
)
'''

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("Apple")
print(embedding[:50])
print(f"Length: {len(embedding)}")
