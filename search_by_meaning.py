import json
import os
import numpy as np
from dotenv import load_dotenv
from google import genai

# Load API key
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found!")

# Gemini client
client = genai.Client(api_key=GOOGLE_API_KEY)

# -----------------------------
# 1. Load knowledge base
# -----------------------------
with open(
    r"C:\Users\14ARE05\Downloads\ITSkillSprint_Labs\m9-01-lab-embeddings-vector-search\knowledge_base.json",
    "r",
    encoding="utf-8"
) as f:
    knowledge_base = json.load(f)

# -----------------------------
# 2. Function to create embedding
# -----------------------------
def get_embedding(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return np.array(response.embeddings[0].values)

# -----------------------------
# 3. Create embeddings for documents
# -----------------------------
documents = []

print("Creating embeddings for knowledge base...")

for item in knowledge_base:
    embedding = get_embedding(item["text"])

    documents.append({
        "id": item["id"],
        "source": item["source"],
        "text": item["text"],
        "embedding": embedding
    })

print(f"Embedded {len(documents)} documents.\n")

# -----------------------------
# 4. Cosine similarity (by hand)
# -----------------------------
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)

    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    return dot_product / (norm_vec1 * norm_vec2)

# -----------------------------
# 5. Test queries
# -----------------------------
queries = [
    "my laptop won't switch on",
    "how do I stop being billed every month?",
    "access denied error when saving a file",
    "where do I leave my car in the evening?"
]

# Optional stretch
queries.append("what's the wifi password?")

# -----------------------------
# 6. Search
# -----------------------------
for query in queries:

    print("=" * 70)
    print(f"QUERY: {query}\n")

    query_embedding = get_embedding(query)

    scores = []

    for doc in documents:
        score = cosine_similarity(query_embedding, doc["embedding"])

        scores.append({
            "score": score,
            "id": doc["id"],
            "source": doc["source"],
            "text": doc["text"]
        })

    scores.sort(key=lambda x: x["score"], reverse=True)

    print("Top 3 Matches:\n")

    for result in scores[:3]:
        print(f"Score: {result['score']:.4f}")
        print(f"ID: {result['id']}")
        print(f"Source: {result['source']}")
        print(f"Text: {result['text']}")
        print("-" * 60)