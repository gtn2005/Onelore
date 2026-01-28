import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBED_FILE = os.path.join(BASE_DIR, "embeddings.pkl")
MODEL_NAME = "all-MiniLM-L6-v2"

# Load once global to avoid reloading on every call if imported
if os.path.exists(EMBED_FILE):
    with open(EMBED_FILE, "rb") as f:
        CHUNKS, EMBEDDINGS = pickle.load(f)
else:
    CHUNKS, EMBEDDINGS = [], []

model = SentenceTransformer(MODEL_NAME)

def retrieve(query, top_k=3, threshold=0.3):
    """
    Returns a list of dictionaries:
    [
      {'text': '...', 'metadata': {'title': 'Luffy', ...}, 'score': 0.85},
      ...
    ]
    """
    if not CHUNKS:
        return []

    # Encode query
    query_vec = model.encode([query])
    
    # Calculate similarity
    scores = cosine_similarity(query_vec, EMBEDDINGS)[0]
    
    # Rank results
    ranked_indices = scores.argsort()[::-1]
    
    results = []
    for idx in ranked_indices:
        score = scores[idx]
        if score < threshold:
            break
            
        chunk_data = CHUNKS[idx]
        results.append({
            "text": chunk_data["display_text"],
            "metadata": chunk_data["metadata"],
            "score": float(score)
        })
        
        if len(results) >= top_k:
            break
            
    return results