import json
import os
import pickle
from sentence_transformers import SentenceTransformer

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "lore_data.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "embeddings.pkl")

MODEL_NAME = "all-MiniLM-L6-v2"

def create_sliding_window_chunks(text, chunk_size=100, overlap=30):
    """
    Splits text into chunks with overlap to preserve context.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break
    return chunks

def main():
    print("📂 Loading structured lore...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_chunks = []
    
    print("🧠 Processing and Embedding...")
    
    for entry in data:
        # 1. Clean raw content
        raw_text = entry["content"]
        
        # 2. Chunk text with overlap
        text_chunks = create_sliding_window_chunks(raw_text)
        
        for chunk in text_chunks:
            # 3. Context Injection: Add metadata to the text itself for better matching
            # Format: "Category: Title - Content..."
            enriched_text = f"{entry['category']}: {entry['title']} - {chunk}"
            
            # Store both the vector-ready text and the metadata for the UI
            processed_chunks.append({
                "text_for_embedding": enriched_text,
                "display_text": chunk,
                "metadata": {
                    "title": entry["title"],
                    "category": entry["category"]
                }
            })

    # 4. Create Embeddings
    model = SentenceTransformer(MODEL_NAME)
    texts_to_embed = [item["text_for_embedding"] for item in processed_chunks]
    embeddings = model.encode(texts_to_embed, show_progress_bar=True)

    # 5. Save everything
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump((processed_chunks, embeddings), f)

    print(f"✅ Saved {len(processed_chunks)} searchable lore chunks to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()