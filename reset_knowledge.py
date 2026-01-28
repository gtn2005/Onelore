import json
import os
import pickle
from sentence_transformers import SentenceTransformer

# 1. Setup Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(BASE_DIR, "lore_data.json")
PICKLE_PATH = os.path.join(BASE_DIR, "embeddings.pkl")

# 2. Define the Knowledge Base (The Lore)
lore_data = [
  {
    "category": "Character",
    "title": "Monkey D. Luffy",
    "content": "Monkey D. Luffy is the captain of the Straw Hat Pirates. Born in Foosha Village, he is the son of the revolutionary Monkey D. Dragon and the grandson of Marine hero Monkey D. Garp. He is the sworn brother of Portgas D. Ace and Sabo. He ate the Gomu Gomu no Mi (Model: Nika). He dreams of becoming the Pirate King."
  },
  {
    "category": "Character",
    "title": "Roronoa Zoro",
    "content": "Roronoa Zoro is the combatant of the Straw Hat Pirates. A former bounty hunter trained at the Shimotsuki Dojo, he is a descendant of the Shimotsuki line. He aims to fulfill his promise to his deceased rival Kuina to become the world's greatest swordsman. He uses the Three-Sword Style."
  },
  {
    "category": "Character",
    "title": "Sanji",
    "content": "Sanji is the cook of the Straw Hat Pirates. He was mentored by 'Red-Leg' Zeff, who saved him from a shipwreck and taught him to cook and fight. Sanji seeks the All Blue. He is a prince of the Germa Kingdom but was rejected by his father, Judge."
  },
  {
    "category": "Character",
    "title": "Nico Robin",
    "content": "Nico Robin is the archaeologist of the Straw Hat Pirates. Known as the 'Devil Child', she is the sole survivor of the Ohara Buster Call. She is the only person alive who can read Poneglyphs to reveal the Void Century."
  },
  {
    "category": "Character",
    "title": "Joy Boy",
    "content": "Joy Boy was a legendary figure from the Void Century who possessed the Nika fruit (Sun God Nika). He left the One Piece treasure on Laugh Tale. He made a promise to Fishman Island involving the ark Noah."
  },
  {
    "category": "Event",
    "title": "God Valley Incident",
    "content": "38 years ago, the Rocks Pirates were defeated at God Valley by a joint force of Gol D. Roger and Monkey D. Garp. Rocks D. Xebec, who wanted to be King of the World, fell here."
  },
  {
    "category": "Arc",
    "title": "Marineford Arc",
    "content": "In the Summit War Saga, Luffy infiltrates Marineford to save his brother Ace. The Whitebeard Pirates clash with the Marines. Portgas D. Ace and Whitebeard are killed, ending an era."
  }
]

# 3. Save JSON
print("📝 Writing lore_data.json...")
with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(lore_data, f, indent=4)

# 4. Create Embeddings
print("🧠 Encoding data...")
model = SentenceTransformer("all-MiniLM-L6-v2")

processed_chunks = []
for entry in lore_data:
    # Create the text vector
    # We combine Title + Content so the search is accurate
    text_to_embed = f"{entry['title']} {entry['content']}"
    
    processed_chunks.append({
        "display_text": entry["content"], # What the user sees
        "text_for_embedding": text_to_embed, # What the computer searches
        "metadata": {"title": entry["title"], "category": entry["category"]}
    })

texts = [x["text_for_embedding"] for x in processed_chunks]
embeddings = model.encode(texts, show_progress_bar=True)

# 5. Save Pickle
with open(PICKLE_PATH, "wb") as f:
    pickle.dump((processed_chunks, embeddings), f)

print(f"\n✅ SUCCESS! Saved {len(processed_chunks)} lore entries.")
print("👉 PLEASE RESTART YOUR STREAMLIT SERVER NOW.")