import os
import json
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import numpy as np

# -------------------------------
# Paths
# -------------------------------
DATA_PATH = "diabetes_kb_data/master_kb.json"
OUTPUT_DIR = "diabetes_kb_data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# Load KB
# -------------------------------
print("📖 Loading knowledge base...")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    kb_data = json.load(f)

if isinstance(kb_data, list):
    kb_data = {f"section_{i}": text for i, text in enumerate(kb_data)}

# -------------------------------
# Initialize Embeddings Model
# -------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# Utility: chunk text
# -------------------------------
def chunk_text(text, max_sentences=10):
    """Split text into chunks of max_sentences sentences."""
    sentences = sent_tokenize(text)
    for i in range(0, len(sentences), max_sentences):
        yield " ".join(sentences[i:i + max_sentences])

# -------------------------------
# Create Embeddings
# -------------------------------
embeddings_dict = {}
print("🧩 Creating embeddings for each section...")

for section, content in kb_data.items():
    print(f"Section: {section}")

    # ✅ Extract actual text
    if isinstance(content, dict):
        text = content.get("content", "")
    else:
        text = str(content)

    if not text.strip():
        print(f"⚠️ Skipping empty section: {section}")
        continue

    # Chunk and embed
    chunks = list(chunk_text(text, max_sentences=10))
    chunk_embeddings = model.encode(chunks, show_progress_bar=True)

    # Average chunk embeddings for the section
    section_embedding = np.mean(chunk_embeddings, axis=0)
    embeddings_dict[section] = section_embedding.tolist()

# -------------------------------
# Save Embeddings
# -------------------------------
output_file = os.path.join(OUTPUT_DIR, "kb_embeddings.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(embeddings_dict, f)

print(f"\n✅ Embeddings created and saved at: {output_file}")
