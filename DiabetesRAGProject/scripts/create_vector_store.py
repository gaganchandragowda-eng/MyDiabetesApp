# build_vectorstore_pinecone.py
import json
import os
import time
from typing import List, Dict

import numpy as np
import pinecone

# -------------------------------
# Paths / config
# -------------------------------
EMBEDDINGS_FILE = "diabetes_kb_data/processed/kb_embeddings.json"
OUTPUT_DIR = "diabetes_kb_data/processed"
SECTIONS_FILE = os.path.join(OUTPUT_DIR, "sections.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Pinecone config from environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "diabetes-rag-index")

if not PINECONE_API_KEY or not PINECONE_ENV:
    raise RuntimeError("Set PINECONE_API_KEY and PINECONE_ENV environment variables before running.")

# -------------------------------
# Load embeddings
# -------------------------------
print("📖 Loading embeddings...")
with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
    embeddings_dict = json.load(f)

sections = list(embeddings_dict.keys())
vectors = []

for sec in sections:
    vec = np.array(embeddings_dict[sec], dtype="float32")
    # If embedding is 2D (chunk-averages etc.), reduce to 1D
    if vec.ndim == 1:
        arr = vec
    else:
        arr = np.mean(vec, axis=0).astype("float32")
    vectors.append(arr)

vectors = np.stack(vectors)
print(f"✅ Loaded {len(sections)} sections with embedding dimension {vectors.shape[1]}")

dimension = int(vectors.shape[1])

# -------------------------------
# Init Pinecone and create index if missing
# -------------------------------
print("⚙️ Initializing Pinecone...")
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

existing_indexes = pinecone.list_indexes()
if PINECONE_INDEX not in existing_indexes:
    print(f"🔨 Creating Pinecone index '{PINECONE_INDEX}' (dim={dimension})...")
    pinecone.create_index(name=PINECONE_INDEX, dimension=dimension, metric="cosine")
    # Wait until index is available (small sleep)
    time.sleep(3)
else:
    print(f"ℹ️ Pinecone index '{PINECONE_INDEX}' already exists.")

index = pinecone.Index(PINECONE_INDEX)

# -------------------------------
# Upsert vectors to Pinecone (batched)
# -------------------------------
print("📤 Upserting vectors to Pinecone...")
BATCH_SIZE = 100  # adjust if needed
total = vectors.shape[0]

def to_upsert_batch(start_idx: int, batch_size: int):
    end = min(start_idx + batch_size, total)
    batch = []
    for i in range(start_idx, end):
        # Use a stable id for each section (string)
        sec_id = f"sec-{i}"
        values = vectors[i].astype(float).tolist()  # ensure native python floats
        metadata = {"section": sections[i]}
        batch.append({"id": sec_id, "values": values, "metadata": metadata})
    return batch

for i in range(0, total, BATCH_SIZE):
    batch = to_upsert_batch(i, BATCH_SIZE)
    index.upsert(vectors=batch)
    print(f"  - upserted {i + 1}-{min(i + BATCH_SIZE, total)} / {total}")

print("✅ All vectors upserted to Pinecone.")

# -------------------------------
# Save local metadata (sections mapping)
# -------------------------------
with open(SECTIONS_FILE, "w", encoding="utf-8") as f:
    json.dump(sections, f, ensure_ascii=False, indent=2)

print(f"✅ Sections metadata saved at: {SECTIONS_FILE}")

# -------------------------------
# Optional: show index stats
# -------------------------------
try:
    stats = index.describe_index_stats()
    total_vectors = sum([v["count"] for v in stats.get("namespaces", {}).values()]) if stats else "unknown"
    print("📊 Index stats:", json.dumps(stats, indent=2) if stats else stats)
except Exception as e:
    print("⚠️ Could not fetch index stats:", e)

print("\nAll done. Pinecone index populated.")
