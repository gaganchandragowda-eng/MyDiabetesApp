# scripts/rag_retrieval.py

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -------------------------------
# Paths
# -------------------------------
EMBEDDINGS_FILE = "diabetes_kb_data/processed/kb_embeddings.json"

# -------------------------------
# Load embeddings
# -------------------------------
print("📖 Loading embeddings...")
with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
    embeddings_dict = json.load(f)

# Prepare vectors and section mapping
vectors = []
sections = []

for section, emb in embeddings_dict.items():
    vectors.append(emb)  # each emb should be a 1D list
    sections.append(section)

# Convert to 2D numpy array, dtype float32 for FAISS
vectors = np.array(vectors, dtype="float32")
print("Vectors shape:", vectors.shape)

# -------------------------------
# Build FAISS index
# -------------------------------
vector_dim = vectors.shape[1]
index = faiss.IndexFlatL2(vector_dim)  # L2 distance
index.add(vectors)
print("✅ FAISS index built. Total vectors:", index.ntotal)

# -------------------------------
# Initialize model for query encoding
# -------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# Interactive query
# -------------------------------
print("\n💬 Ask questions about diabetes! Type 'exit' to quit.")

while True:
    query = input("\nEnter your question: ").strip()
    if query.lower() == "exit":
        break

    # Encode query
    query_emb = model.encode([query], convert_to_numpy=True).astype("float32")

    # Search top 3 similar sections
    D, I = index.search(query_emb, k=3)

    print("\nTop results:")
    for idx, dist in zip(I[0], D[0]):
        print(f"- Section: {sections[idx]} | Distance: {dist:.4f}")
import os
import json
import numpy as np
import faiss

# -------------------------------
# Paths
# -------------------------------
EMBEDDINGS_FILE = "diabetes_kb_data/processed/kb_embeddings.json"
KB_FILE = "diabetes_kb_data/master_kb.json"

# -------------------------------
# Load knowledge base text
# -------------------------------
print("📖 Loading knowledge base...")
with open(KB_FILE, "r", encoding="utf-8") as f:
    kb_data = json.load(f)

# Ensure kb_data is dict
if isinstance(kb_data, list):
    kb_data = {f"section_{i}": text for i, text in enumerate(kb_data)}

# -------------------------------
# Load embeddings
# -------------------------------
print("📖 Loading embeddings...")
with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
    embeddings_dict = json.load(f)

vectors = []
sections = []

for section, emb in embeddings_dict.items():
    emb_array = np.array(emb, dtype="float32")
    if len(emb_array.shape) != 1:
        print(f"⚠ Section {section} has wrong embedding shape: {emb_array.shape}")
        continue
    vectors.append(emb_array)
    sections.append(section)

vectors = np.stack(vectors)
print(f"✅ Loaded {len(sections)} embeddings. Shape: {vectors.shape}")

# -------------------------------
# Build FAISS index
# -------------------------------
vector_dim = vectors.shape[1]
index = faiss.IndexFlatL2(vector_dim)
index.add(vectors)
print(f"✅ FAISS index built with {index.ntotal} vectors.")

# -------------------------------
# Retrieval function
# -------------------------------
def retrieve(query_embedding, top_k=5):
    """
    Retrieve top-k sections from the KB given a query embedding.
    Returns section names and their text content.
    """
    query_vector = np.array(query_embedding, dtype="float32").reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        section_name = sections[idx]
        section_text = kb_data.get(section_name, "")
        results.append({
            "section": section_name,
            "text": section_text,
            "distance": float(distances[0][i])
        })
    return results

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    query = "What are the symptoms of diabetes?"
    query_emb = model.encode(query)
    top_results = retrieve(query_emb, top_k=5)

    print("\n🔎 Top 5 retrieved sections with content:")
    for res in top_results:
        print(f"\nSection: {res['section']} (distance: {res['distance']:.4f})")
        print(f"Content: {res['text'][:500]}...")  # Show first 500 chars
import json
import numpy as np
import faiss

# -------------------------------
# Paths
# -------------------------------
EMBEDDINGS_FILE = "diabetes_kb_data/processed/kb_embeddings.json"

# -------------------------------
# Load embeddings
# -------------------------------
print("📖 Loading embeddings...")
with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
    kb_embeddings = json.load(f)

vectors = []
sections = []

for section, emb in kb_embeddings.items():
    vec = np.array(emb, dtype='float32')
    
    if vec.ndim == 1:
        vectors.append(vec)
        sections.append(section)
    else:
        print(f"⚠ Section {section} has wrong embedding shape: {vec.shape}")

if not vectors:
    raise ValueError("❌ No valid embeddings to add to FAISS index!")

vectors = np.stack(vectors)  # shape (num_sections, 384)

# -------------------------------
# Create FAISS index
# -------------------------------
dimension = vectors.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(vectors)
print(f"✅ FAISS index created with {index.ntotal} vectors.")

# -------------------------------
# Example Retrieval
# -------------------------------
query = "What are the early symptoms of diabetes?"
query_embedding = np.array(kb_embeddings["section_0"], dtype='float32')  # Example using first section
query_embedding = query_embedding.reshape(1, -1)

k = 5
distances, indices = index.search(query_embedding, k)
print("\nTop results:")
for i, idx in enumerate(indices[0]):
    print(f"{i+1}. Section: {sections[idx]}, Distance: {distances[0][i]:.4f}")
