import os
import json
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

INDEX_NAME = "diabetes-kb"
DIMENSION = 384
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BASE_DIR = os.path.dirname(__file__)
KB_PATH = os.path.join(BASE_DIR, "diabetes_kb_data", "processed", "sections.json")


def enhanced_rag():
    api_key = os.getenv("PINECONE_API_KEY")

    if not api_key:
        return {"status": "inactive", "reason": "PINECONE_API_KEY not set"}

    try:
        # ✅ Init Pinecone
        pc = Pinecone(api_key=api_key)

        # ✅ Fix index list (version-safe)
        index_list = pc.list_indexes()
        index_names = [idx["name"] for idx in index_list] if isinstance(index_list, list) else []

        # ✅ Create index if not exists
        if INDEX_NAME not in index_names:
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"✅ Created index: {INDEX_NAME}")

        index = pc.Index(INDEX_NAME)

        # ✅ Load embedding model
        embedder = SentenceTransformer(MODEL_NAME)

        # ✅ FIXED PATH
        print("📂 Loading KB from:", KB_PATH)

        if not os.path.exists(KB_PATH):
            return {"status": "inactive", "reason": f"KB not found: {KB_PATH}"}

        with open(KB_PATH, "r", encoding="utf-8") as f:
            sections = json.load(f)

        # ✅ Fix JSON format
        texts = []
        for sec in sections:
            if isinstance(sec, dict):
                text = sec.get("text", "")
            else:
                text = str(sec)

            if text.strip():
                texts.append(text)

        if not texts:
            return {"status": "inactive", "reason": "No valid text found in KB"}

        # ✅ Check index stats safely
        stats = index.describe_index_stats()
        total_vectors = stats.get("total_vector_count", 0)

        # ✅ Upload only once
        if total_vectors == 0:
            print("🚀 Upserting KB into Pinecone...")

            vectors = []
            for i, text in enumerate(texts):
                emb = embedder.encode(text).tolist()
                vectors.append({
                    "id": str(i),
                    "values": emb,
                    "metadata": {"text": text}
                })

            index.upsert(vectors=vectors)
            print(f"✅ Upserted {len(vectors)} vectors")

        # ✅ Retrieval function
        def retrieve(query: str, k: int = 5):
            q_emb = embedder.encode(query).tolist()
            res = index.query(vector=q_emb, top_k=k, include_metadata=True)

            results = []
            for match in res.get("matches", []):
                metadata = match.get("metadata", {})
                if "text" in metadata:
                    results.append(metadata["text"])

            return results

        return {
            "status": "active",
            "index": INDEX_NAME,
            "total_vectors": total_vectors,
            "retrieve_fn": retrieve
        }

    except Exception as e:
        return {"status": "inactive", "reason": str(e)}