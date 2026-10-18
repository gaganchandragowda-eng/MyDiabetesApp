import os
import json
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import openai

# ================================
# ✅ CONFIGURATION
# ================================
BASE_DIR = Path(__file__).resolve().parents[1]
KB_PATH = BASE_DIR / "diabetes_kb_data" / "master_kb.json"

# Optional: You can set your OpenAI API key here
openai.api_key = os.getenv("OPENAI_API_KEY", "dummy-key-if-offline")

# ================================
# ✅ LOAD KNOWLEDGE BASE
# ================================
if not KB_PATH.exists():
    raise FileNotFoundError(f"Knowledge base not found at {KB_PATH}")

with open(KB_PATH, "r", encoding="utf-8") as f:
    kb_data = json.load(f)

# Each KB entry must be a dictionary with fields like {"topic": "diet", "content": "..."}
if isinstance(kb_data, dict):
    kb_data = [kb_data]

kb_texts = [entry.get("content", "") for entry in kb_data]
kb_topics = [entry.get("topic", "general") for entry in kb_data]

# ================================
# ✅ EMBEDDING MODEL
# ================================
print("🔍 Loading embedding model (all-MiniLM-L6-v2)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
kb_embeddings = embedder.encode(kb_texts, convert_to_tensor=False, show_progress_bar=False)


# ================================
# ✅ RAG CLASS
# ================================
class DiabetesRAG:
    def __init__(self):
        self.kb_data = kb_data
        self.kb_texts = kb_texts
        self.kb_embeddings = kb_embeddings
        self.embedder = embedder

    def retrieve(self, query, top_k=3):
        """Retrieve the most relevant KB chunks for a given query"""
        query_embedding = self.embedder.encode([query], convert_to_tensor=False)
        similarities = cosine_similarity(query_embedding, self.kb_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        retrieved = [
            {
                "topic": self.kb_data[i].get("topic", "general"),
                "content": self.kb_data[i].get("content", ""),
                "score": float(similarities[i]),
            }
            for i in top_indices
        ]
        return retrieved

    def generate_response(self, query, retrieved):
        """Use GPT (or fallback) to summarize retrieved context into a descriptive answer"""
        context = "\n".join([r["content"] for r in retrieved])

        prompt = f"""
        You are a diabetes health assistant.
        Based on the following knowledge base context, write a clear and descriptive report:

        Query: {query}
        Context:
        {context}

        Give a short, friendly and medically accurate summary (4-5 lines max).
        """

        try:
            # 🔹 Use OpenAI (if API key available)
            if openai.api_key and openai.api_key != "dummy-key-if-offline":
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a diabetes medical assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.5,
                    max_tokens=200,
                )
                return response["choices"][0]["message"]["content"].strip()

            # 🔹 Fallback (offline mode)
            else:
                return (
                    "Your parameters suggest some diabetic risk. "
                    "Maintain a balanced low-carb diet, exercise daily, "
                    "and monitor glucose regularly for prevention."
                )

        except Exception as e:
            print("❌ Error generating RAG response:", e)
            return "Could not generate report at this moment. Please retry later."


# ================================
# ✅ HELPER FUNCTION
# ================================
def query_rag(rag_instance, query):
    """Single helper for external modules (Flask app, etc.)"""
    results = rag_instance.retrieve(query)
    answer = rag_instance.generate_response(query, results)
    return answer


# ================================
# ✅ STANDALONE TEST (OPTIONAL)
# ================================
if __name__ == "__main__":
    rag = DiabetesRAG()
    query = "Generate a descriptive report for moderate diabetes risk with high glucose and BMI"
    print("\n--- Retrieved Context ---")
    for r in rag.retrieve(query):
        print(f"- {r['topic']}: {r['content'][:80]}...")
    print("\n--- Generated Report ---")
    print(rag.generate_response(query, rag.retrieve(query)))
