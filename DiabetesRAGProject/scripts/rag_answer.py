import sys
import os
from pathlib import Path

# -------------------------------------------------------
# Ensure the script can import the RAG connector properly
# -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "scripts"))

try:
    from rag_connector import generate_answer
except ImportError as e:
    raise ImportError(f"❌ Unable to import RAG connector: {e}")

# =====================================================
# 🧠 Interface Function for App Integration
# =====================================================
def get_answer(user_query):
    """
    Get an AI-generated answer for the user's diabetes-related query.
    """
    try:
        print("🔍 Generating RAG-based answer...")
        answer = generate_answer(user_query)
        return answer
    except Exception as e:
        return f"⚠️ Error while generating answer: {str(e)}"

# =====================================================
# 🧪 CLI TEST (Optional)
# =====================================================
if __name__ == "__main__":
    print("\n💬 RAG Answer Interface ready! Type your question (or 'exit' to quit):\n")
    while True:
        q = input("Your question: ")
        if q.lower() in ["exit", "quit"]:
            print("👋 Exiting...")
            break
        ans = get_answer(q)
        print("\n✅ Answer:\n", ans)
        print("\n" + "-" * 60 + "\n")
