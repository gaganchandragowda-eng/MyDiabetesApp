#!/usr/bin/env python3
"""
upload_kb_to_pinecone.py (updated for new Pinecone SDK)

Usage:
  export PINECONE_API_KEY="..."
  export PINECONE_ENV="us-west1-gcp"   # or other env string you use
  export PINECONE_INDEX="diabetes-rag-index"
  python upload_kb_to_pinecone.py
"""

import os
import json
import time
import numpy as np

# new SDK
from pinecone import Pinecone, ServerlessSpec

# ---- Adjust base dir if your project layout differs ----
BASE_DIR = "/Users/gagan/iCloud Drive (Archive)/Documents/MyDiabetesApp copy/DiabetesRAGProject"
EMBEDDINGS_FILE = os.path.join(BASE_DIR, "diabetes_kb_data", "processed", "kb_embeddings.json")
SECTIONS_FILE = os.path.join(BASE_DIR, "diabetes_kb_data", "processed", "sections.json")

# ---- Read Pinecone config from environment ----
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g. "us-west1-gcp" or "us-west1"
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "diabetes-rag-index")

if not PINECONE_API_KEY or not PINECONE_ENV:
    raise SystemExit("ERROR: Set PINECONE_API_KEY and PINECONE_ENV environment variables before running.")

# ---- Helper: parse env like "us-west1-gcp" -> region="us-west1", cloud="gcp" ----
def parse_env(env_str: str):
    # Accept patterns like "us-west1-gcp", "us-west1", "gcp-us-west1"
    env = env_str.strip()
    cloud = None
    region = None
    # common pattern "region-cloud" (e.g., "us-west1-gcp")
    if "-" in env:
        parts = env.split("-")
        # if last part is known cloud token
        last = parts[-1].lower()
        if last in ("gcp", "aws", "azure"):
            cloud = last
            region = "-".join(parts[:-1])
        else:
            # maybe first is cloud (unlikely), fallback to whole string as region
            region = env
    else:
        region = env
    # normalize cloud name
    if cloud == "gcp":
        cloud_name = "gcp"
    elif cloud == "aws":
        cloud_name = "aws"
    elif cloud == "azure":
        cloud_name = "azure"
    else:
        cloud_name = None
    return region, cloud_name

region, cloud = parse_env(PINECONE_ENV)

# ---- Create Pinecone client ----
pc = Pinecone(api_key=PINECONE_API_KEY)

# ---- Load local embeddings ----
if not os.path.exists(EMBEDDINGS_FILE):
    raise SystemExit(f"ERROR: Embeddings file not found at: {EMBEDDINGS_FILE}")

with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
    emb_dict = json.load(f)

keys = list(emb_dict.keys())
if len(keys) == 0:
    raise SystemExit("ERROR: No embeddings found in kb_embeddings.json")

# Convert embeddings to numpy arrays (and flatten any 2D -> 1D)
vectors = []
for k in keys:
    v = np.array(emb_dict[k], dtype="float32")
    if v.ndim > 1:
        v = np.mean(v, axis=0)
    vectors.append(v)

# ---- Ensure index exists (create if missing) ----
# pc.list_indexes() may return a list of names or dicts depending on SDK
existing = []
try:
    li = pc.list_indexes()
    # li could be list of names or list of dicts with 'name'
    if isinstance(li, list):
        if all(isinstance(x, dict) and "name" in x for x in li):
            existing = [x["name"] for x in li]
        else:
            existing = li
except Exception:
    existing = []

if PINECONE_INDEX not in existing:
    # infer dimension from vectors
    dim = int(vectors[0].shape[0])
    print(f"Index '{PINECONE_INDEX}' not found. Creating with dimension={dim} ...")
    # Provide ServerlessSpec if we parsed cloud/region
    spec = None
    if cloud and region:
        try:
            spec = ServerlessSpec(cloud=cloud, region=region)
        except Exception:
            spec = None
    try:
        if spec is not None:
            pc.create_index(name=PINECONE_INDEX, dimension=dim, metric="cosine", spec=spec)
        else:
            pc.create_index(name=PINECONE_INDEX, dimension=dim, metric="cosine")
    except Exception as e:
        print("WARNING: create_index raised an exception (continuing):", e)
    # Wait briefly for index readiness
    time.sleep(2)
else:
    print(f"Index '{PINECONE_INDEX}' already exists.")

# ---- Obtain index object and upsert in batches ----
# Using pc.Index(...) interface
idx = pc.Index(PINECONE_INDEX)

BATCH_SIZE = 100
total = len(vectors)
print(f"Uploading {total} vectors to Pinecone index '{PINECONE_INDEX}' (batch size {BATCH_SIZE})...")

for start in range(0, total, BATCH_SIZE):
    end = min(start + BATCH_SIZE, total)
    batch = []
    for i in range(start, end):
        vec = vectors[i].astype(float).tolist()
        metadata = {"section": keys[i]}
        batch.append({"id": f"sec-{i}", "values": vec, "metadata": metadata})
    # Upsert
    try:
        idx.upsert(vectors=batch)
    except Exception as e:
        # Some SDKs require tuples format: (id, values, metadata)
        try:
            tuples = [(b["id"], b["values"], b["metadata"]) for b in batch]
            idx.upsert(vectors=tuples)
        except Exception as e2:
            print("ERROR upserting batch:", e, e2)
            raise
    print(f"  - Upserted {start + 1}..{end} / {total}")

print("✅ All vectors uploaded to Pinecone.")

# ---- Optional: write / verify sections file (if not present) ----
if not os.path.exists(SECTIONS_FILE):
    try:
        with open(SECTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(keys, f, ensure_ascii=False, indent=2)
        print(f"✅ Wrote sections file at: {SECTIONS_FILE}")
    except Exception as e:
        print("⚠️ Could not write sections.json:", e)

# ---- Print index stats (best-effort) ----
try:
    stats = pc.describe_index(PINECONE_INDEX)
    print("Index description:", json.dumps(stats, indent=2))
    # try data-plane stats via Index.describe_index_stats if available
    try:
        stats2 = idx.describe_index_stats()
        print("Index stats (data-plane):", json.dumps(stats2, indent=2))
    except Exception:
        pass
except Exception as e:
    print("⚠️ Could not fetch index stats (control-plane):", e)

print("Done.")
