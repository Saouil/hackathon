# RAG.py
import json
from sentence_transformers import SentenceTransformer
import chromadb

# Load fuzzy Layer 2 rules for weather explanation
with open("C:\Users\Huang\Documents\hack\fuzzy_inference_system\fuzzy_rules_layer2_clean.json", encoding="utf-8") as f:
    fuzzy_rules_layer2 = json.load(f)["rules"]

# Setup ChromaDB client and embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.get_or_create_collection("layer2_weather_explanation")

# Embed all rules only if collection is empty
if not collection.get()["ids"]:
    for idx, rule in enumerate(fuzzy_rules_layer2):
        conditions = " AND ".join([f"{k} is {v}" for k, v in rule["IF"].items()])
        rule_text = f"IF {conditions} THEN weather conditions indicate potential rainfall pattern."
        embedding = embedding_model.encode(rule_text).tolist()
        collection.add(
            ids=[str(idx)],
            documents=[rule_text],
            embeddings=[embedding],
            metadatas=[{"text": rule_text}]
        )

# Retrieval function for fuzzy explanation context

def retrieve_fuzzy_rules_context(query, top_k=3):
    query_embedding = embedding_model.encode(query).tolist()
    all_ids = collection.get()["ids"]
    top_k = top_k or len(all_ids)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return "\n".join([meta["text"] for meta in results["metadatas"][0]])
