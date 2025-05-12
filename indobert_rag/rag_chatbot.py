import chromadb
from sentence_transformers import SentenceTransformer
import torch

# Load embedder IndoBERT
print("🔁 Loading SentenceTransformer IndoBERT...")
embedder = SentenceTransformer("indobenchmark/indobert-base-p1")

# Connect ke ChromaDB dan koleksi
client = chromadb.PersistentClient(path="db_chroma")
collection = client.get_collection("kuhp2023")

def buat_prompt(user_query):
    """Ambil dokumen relevan dari ChromaDB untuk pertanyaan"""
    query_vec = embedder.encode(user_query).tolist()
    hasil = collection.query(query_embeddings=[query_vec], n_results=3)
    dokumen_terkait = "\n\n".join([doc for doc in hasil["documents"][0]])
    return dokumen_terkait

def chat_loop():
    print("🤖 RAG Chatbot aktif. Ketik 'exit' untuk keluar.")
    while True:
        user_input = input("Anda: ")
        if user_input.lower() == "exit":
            break

        dokumen = buat_prompt(user_input)

        print("🧠 Jawaban berdasarkan KUHP:\n")
        print(dokumen.strip())
        print("-" * 50)

if __name__ == "__main__":
    chat_loop()