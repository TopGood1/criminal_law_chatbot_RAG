import chromadb
from sentence_transformers import SentenceTransformer
import pickle

# Inisialisasi client dan model
client = chromadb.PersistentClient(path="db_chroma")  # Memastikan path yang sama dengan retrieval_chat.py
model = SentenceTransformer("indobenchmark/indobert-base-p1")  # Load model sekali

def load_pasal_list(pickle_path="pasal_list.pkl"):
    """Memuat ulang pasal dari pickle file"""
    with open(pickle_path, "rb") as f:
        return pickle.load(f)

def simpan_ke_chroma(pasal_list, collection_name="kuhp2023"):
    """Menyimpan embedding pasal ke dalam ChromaDB"""
    
    # Hapus koleksi jika sudah ada
    if collection_name in [col.name for col in client.list_collections()]:
        client.delete_collection(name=collection_name)

    # Membuat koleksi baru
    collection = client.create_collection(name=collection_name)

    print("🧠 Membuat embedding IndoBERT...")

    for i, pasal in enumerate(pasal_list):
        # Menambahkan dokumen dengan embedding ke ChromaDB
        embedding = model.encode(pasal).tolist()
        collection.add(
            documents=[pasal],
            metadatas=[{"source": f"pasal_{i}"}],
            ids=[str(i)],
            embeddings=[embedding]
        )

    print(f"✅ Total {len(pasal_list)} pasal disimpan ke ChromaDB collection '{collection_name}'")

if __name__ == "__main__":
    pasal_list = load_pasal_list()
    simpan_ke_chroma(pasal_list)