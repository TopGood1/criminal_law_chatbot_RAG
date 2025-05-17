import chromadb
from sentence_transformers import SentenceTransformer
import pickle

client = chromadb.PersistentClient(path="db_chroma")
model = SentenceTransformer("indobenchmark/indobert-base-p1")

def load_pasal_list(pickle_path="pasal_list.pkl"):
    with open(pickle_path, "rb") as f:
        return pickle.load(f)

def simpan_ke_chroma(pasal_list, collection_name="kuhp2023"):
    if collection_name in [col.name for col in client.list_collections()]:
        client.delete_collection(name=collection_name)

    collection = client.create_collection(name=collection_name)

    print("🧠 Membuat embedding IndoBERT...")

    for i, pasal in enumerate(pasal_list):
        nomor = pasal["nomor"]
        isi = pasal["isi"]
        full_text = f"{nomor}\n{isi}".lower().strip()
        embedding = model.encode(full_text).tolist()

        collection.add(
            documents=[isi],
            metadatas=[{"source": f"pasal_{i}", "nomor": nomor}],
            ids=[str(i)],
            embeddings=[embedding]
        )

    print(f"✅ Total {len(pasal_list)} pasal disimpan ke ChromaDB collection '{collection_name}'")

if __name__ == "__main__":
    pasal_list = load_pasal_list()
    simpan_ke_chroma(pasal_list)