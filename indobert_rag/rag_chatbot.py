import chromadb
from sentence_transformers import SentenceTransformer
import re

# Load model IndoBERT
embedder = SentenceTransformer("indobenchmark/indobert-base-p1")
client = chromadb.PersistentClient(path="db_chroma")
collection = client.get_collection("kuhp2023")

# Daftar keyword per topik
KAMUS_KEYWORD = {
    "penggelapan": ["penggelapan", "uang", "dana"],
    "penipuan": ["penipuan", "tipu", "bohong"],
    "pencurian": ["pencurian", "curi", "mengambil"],
    "kekerasan": ["aniaya", "kekerasan", "pukul"],
    "korupsi": ["korupsi", "gratifikasi", "suap"],
}

def deteksi_keywords(query):
    lower = query.lower()
    for topik, keywords in KAMUS_KEYWORD.items():
        if any(k in lower for k in keywords):
            return keywords
    return []

def ambil_nomor_pasal(query):
    return re.findall(r"pasal\s(\d+[a-zA-Z]?)", query.lower())

def buat_prompt(user_query):
    nomor_dicari = ambil_nomor_pasal(user_query)

    # Jika menyebut pasal tertentu → ambil berdasarkan metadata
    if nomor_dicari:
        hasil = collection.get(
            where={"nomor": {"$in": [f"Pasal {n}" for n in nomor_dicari]}}
        )
        if hasil["documents"]:
            return "\n\n".join(
                f"{meta['nomor']}\n{doc.strip()}"
                for doc, meta in zip(hasil["documents"], hasil["metadatas"])
            )

    # Jika tidak menyebut pasal → gunakan embedding + keyword
    query_vec = embedder.encode(user_query.lower().strip()).tolist()
    hasil = collection.query(query_embeddings=[query_vec], n_results=20)

    pasal_list = hasil["documents"][0]
    metadatas = hasil["metadatas"][0]

    keywords = deteksi_keywords(user_query)

    pasal_terfilter = [
        f"{meta['nomor']}\n{isi.strip()}"
        for isi, meta in zip(pasal_list, metadatas)
        if any(k in isi.lower() for k in keywords)
    ]

    if pasal_terfilter:
        return "\n\n".join(pasal_terfilter)
    else:
        return "\n\n".join([f"{m['nomor']}\n{d.strip()}" for d, m in zip(pasal_list[:5], metadatas[:5])])

def chat_loop():
    print("🤖 Chatbot KUHP berbasis IndoBERT aktif. Ketik 'exit' untuk keluar.")
    while True:
        user_input = input("Anda: ")
        if user_input.lower() == "exit":
            break
        jawaban = buat_prompt(user_input)
        print("\n🧠 Jawaban berdasarkan KUHP:\n")
        print(jawaban)
        print("-" * 60)

if __name__ == "__main__":
    chat_loop()