import fitz  # PyMuPDF
import re
import os
import pickle

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def bersihkan_teks(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\n', ' ').strip()
    return text

def split_per_pasal(text):
    pattern = r"(Pasal\s\d+[A-Z]?)"
    split_text = re.split(pattern, text)
    
    pasal_list = []
    for i in range(1, len(split_text), 2):
        nomor = split_text[i].strip()
        isi = split_text[i + 1].strip() if i + 1 < len(split_text) else ""
        if len(isi) > 30:
            pasal_list.append({"nomor": nomor, "isi": isi})
    
    return pasal_list

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(base_dir, "dataset", "KUHP_2023.pdf")

    print(f"📂 Membaca PDF dari: {pdf_path}")
    raw_text = extract_text_from_pdf(pdf_path)

    print("🧹 Membersihkan teks...")
    clean_text = bersihkan_teks(raw_text)

    print("🔪 Memotong per pasal...")
    pasal_list = split_per_pasal(clean_text)

    print(f"📦 Total pasal diproses: {len(pasal_list)}")

    output_file = os.path.join(base_dir, "pasal_list.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(pasal_list, f)
    print(f"✅ Hasil disimpan ke: {output_file}")