import re
from pathlib import Path

from pypdf import PdfReader

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PDF_PATH = DATA_DIR / "Kebijakan Perusahaan - PT XYZ Tech.pdf"


def load_documents():
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"File kebijakan tidak ditemukan: {PDF_PATH}"
        )

    reader = PdfReader(str(PDF_PATH))
    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""

        # Hapus footer dokumen yang muncul di setiap halaman.
        text = re.sub(
        r'PT XYZ Tech - "DOKUMEN FIKTIF - Hanya Untuk Keperluan Portofolio AI" Halaman \d+',
        "",
        text,
    )

        if text.strip():
            pages.append(text.strip())

    content = "\n\n".join(pages)

    if not content.strip():
        raise ValueError(
            f"Tidak ada teks yang berhasil diekstrak dari {PDF_PATH.name}"
        )

    return [
        {
            "source": PDF_PATH.name,
            "content": content,
        }
    ]


if __name__ == "__main__":
    docs = load_documents()

    print(f"Jumlah dokumen: {len(docs)}")
    print(f"\nSumber: {docs[0]['source']}")
    print(f"Isi dokumen:\n{docs[0]['content']}")
