
from sentence_transformers import CrossEncoder

model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker = CrossEncoder(model_name)

question = "Berapa lama waktu istirahat karyawan setiap hari kerja?"

candidates = [
    {
        "source": "kebijakan_jam_kerja.txt",
        "content": (
            "4. Keterlambatan\n"
            "Karyawan yang terlambat wajib memberikan keterangan "
            "melalui sistem kehadiran."
        ),
    },
    {
        "source": "kebijakan_jam_kerja.txt",
        "content": (
            "2. Waktu Istirahat\n"
            "Karyawan mendapatkan waktu istirahat selama 60 menit "
            "setiap hari kerja. Waktu istirahat tidak termasuk "
            "dalam jam kerja efektif."
        ),
    },
    {
        "source": "kebijakan_cuti.txt",
        "content": (
            "Setiap karyawan tetap yang telah bekerja selama "
            "12 bulan berhak mendapatkan cuti tahunan sebanyak "
            "12 hari kerja."
        ),
    },
]

pairs = [
    (question, candidate["content"])
    for candidate in candidates
]

scores = reranker.predict(pairs)

ranked_results = sorted(
    zip(scores, candidates),
    key=lambda item: item[0],
    reverse=True,
)

for rank, (score, candidate) in enumerate(ranked_results, start=1):
    print(f"\n--- Peringkat {rank} ---")
    print(f"Skor: {float(score):.4f}")
    print(f"Sumber: {candidate['source']}")
    print(f"Isi: {candidate['content']}")
