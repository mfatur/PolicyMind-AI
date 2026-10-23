import os

from dotenv import load_dotenv

load_dotenv()

from groq import Groq
from langfuse import observe

load_dotenv()

MODEL_NAME = "qwen/qwen3.8-27b"

@observe(name="generate-answer", as_type="generation")
def generate_answer(question, context):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY belum dikonfigurasi."
        )

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Anda adalah PolicyMind AI, asisten kebijakan HR. "
                    "Jawab dalam bahasa Indonesia yang jelas. "
                    "Gunakan hanya informasi dari konteks yang diberikan. "
                    "Jangan mengarang aturan, angka, atau ketentuan. "
                    "Jika konteks tidak memuat jawabannya, katakan "
                    "bahwa informasi tersebut tidak ditemukan dalam "
                    "dokumen dan sarankan pengguna menghubungi HR."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Konteks kebijakan:\n{context}\n\n"
                    f"Pertanyaan: {question}"
                ),
            },
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    question = "Berapa nominal tunjangan komunikasi?"

    context = (
        "Tunjangan komunikasi sebesar Rp350.000 per bulan. "
        "Anggaran pengembangan diri sebesar Rp2.000.000 per tahun."
    )

    answer = generate_answer(question, context)
    print(answer)
    