
import gradio as gr
import httpx

API_URL = "https://policymind-ai.fastapicloud.dev/ask"


def ask_policy(message, history):
    question = message.strip()

    if not question:
        return "Silakan masukkan pertanyaan terlebih dahulu."

    try:
        response = httpx.post(
            API_URL,
            json={"question": question},
            timeout=60.0,
        )
        response.raise_for_status()
        result = response.json()

        answer = result.get("answer", "Belum ada jawaban dari API.")
        sources = result.get("sources", [])

        if sources:
            answer += "\n\n**Sumber dokumen:**\n"
            answer += "\n".join(f"- `{source}`" for source in sources)

        return answer

    except httpx.TimeoutException:
        return "Permintaan terlalu lama. Silakan coba lagi."

    except httpx.HTTPStatusError as error:
        return (
            f"API mengembalikan status "
            f"{error.response.status_code}. Silakan coba lagi."
        )

    except httpx.RequestError:
        return "Tidak dapat terhubung ke backend PolicyMind AI."


demo = gr.ChatInterface(
    fn=ask_policy,
    title="PolicyMind AI",
    description=(
        "Company Policy Assistant - tanyakan tentang cuti, "
        "reimbursement, tunjangan, dan kebijakan perusahaan."
    ),
    examples=[
        "Berapa hari cuti tahunan yang diberikan kepada karyawan?",
        "Berapa jam kerja standar perusahaan dalam satu minggu?",
        "Fasilitas asuransi kesehatan apa yang diberikan perusahaan?",
    ],
    save_history=False,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
