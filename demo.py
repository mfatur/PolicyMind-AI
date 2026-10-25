import json
import logging
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import gradio as gr

logger = logging.getLogger(__name__)
API_URL = os.getenv("POLICYMIND_API_URL", "http://127.0.0.1:8000").rstrip("/")

EXAMPLES = [
    ("▧", "Cuti tahunan", "Berapa hari cuti tahunan karyawan?"),
    ("◎", "Kerja dari rumah", "Berapa hari kerja dari rumah diperbolehkan?"),
    ("◇", "Tunjangan", "Berapa tunjangan komunikasi per bulan?"),
    ("♧", "Pengembangan diri", "Apa saja klaim pengembangan diri?"),
    ("···", "Lainnya", "Apa saja kebijakan perusahaan yang tersedia?"),
]


def ask_policy(message):
    question = (message or "").strip()

    if not question:
        return "Silakan masukkan pertanyaan terlebih dahulu."

    request = Request(
        f"{API_URL}/ask",
        data=json.dumps({"question": question}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
        if not isinstance(result, dict):
            raise ValueError("Respons backend harus berupa objek JSON.")

        answer = result.get("answer", "Belum ada jawaban dari backend.")
        sources = result.get("sources", [])
        if not isinstance(answer, str):
            raise ValueError("Jawaban backend bukan teks.")
        if not isinstance(sources, list) or not all(
            isinstance(source, str) for source in sources
        ):
            raise ValueError("Daftar sumber backend tidak valid.")

        if sources:
            answer += "\n\n**Sumber dokumen**\n"
            answer += "\n".join(f"- `{source}`" for source in sources)

        return answer

    except HTTPError as error:
        logger.error("Backend /ask mengembalikan HTTP %s.", error.code)
        return "Backend gagal memproses pertanyaan. Silakan coba lagi."
    except (URLError, TimeoutError) as error:
        logger.error("Tidak dapat menghubungi backend pada %s: %s", API_URL, error)
        return "Backend tidak dapat dihubungi. Pastikan API sedang berjalan."
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
        logger.exception("Respons backend /ask tidak valid: %s", error)
        return "Terjadi kesalahan saat memproses pertanyaan. Silakan coba lagi."


def submit_question(message, history):
    question = (message or "").strip()
    history = history or []

    if not question:
        yield (
            "",
            gr.update(value=history),
            gr.update(),
            gr.update(),
            gr.update(visible=False),
            gr.update(interactive=True),
        )
        return

    history = history + [
        {"role": "user", "content": question},
    ]

    yield (
        gr.update(value="", interactive=True),
        gr.update(value=history, visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(interactive=False),
    )

    answer = ask_policy(question)
    history = history + [{"role": "assistant", "content": answer}]

    yield (
        gr.update(interactive=True),
        gr.update(value=history, visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(interactive=True),
    )


css = """
:root {
    color-scheme: light;
    --page-bg: #ffffff;
    --text: #171717;
    --muted: #676767;
    --border: #e6e6e6;
    --composer: #f4f4f4;
}

html,
body,
.gradio-container,
.gradio-container main {
    width: 100%;
    min-height: 100%;
    margin: 0 !important;
    padding: 0 !important;
    background: var(--page-bg) !important;
}

html,
body {
    min-height: 100vh;
    overflow-x: hidden;
}

.gradio-container {
    display: flex !important;
    align-items: center;
    justify-content: center;
    max-width: none !important;
    min-height: 100vh !important;
}

.gradio-container main {
    display: flex;
    align-items: center;
    justify-content: center;
}

footer {
    display: none !important;
}

#app-shell {
    position: relative;
    display: flex !important;
    flex-direction: column;
    gap: 0 !important;
    box-sizing: border-box;
    width: 91.7vw !important;
    max-width: 1440px;
    height: 86.7vh;
    min-height: 560px;
    margin: 4.5vh auto !important;
    padding: 0 8px 12px !important;
    overflow: hidden;
    border: 1px solid #e7e7e7 !important;
    border-radius: 12px !important;
    background: #fff !important;
    box-shadow: 0 2px 5px rgba(0, 0, 0, .07),
                0 5px 14px rgba(0, 0, 0, .035) !important;
}

#topbar {
    display: flex;
    flex: 0 0 48px;
    align-items: center;
    width: 100%;
    padding: 0 2px;
    color: #5f6368;
    transform: translateY(3px);
}

#topbar .brand {
    display: flex;
    align-items: center;
    gap: 19px;
}

#topbar .header-icon {
    display: inline-flex;
    width: 22px;
    height: 22px;
    align-items: center;
    justify-content: center;
    padding: 0;
    border: 0;
    background: transparent;
    color: #5f6368;
}

#topbar .header-icon svg {
    width: 21px;
    height: 21px;
    color: #5f6368 !important;
    fill: none;
    stroke: currentColor;
    stroke-linecap: round;
    stroke-linejoin: round;
    stroke-width: 2.2;
}

#topbar .brand-name {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    margin-left: 2px;
    color: #555;
    font-size: 18px;
    font-weight: 600;
    letter-spacing: -.3px;
}

#topbar .brand-name svg {
    width: 13px;
    height: 13px;
    color: #555 !important;
    fill: none;
    stroke: currentColor;
    stroke-width: 1.8;
}

#topbar .profile {
    display: flex;
    width: 32px;
    height: 32px;
    align-items: center;
    justify-content: center;
    margin-left: auto;
    overflow: hidden;
    border-radius: 50%;
    background: #ededed;
    color: #3f3f3f;
    font-size: 12px;
    font-weight: 600;
}

#topbar .profile svg {
    width: 26px;
    height: 26px;
    fill: #555;
}

#main-area {
    display: flex !important;
    flex: 1 1 auto;
    flex-direction: column;
    gap: 0 !important;
    width: 100%;
    min-height: 0;
    box-sizing: border-box;
    padding: 0 0 24px !important;
    background: transparent !important;
}

#welcome-block {
    margin-top: auto !important;
    margin-bottom: 11px !important;
}

#welcome {
    margin: 0;
    text-align: center;
}

#welcome h1 {
    margin: 0;
    color: #111;
    font-size: clamp(28px, 2.6vw, 32px);
    font-weight: 600;
    letter-spacing: -.75px;
    line-height: 1.25;
}

#conversation {
    flex: 1 1 auto;
    width: 100%;
    max-width: 850px;
    min-height: 0;
    height: 100% !important;
    margin: 18px auto 14px !important;
    overflow-y: auto;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    color: #171717 !important;
}

#typing-indicator {
    width: 100%;
    max-width: 850px;
    margin: 0 auto 14px !important;
}

.typing-dots {
    display: flex;
    align-items: center;
    gap: 5px;
    min-height: 24px;
    padding-left: 8px;
}

.typing-dots span {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #777;
    animation: typing-bounce 1s infinite ease-in-out;
}

.typing-dots span:nth-child(2) {
    animation-delay: .15s;
}

.typing-dots span:nth-child(3) {
    animation-delay: .3s;
}

@keyframes typing-bounce {
    0%, 60%, 100% {
        transform: translateY(0);
        opacity: .45;
    }
    30% {
        transform: translateY(-5px);
        opacity: 1;
    }
}

#conversation .bubble-wrap {
    background: #fff !important;
    color: #171717 !important;
}

#conversation .message.user {
    border-color: transparent !important;
    background: #f4f4f4 !important;
}

#conversation .message.bot {
    border-color: transparent !important;
    background: transparent !important;
}

#conversation code {
    border-color: #e5e5e5 !important;
    background: #f4f4f4 !important;
}

#conversation .message-wrap,
#conversation .message-wrap * {
    color: #171717 !important;
}

#conversation .icon-button-wrapper,
#conversation .icon-button {
    border-color: #e5e5e5 !important;
    background: #fff !important;
}

#conversation .top-panel .icon-button {
    border-radius: 8px !important;
    background: #fafafa !important;
    color: #444 !important;
    box-shadow: inset 0 0 0 1px #e5e5e5 !important;
    transition: background-color .15s ease, box-shadow .15s ease;
}

#conversation .top-panel.icon-button-wrapper {
    border: 0 !important;
    border-radius: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

#conversation .top-panel {
    width: max-content !important;
    flex: 0 0 auto !important;
    justify-content: flex-end !important;
}

#conversation .top-panel .icon-button {
    flex: 0 0 max-content !important;
    width: max-content !important;
    height: auto !important;
    min-height: 30px !important;
    gap: 7px;
    padding: 5px 9px !important;
    font-size: 12px !important;
    font-weight: 500;
    line-height: 1.2;
    white-space: nowrap;
}

#conversation .top-panel .icon-button[aria-label="Clear"]::before {
    content: "Bersihkan Obrolan";
}

#conversation .top-panel .icon-button[aria-label="Copy conversation"]::after {
    content: "Copy Conversation";
}

#conversation .top-panel .icon-button[aria-label="Clear"]::after {
    content: none !important;
    display: none !important;
}

#conversation .top-panel .icon-button:hover {
    background: #f1f1f1 !important;
    box-shadow: inset 0 0 0 1px #d8d8d8 !important;
}

#conversation .message {
    border-radius: 16px !important;
    box-shadow: none !important;
    line-height: 1.75 !important;
}

#composer {
    display: flex !important;
    flex: 0 0 auto;
    align-items: center;
    gap: 12px !important;
    width: 100%;
    max-width: 800px;
    min-height: 54px;
    box-sizing: border-box;
    margin: 0 auto !important;
    padding: 5px 9px 5px 17px !important;
    border: 1px solid transparent !important;
    border-radius: 999px !important;
    background: var(--composer) !important;
    box-shadow: none !important;
}

#attachment-icon {
    display: flex;
    flex: 0 0 22px;
    align-items: center;
    justify-content: center;
    margin-right: 3px;
    color: #171717;
}

#attachment-wrap {
    flex: 0 0 25px !important;
    width: 25px !important;
    min-width: 25px !important;
    margin: 0 !important;
    padding: 0 !important;
}

#attachment-wrap .html-container,
#attachment-wrap .prose {
    width: 100% !important;
    height: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

#attachment-icon {
    width: 22px !important;
    min-width: 22px !important;
    padding: 0 !important;
}

#attachment-icon svg {
    width: 18px;
    height: 18px;
    fill: none;
    color: #171717 !important;
    stroke: currentColor;
    stroke-linecap: round;
    stroke-linejoin: round;
    stroke-width: 2;
}

#composer:focus-within {
    border-color: #e0e0e0 !important;
}

#question-input,
#question-input > div,
#question-input textarea {
    min-height: 40px !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    color: #222 !important;
}

#question-input {
    flex: 1 1 auto;
    min-width: 0;
    width: auto !important;
}

#question-input textarea {
    padding: 10px 0 !important;
    font-size: 16px !important;
    line-height: 20px !important;
    resize: none !important;
}

#question-input textarea::placeholder {
    color: #666 !important;
    opacity: 1 !important;
}

#send-button {
    flex: 0 0 34px !important;
    width: 34px !important;
    min-width: 34px !important;
    height: 34px !important;
    min-height: 34px !important;
    padding: 0 !important;
    border: 0 !important;
    border-radius: 50% !important;
    background: #d6d6d6 !important;
    color: #fff !important;
    box-shadow: none !important;
    font-size: 21px !important;
    font-weight: 500 !important;
}

#send-button:hover {
    background: #bdbdbd !important;
}

#suggestions {
    display: flex !important;
    flex: 0 0 auto;
    flex-wrap: wrap !important;
    justify-content: center !important;
    gap: 9px !important;
    width: 100%;
    max-width: 900px;
    margin: 28px auto auto !important;
}

#suggestions button {
    flex: 0 0 auto !important;
    width: auto !important;
    min-width: 0 !important;
    min-height: 43px !important;
    padding: 0 9px !important;
    border: 1px solid #e2e2e2 !important;
    border-radius: 999px !important;
    background: #fff !important;
    color: #626262 !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, .035) !important;
    font-size: 14px !important;
    font-weight: 400 !important;
    white-space: nowrap !important;
}

#suggestions button:hover {
    border-color: #cfcfcf !important;
    background: #f8f8f8 !important;
}

#disclaimer {
    position: absolute;
    right: 16px;
    bottom: -7px;
    left: 16px;
    color: #646464;
    font-size: 12px;
    text-align: center;
    pointer-events: none;
}

@media (max-width: 700px) {
    #conversation .top-panel .icon-button {
        gap: 4px;
        padding: 4px 6px !important;
        font-size: 10px !important;
    }

    #app-shell {
        position: fixed !important;
        top: 8px !important;
        left: 8px !important;
        width: calc(100vw - 16px) !important;
        height: calc(100vh - 16px);
        min-height: 0;
        margin: 0 !important;
        padding: 0 12px 12px !important;
        border-radius: 12px !important;
    }

    #welcome-block {
        margin-bottom: 11px !important;
    }

    #topbar .brand {
        gap: 11px;
    }

    #topbar .brand-name {
        font-size: 16px;
    }

    #welcome h1 {
        font-size: 26px;
    }

    #suggestions {
        gap: 7px !important;
        margin-top: 20px !important;
    }

    #suggestions button {
        min-height: 39px !important;
        padding: 0 11px !important;
        font-size: 12px !important;
    }
}

@media (max-height: 640px) and (min-width: 701px) {
    #app-shell {
        height: calc(100vh - 32px);
        min-height: 0;
        margin-top: 16px !important;
        margin-bottom: 16px !important;
    }
}
"""

with gr.Blocks(title="PolicyMind AI") as demo:
    with gr.Column(elem_id="app-shell", variant="default"):
        gr.HTML(
            """
            <header id="topbar">
                <div class="brand">
                    <span class="brand-name">
                        PolicyMind AI
                        <svg viewBox="0 0 16 16" aria-hidden="true">
                            <path d="m3 6 5 5 5-5"></path>
                        </svg>
                    </span>
                </div>
                <span class="profile" aria-label="Profil">
                    <svg viewBox="0 0 32 32" aria-hidden="true">
                        <circle cx="16" cy="16" r="15" fill="#e9e9e9"></circle>
                        <circle cx="16" cy="12" r="5" fill="#555"></circle>
                        <path d="M6 28c1-6 4.7-9 10-9s9 3 10 9" fill="#555"></path>
                    </svg>
                </span>
            </header>
            """
        )

        with gr.Column(elem_id="main-area", variant="default"):
            welcome = gr.HTML(
                """
                <div id="welcome">
                    <h1>What can I help with?</h1>
                </div>
                """,
                elem_id="welcome-block",
            )

            chatbot = gr.Chatbot(
                value=[],
                height="100%",
                visible=False,
                elem_id="conversation",
                container=False,
                placeholder="Percakapan akan tampil di sini.",
                buttons=["copy_all"],
            )

            typing_indicator = gr.HTML(
                """
                <div class="typing-dots" aria-label="Asisten sedang berpikir">
                    <span></span><span></span><span></span>
                </div>
                """,
                visible=False,
                elem_id="typing-indicator",
            )

            with gr.Row(elem_id="composer", variant="default"):
                gr.HTML(
                    """
                    <span id="attachment-icon" aria-hidden="true">
                        <svg viewBox="0 0 24 24">
                            <path d="m20.5 11.5-8.6 8.6a5 5 0 0 1-7.1-7.1l9.2-9.2a3.4 3.4 0 0 1 4.8 4.8l-9.2 9.2a1.8 1.8 0 0 1-2.5-2.5l8.5-8.5"></path>
                        </svg>
                    </span>
                    """,
                    elem_id="attachment-wrap",
                )
                question_input = gr.Textbox(
                    placeholder="Tanyakan kebijakan",
                    show_label=False,
                    lines=1,
                    max_lines=5,
                    scale=1,
                    container=False,
                    elem_id="question-input",
                )

                send_button = gr.Button(
                    "↑",
                    variant="secondary",
                    size="sm",
                    elem_id="send-button",
                )

            with gr.Row(elem_id="suggestions", variant="default") as suggestions:
                example_buttons = [
                    gr.Button(f"{icon}  {label}", size="sm")
                    for icon, label, _ in EXAMPLES
                ]

        gr.HTML(
            """
            <div id="disclaimer">
                Hanya menjawab pertanyaan terkait kebijakan perusahaan.
            </div>
            """
        )

    for button, (_, _, question) in zip(example_buttons, EXAMPLES):
        button.click(
            fn=lambda q=question: q,
            inputs=None,
            outputs=question_input,
        )

    outputs = [
        question_input,
        chatbot,
        welcome,
        suggestions,
        typing_indicator,
        send_button,
    ]

    send_button.click(
        fn=submit_question,
        inputs=[question_input, chatbot],
        outputs=outputs,
        concurrency_id="policy-question",
        concurrency_limit=1,
    )

    question_input.submit(
        fn=submit_question,
        inputs=[question_input, chatbot],
        outputs=outputs,
        concurrency_id="policy-question",
        concurrency_limit=1,
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(theme=gr.themes.Base(), css=css)
