const API_URL = "https://policymind-ai.fastapicloud.dev/ask";
const REQUEST_TIMEOUT_MS = 45000;

const questionForm = document.querySelector("#question-form");
const questionInput = document.querySelector("#question-input");
const sendButton = document.querySelector("#send-button");
const conversation = document.querySelector("#conversation");
const welcomeBlock = document.querySelector("#welcome-block");
const suggestions = document.querySelector("#suggestions");
const typingIndicator = document.querySelector("#typing-indicator");

function appendMessage(role, text, isError = false) {
  const message = document.createElement("div");
  message.className = `message message-${role}${isError ? " message-error" : ""}`;
  message.textContent = text;
  conversation.append(message);
  return message;
}

function appendSources(message, sources) {
  if (!sources.length) {
    return;
  }

  const section = document.createElement("section");
  section.className = "sources";

  const heading = document.createElement("p");
  heading.className = "sources-title";
  heading.textContent = "Sumber dokumen";

  const list = document.createElement("ul");
  for (const source of sources) {
    const item = document.createElement("li");
    item.textContent = source;
    list.append(item);
  }

  section.append(heading, list);
  message.append(section);
}

function scrollConversationToBottom() {
  conversation.scrollTop = conversation.scrollHeight;
}

async function askBackend(question) {
  const controller = new AbortController();
  let timedOut = false;
  const timeoutId = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, REQUEST_TIMEOUT_MS);

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question }),
      signal: controller.signal,
    });

    let result;
    try {
      result = await response.json();
    } catch {
      throw new Error("Backend mengirim respons JSON yang tidak valid.");
    }

    if (!response.ok) {
      throw new Error(`Backend mengembalikan error (HTTP ${response.status}).`);
    }
    if (!result || typeof result !== "object" || Array.isArray(result)) {
      throw new Error("Format respons backend tidak valid.");
    }
    if (typeof result.answer !== "string") {
      throw new Error("Respons backend tidak berisi jawaban yang valid.");
    }

    const sources = result.sources === undefined ? [] : result.sources;
    if (!Array.isArray(sources) || !sources.every((source) => typeof source === "string")) {
      throw new Error("Daftar sumber dari backend tidak valid.");
    }

    return { answer: result.answer, sources };
  } catch (error) {
    if (timedOut) {
      throw new Error("Permintaan terlalu lama. Silakan coba lagi.");
    }
    if (error instanceof TypeError) {
      throw new Error("Tidak dapat terhubung ke backend. Periksa koneksi dan konfigurasi CORS.");
    }
    throw error;
  } finally {
    window.clearTimeout(timeoutId);
  }
}

async function submitQuestion(question) {
  const normalizedQuestion = question.trim();
  if (!normalizedQuestion) {
    questionInput.setCustomValidity("Silakan masukkan pertanyaan terlebih dahulu.");
    questionInput.reportValidity();
    return;
  }

  questionInput.setCustomValidity("");
  welcomeBlock.hidden = true;
  suggestions.hidden = true;
  conversation.hidden = false;
  appendMessage("user", normalizedQuestion);
  typingIndicator.hidden = false;
  sendButton.disabled = true;
  questionInput.disabled = true;
  scrollConversationToBottom();

  try {
    const result = await askBackend(normalizedQuestion);
    const assistantMessage = appendMessage("assistant", result.answer);
    appendSources(assistantMessage, result.sources);
  } catch (error) {
    appendMessage(
      "assistant",
      error instanceof Error ? error.message : "Terjadi kesalahan yang tidak diketahui.",
      true,
    );
  } finally {
    typingIndicator.hidden = true;
    sendButton.disabled = false;
    questionInput.disabled = false;
    questionInput.value = "";
    questionInput.focus();
    scrollConversationToBottom();
  }
}

questionForm.addEventListener("submit", (event) => {
  event.preventDefault();
  submitQuestion(questionInput.value);
});

questionInput.addEventListener("input", () => {
  questionInput.setCustomValidity("");
  questionInput.style.height = "auto";
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 120)}px`;
});

suggestions.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-question]");
  if (!button) {
    return;
  }

  questionInput.value = button.dataset.question;
  submitQuestion(questionInput.value);
});
