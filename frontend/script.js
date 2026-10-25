const API_URL = "https://policymind-ai.fastapicloud.dev/ask";
const REQUEST_TIMEOUT_MS = 45000;

const questionForm = document.querySelector("#question-form");
const questionInput = document.querySelector("#question-input");
const sendButton = document.querySelector("#send-button");
const conversation = document.querySelector("#conversation");
const welcomeBlock = document.querySelector("#welcome-block");
const suggestions = document.querySelector("#suggestions");
const typingIndicator = document.querySelector("#typing-indicator");
const conversationControls = document.querySelector("#conversation-controls");
const clearChatButton = document.querySelector("#clear-chat-button");
const copyChatButton = document.querySelector("#copy-chat-button");
const controlFeedback = document.querySelector("#control-feedback");

let conversationState = [];
let activeRequestController = null;
let requestSequence = 0;
let isRequestInProgress = false;
let feedbackTimeoutId = null;

function appendInlineMarkdown(parent, text) {
  const inlinePattern = /(\*\*[^*]+?\*\*|\*[^*]+?\*)/g;
  let lastIndex = 0;

  for (const match of text.matchAll(inlinePattern)) {
    if (match.index > lastIndex) {
      parent.append(document.createTextNode(text.slice(lastIndex, match.index)));
    }

    const isBold = match[0].startsWith("**");
    const content = isBold ? match[0].slice(2, -2) : match[0].slice(1, -1);
    const node = document.createElement(isBold ? "strong" : "em");
    node.textContent = content;
    parent.append(node);
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < text.length) {
    parent.append(document.createTextNode(text.slice(lastIndex)));
  }
}

function renderMarkdown(parent, markdown) {
  const normalized = markdown.replace(/\\([\\`*_{}\[\]()#+\-.!>])/g, "$1");
  const lines = normalized.split(/\r?\n/);
  let paragraphLines = [];
  let list = null;

  const flushParagraph = () => {
    if (paragraphLines.length === 0) {
      return;
    }

    const paragraph = document.createElement("p");
    appendInlineMarkdown(paragraph, paragraphLines.join(" "));
    parent.append(paragraph);
    paragraphLines = [];
  };

  const flushList = () => {
    if (list) {
      parent.append(list);
      list = null;
    }
  };

  for (const line of lines) {
    const listItem = line.match(/^\s*[-*]\s+(.+)$/);
    if (listItem) {
      flushParagraph();
      if (!list) {
        list = document.createElement("ul");
      }
      const item = document.createElement("li");
      appendInlineMarkdown(item, listItem[1]);
      list.append(item);
      continue;
    }

    flushList();
    if (line.trim() === "") {
      flushParagraph();
    } else {
      paragraphLines.push(line.trim());
    }
  }

  flushParagraph();
  flushList();
}

function renderConversation() {
  conversation.replaceChildren();

  for (const entry of conversationState) {
    const message = document.createElement("div");
    const isUser = entry.role === "user";
    message.className = `message message-${entry.role}${entry.isError ? " message-error" : ""}`;

    if (isUser || entry.isError) {
      message.textContent = entry.content;
    } else {
      renderMarkdown(message, entry.content);
      appendSources(message, entry.sources);
    }

    conversation.append(message);
  }
}

function appendSources(message, sources = []) {
  if (sources.length === 0) {
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

function showControlFeedback(message, isError = false) {
  window.clearTimeout(feedbackTimeoutId);
  controlFeedback.textContent = message;
  controlFeedback.classList.toggle("feedback-error", isError);
  if (message) {
    feedbackTimeoutId = window.setTimeout(() => {
      controlFeedback.textContent = "";
      controlFeedback.classList.remove("feedback-error");
    }, 2500);
  }
}

function scrollConversationToBottom() {
  conversation.scrollTop = conversation.scrollHeight;
}

async function askBackend(question, controller) {
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
  if (isRequestInProgress) {
    return;
  }
  if (!normalizedQuestion) {
    questionInput.setCustomValidity("Silakan masukkan pertanyaan terlebih dahulu.");
    questionInput.reportValidity();
    return;
  }

  questionInput.setCustomValidity("");
  welcomeBlock.hidden = true;
  suggestions.hidden = true;
  conversation.hidden = false;
  conversationControls.hidden = false;
  conversationState.push({ role: "user", content: normalizedQuestion });
  renderConversation();
  typingIndicator.hidden = false;
  isRequestInProgress = true;
  const requestId = ++requestSequence;
  const controller = new AbortController();
  activeRequestController = controller;
  sendButton.disabled = true;
  questionInput.disabled = true;
  scrollConversationToBottom();

  try {
    const result = await askBackend(normalizedQuestion, controller);
    if (requestId !== requestSequence) {
      return;
    }
    conversationState.push({
      role: "assistant",
      content: result.answer,
      sources: result.sources,
    });
    renderConversation();
  } catch (error) {
    if (requestId !== requestSequence) {
      return;
    }
    conversationState.push({
      role: "assistant",
      content: error instanceof Error ? error.message : "Terjadi kesalahan yang tidak diketahui.",
      sources: [],
      isError: true,
    });
    renderConversation();
  } finally {
    if (requestId === requestSequence) {
      activeRequestController = null;
      isRequestInProgress = false;
      typingIndicator.hidden = true;
      sendButton.disabled = false;
      questionInput.disabled = false;
      questionInput.value = "";
      questionInput.style.height = "auto";
      questionInput.focus();
      scrollConversationToBottom();
    }
  }
}

function resetConversation() {
  requestSequence += 1;
  activeRequestController?.abort();
  activeRequestController = null;
  isRequestInProgress = false;
  conversationState = [];
  renderConversation();

  conversation.hidden = true;
  conversationControls.hidden = true;
  typingIndicator.hidden = true;
  welcomeBlock.hidden = false;
  suggestions.hidden = false;
  sendButton.disabled = false;
  questionInput.disabled = false;
  questionInput.value = "";
  questionInput.style.height = "auto";
  questionInput.setCustomValidity("");
  showControlFeedback("");
  questionInput.focus();
}

async function copyConversation() {
  if (conversationState.length === 0) {
    showControlFeedback("Belum ada percakapan untuk disalin.", true);
    return;
  }

  const transcript = conversationState.map((entry) => {
    const speaker = entry.role === "user" ? "Pengguna" : "PolicyMind AI";
    const sources = entry.sources?.length
      ? `\n\nSumber dokumen:\n${entry.sources.map((source) => `- ${source}`).join("\n")}`
      : "";
    return `${speaker}:\n${entry.content}${sources}`;
  }).join("\n\n");

  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(transcript);
    } else {
      const temporaryInput = document.createElement("textarea");
      temporaryInput.value = transcript;
      temporaryInput.setAttribute("readonly", "");
      temporaryInput.className = "clipboard-fallback";
      document.body.append(temporaryInput);
      temporaryInput.select();
      const copied = document.execCommand("copy");
      temporaryInput.remove();
      if (!copied) {
        throw new Error("Clipboard tidak tersedia.");
      }
    }
    showControlFeedback("Percakapan berhasil disalin.");
  } catch {
    showControlFeedback("Gagal menyalin percakapan.", true);
  }
}

questionForm.addEventListener("submit", (event) => {
  event.preventDefault();
  submitQuestion(questionInput.value);
});

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    questionForm.requestSubmit();
  }
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

clearChatButton.addEventListener("click", resetConversation);
copyChatButton.addEventListener("click", copyConversation);
