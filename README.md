# PolicyMind AI

**A Retrieval-Augmented Generation (RAG) system for answering questions about company policies.**

PolicyMind AI is a learning and portfolio project designed to retrieve relevant information from company policy documents and generate answers grounded in the retrieved context.

The project explores document processing, semantic search, reranking, LLM-based answer generation, and RAG evaluation.

> **Project status:** In development. The policy documents currently used for testing are synthetic and intended for learning and experimentation.

## Key Features

- **Document processing:** Loads company policy documents and splits them into manageable text chunks.
- **Semantic retrieval:** Uses sentence embeddings and ChromaDB to retrieve relevant policy information.
- **Reranking:** Uses a CrossEncoder model to rerank retrieved candidates.
- **LLM-based generation:** Uses the Groq API to generate answers based on retrieved context.
- **RAG evaluation:** Uses RAGAS metrics to evaluate answer faithfulness, answer relevancy, context precision, and context recall.
- **Evaluation result validation:** Includes a script to validate saved evaluation results.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| Embeddings | Sentence Transformers |
| Vector database | ChromaDB |
| Reranker | CrossEncoder |
| LLM inference | Groq API |
| Evaluation | RAGAS |
| Evaluation embeddings | Hugging Face Sentence Transformers |

## RAG Pipeline

1. Load company policy documents.
2. Split documents into text chunks.
3. Generate embeddings for document chunks.
4. Retrieve candidate chunks from ChromaDB.
5. Rerank candidates using a CrossEncoder.
6. Build a context from the highest-ranked chunks.
7. Generate an answer using the LLM.
8. Evaluate answer quality using RAGAS.

## RAGAS Evaluation

The initial evaluation uses three questions based on synthetic company policy documents.

| Metric | Average Score |
|---|---:|
| Faithfulness | 1.0000 |
| Answer Relevancy | 0.9311 |
| Context Precision | 1.0000 |
| Context Recall | 1.0000 |

**Evaluation notes:**
- The results represent an initial test with only three questions.
- The scores are specific to the current synthetic dataset and evaluation configuration.
- High scores on this small dataset do not establish production readiness or guarantee performance on real company policies.
- More questions and varied test cases are needed for a more robust evaluation.

### Evaluation Scripts

- `evaluation/evaluate_ragas.py` — runs the RAGAS evaluation.
- `evaluation/check_results.py` — validates saved evaluation results.
- `evaluation/data/eval_dataset.json` — contains the evaluation dataset.
- `evaluation/data/ragas_results.json` — stores evaluation scores and summary.
- `evaluation/data/ragas_results_backup.json` — backup of evaluation results.

Run the result validation script from the project root:

```bash
python evaluation/check_results.py
```

Run the RAGAS evaluation when you intentionally want to perform or resume an evaluation:

```bash
python evaluation/evaluate_ragas.py
```

Evaluation may make API calls and consume model quota.

## Running the API and Demo UI

Run the backend from the project root:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

In a separate terminal, install the optional UI dependency and start the Gradio
demo:

```bash
uv run --extra ui python demo.py
```

The UI sends questions to the backend's `/ask` endpoint over HTTP. By default,
it connects to `http://127.0.0.1:8000`; set `POLICYMIND_API_URL` in the UI
process environment to use a different backend URL.

## Project Structure

```text
PolicyMind-AI/
├── app/
│   ├── load_documents.py
│   ├── chunking.py
│   ├── embedding.py
│   ├── vector_store.py
│   ├── retrieval.py
│   ├── llm.py
│   ├── rag_pipeline.py
│   ├── inspect_chunk.py
│   ├── test_chunking.py
│   └── test_reranker.py
├── data/
│   ├── raw/
│   └── chroma_db/
├── evaluation/
│   ├── evaluate_ragas.py
│   ├── check_results.py
│   └── data/
│       ├── eval_dataset.json
│       ├── ragas_results.json
│       └── ragas_results_backup.json
├── .env
└── README.md
```

The structure above describes the intended layout; adjust it if the local project differs.

## Synthetic Data Disclaimer

The company policy documents used in this project are synthetic examples created for learning and testing. They do not represent actual company policies or automatically applicable legal requirements.

## Future Improvements

- Expand the evaluation dataset with more questions and edge cases.
- Test retrieval quality across different policy categories.
- Add API endpoints and automated tests.
- Improve handling of questions that cannot be answered from the available documents.
- Add experiment tracking and observability.
- Evaluate the system on a broader and more diverse dataset.

## Learning Objectives

This project is part of my hands-on learning journey in AI engineering, with a focus on RAG systems, retrieval quality, LLM evaluation, and reliable AI application development.