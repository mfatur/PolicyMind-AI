from dotenv import load_dotenv

load_dotenv()

from app.retrieval import retrieve_documents
from app.llm import generate_answer
from langfuse import observe

@observe(name="policy-ask", as_type="chain")
def ask_policy(question):
    documents = retrieve_documents(question, top_k=3)

    context = "\n\n".join(
        f"Sumber: {doc['source']}\n{doc['content']}"
        for doc in documents
    )

    answer = generate_answer(question, context)

    sources = list(dict.fromkeys(
        doc["source"] for doc in documents
    ))

    return {
        "question": question,
        "answer": answer,
        "contexts": [doc["content"] for doc in documents],
        "sources": sources,
    }


if __name__ == "__main__":
    question = "Kalau saya sudah membeli kebutuhan kerja, apakah pasti akan diganti?"

    result = ask_policy(question)

    print("Pertanyaan:", result["question"])
    print("\nJawaban PolicyMind AI:")
    print(result["answer"])
    print("\nSumber dokumen:")
    for source in result["sources"]:
        print("-", source)
    print("\nKonteks yang digunakan:")
    for context in result["contexts"]:
        print("-", context)