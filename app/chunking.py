
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text, chunk_size=700, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        add_start_index=True,
    )

    chunks = splitter.create_documents([text])

    return [
        {
            "content": chunk.page_content,
            "start_index": chunk.metadata["start_index"],
        }
        for chunk in chunks
    ]
