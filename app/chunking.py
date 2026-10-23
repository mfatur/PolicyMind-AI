import re

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text, chunk_size=700, overlap=100):
    section_starts = [
        match.start()
        for match in re.finditer(r"(?m)^\d+\.\s+", text)
    ]

    if section_starts:
        boundaries = [0, *section_starts[1:], len(text)]
        return [
            {
                "content": text[start:end],
                "start_index": start,
            }
            for start, end in zip(boundaries, boundaries[1:])
            if start < end
        ]

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
