from load_documents import load_documents
from chunking import chunk_text


documents = load_documents()

all_chunks = []

for document in documents:
    chunks = chunk_text(document["content"])

    for chunk in chunks:
        all_chunks.append({
            "source": document["source"],
            "content": chunk["content"],
            "start_index": chunk["start_index"],
        })

print(f"Jumlah dokumen: {len(documents)}")
print(f"Jumlah chunk: {len(all_chunks)}")
print(f"\nSumber chunk pertama: {all_chunks[0]['source']}")
print(f"Posisi awal chunk pertama: {all_chunks[0]['start_index']}")
print(f"Isi chunk pertama:\n{all_chunks[0]['content']}")

print("\nPanjang setiap chunk:")

for i, chunk in enumerate(all_chunks, start=1):
    length = len(chunk["content"])
    print(f"Chunk {i}: {length} karakter")

print("\n=== CEK OVERLAP PER DOKUMEN ===")

for document in documents:
    source = document["source"]
    original_text = document["content"]

    source_chunks = sorted(
        [
            chunk
            for chunk in all_chunks
            if chunk["source"] == source
        ],
        key=lambda chunk: chunk["start_index"],
    )

    print(f"\n{source}:")

    for previous, current in zip(
        source_chunks,
        source_chunks[1:],
    ):
        previous_start = previous["start_index"]
        current_start = current["start_index"]

        previous_text = previous["content"]
        current_text = current["content"]

        previous_end = previous_start + len(previous_text)
        overlap_start = current_start
        overlap_end = min(previous_end, current_start + len(current_text))

        if overlap_start < overlap_end:
            previous_overlap = original_text[overlap_start:overlap_end]
            current_overlap = original_text[overlap_start:overlap_end]

            print(
                f"  Overlap: {len(previous_overlap)} karakter "
                f"| Cocok: {previous_overlap == current_overlap}"
            )
        else:
            print("  Tidak ada overlap")

print("\n=== DETAIL POSISI CHUNK CUTI ===")

cuti_chunks = [
    chunk
    for chunk in all_chunks
    if chunk["source"] == "kebijakan_cuti.txt"
]

for chunk in cuti_chunks:
    print("Posisi awal:", chunk["start_index"])
    print("Panjang:", len(chunk["content"]))
    print("Akhir teks:", repr(chunk["content"][-80:]))
    print()

print("\n=== JUMLAH CHUNK PER DOKUMEN ===")

for document in documents:
    count = sum(
        1
        for chunk in all_chunks
        if chunk["source"] == document["source"]
    )

    print(f"{document['source']}: {count} chunk")

print("\n=== CEK BATAS UKURAN ===")

for i, chunk in enumerate(all_chunks, start=1):
    length = len(chunk["content"])

    if length > 500:
        print(f"Chunk {i} terlalu panjang: {length} karakter")

print("Pemeriksaan selesai.")

print("\n=== CEK SUMBER DOKUMEN ===")

for document in documents:
    source = document["source"]
    source_chunks = [
        chunk["content"]
        for chunk in all_chunks
        if chunk["source"] == source
    ]

    print(f"{source}: {len(source_chunks)} chunk")

    
print("\n=== CEK CAKUPAN DOKUMEN ===")

for document in documents:
    source = document["source"]

    source_chunks = [
        chunk["content"]
        for chunk in all_chunks
        if chunk["source"] == source
    ]

    print(f"{source}: {len(source_chunks)} chunk")

    

print("\n=== CEK CAKUPAN TEKS ===")

for document in documents:
    source = document["source"]
    original_text = document["content"]

    source_chunks = [
        chunk["content"]
        for chunk in all_chunks
        if chunk["source"] == source
    ]

    covered_chars = sum(
        len(chunk) for chunk in source_chunks
    )

    print(
        f"{source}: "
        f"{len(source_chunks)} chunk, "
        f"{covered_chars} karakter dalam chunk"
    )

print("\n=== CEK POSISI CHUNK ===")

for document in documents:
    source = document["source"]
    original_text = document["content"]

    source_chunks = [
        chunk["content"]
        for chunk in all_chunks
        if chunk["source"] == source
    ]

    last_position = 0
    valid = True

    for chunk in source_chunks:
        position = original_text.find(chunk, last_position)

        if position == -1:
            valid = False
            break

        last_position = position + len(chunk)

    print(f"{source}: {'OK' if valid else 'PERLU CEK'}")

    print("\n=== VERIFIKASI OVERLAP LANGSUNG ===")

document = documents[0]
chunks = chunk_text(document["content"])

for i in range(len(chunks) - 1):
    current = chunks[i]["content"]
    next_chunk = chunks[i + 1]["content"]

    overlap = 0

    for size in range(1, min(len(current), len(next_chunk)) + 1):
        if current[-size:] == next_chunk[:size]:
            overlap = size

    print(f"Chunk {i + 1} → {i + 2}: {overlap} karakter")

    print("\n=== DEBUG POSISI CHUNK CUTI ===")

document = documents[0]
chunks = sorted(
    [
        chunk
        for chunk in all_chunks
        if chunk["source"] == document["source"]
    ],
    key=lambda chunk: chunk["start_index"],
)

for chunk in chunks:
    start = chunk["start_index"]
    content = chunk["content"]

    print(f"\nPosisi metadata: {start}")
    print(f"Panjang chunk: {len(content)}")
    print(f"Posisi aktual: {document['content'].find(content)}")
    print(f"Awal chunk: {content[:60]!r}")
    
print("\n=== PREVIEW CHUNK ===")

for i, chunk in enumerate(all_chunks, start=1):
    print(f"\n--- Chunk {i} ---")
    print(f"Sumber: {chunk['source']}")
    print(f"Posisi: {chunk['start_index']}")
    print(f"Panjang: {len(chunk['content'])} karakter")
    print(chunk["content"])