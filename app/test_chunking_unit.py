from chunking import chunk_text


def test_chunk_text_returns_multiple_chunks():
    text = "PolicyMind AI. " * 100

    chunks = chunk_text(text, chunk_size=100, overlap=10)

    assert len(chunks) > 1


def test_chunk_length_does_not_exceed_chunk_size():
    text = "PolicyMind AI. " * 100

    chunks = chunk_text(text, chunk_size=100, overlap=10)

    assert all(len(chunk["content"]) <= 100 for chunk in chunks)


def test_chunks_include_required_fields():
    text = "Kebijakan cuti tahunan karyawan XYZ Tech."

    chunks = chunk_text(text)

    assert len(chunks) > 0
    assert all("content" in chunk for chunk in chunks)
    assert all("start_index" in chunk for chunk in chunks)
    assert all(isinstance(chunk["start_index"], int) for chunk in chunks)

def test_chunk_text_uses_default_parameters():
    text = "PolicyMind AI. " * 100

    chunks = chunk_text(text)

    assert len(chunks) > 1
    assert all(len(chunk["content"]) <= 500 for chunk in chunks)