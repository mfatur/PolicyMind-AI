from chunking import chunk_text
from load_documents import load_documents


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
    assert all(len(chunk["content"]) <= 700 for chunk in chunks)


def test_policy_chunks_keep_sections_and_all_source_text():
    text = load_documents()[0]["content"]
    chunks = chunk_text(text)
    contents = [chunk["content"] for chunk in chunks]

    assert "".join(contents) == text

    leave_section = next(
        content
        for content in contents
        if "3. Cuti dan Izin Ketidakhadiran" in content
    )
    assert "Cuti Tahunan 12 hari pertahun" in leave_section
    assert "Cuti Melahirkan 3 bulan" in leave_section

    benefits_section = next(
        content
        for content in contents
        if "4. Fasilitas dan Tunjangan" in content
    )
    assert (
        "Tunjangan Komunikasi\n"
        "Subsidi biaya internet dan komunikasi sebesar\n"
        "Rp 350.000,- per bulan yang dibayarkan\n"
        "bersamaan dengan siklus penggajian bulanan."
    ) in benefits_section
    assert (
        "Anggaran Pengembangan Diri\n"
        "Setiap karyawan mendapatkan dana Rp\n"
        "2.000.000,- per tahun yang dapat diklaim\n"
        "untuk pembelian buku, kursus online, atau\n"
        "sertifikasi profesional."
    ) in benefits_section