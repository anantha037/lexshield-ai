import json
import pytest
from pathlib import Path
from data.preprocessor import clean_text, chunk_text, detect_section_header
 
 
def test_clean_text_removes_standalone_page_numbers():
    raw     = "Some legal text.\n\n  42  \n\nMore legal text."
    cleaned = clean_text(raw)
    assert "Some legal text." in cleaned
    assert "More legal text." in cleaned
    # 42 should not appear as a standalone word
    assert "42" not in cleaned.split()
 
 
def test_clean_text_fixes_hyphenation():
    raw     = "This is a hyphen-\nated word in legal text."
    cleaned = clean_text(raw)
    assert "hyphenated" in cleaned
 
 
def test_clean_text_removes_gazette_header():
    raw     = "THE GAZETTE OF INDIA\nSection 420. Cheating."
    cleaned = clean_text(raw)
    assert "GAZETTE OF INDIA" not in cleaned
    assert "Section 420" in cleaned
 
 
def test_detect_section_header_ipc():
    text   = "Section 420. Cheating and dishonestly inducing delivery of property."
    result = detect_section_header(text)
    assert result is not None
    assert "420" in result
 
 
def test_detect_section_header_chapter():
    text   = "CHAPTER IV GENERAL EXCEPTIONS"
    result = detect_section_header(text)
    assert result is not None
    assert "CHAPTER" in result
 
 
def test_chunk_text_produces_valid_chunks():
    sample = """
    Section 420. Cheating.
    Whoever cheats and thereby dishonestly induces the person deceived to deliver
    any property to any person, or to make, alter or destroy the whole or any part
    of a valuable security, shall be punished with imprisonment for a term which may
    extend to seven years, and shall also be liable to fine.
 
    Section 302. Punishment for murder.
    Whoever commits murder shall be punished with death, or imprisonment for life,
    and shall also be liable to fine. Murder requires intention to cause death or
    such bodily injury as the offender knows to be likely to cause the death.
    """
    chunks = chunk_text(sample, source="IPC 1860", doc_type="statute")
    assert len(chunks) >= 1
    for chunk in chunks:
        assert "text"     in chunk
        assert "chunk_id" in chunk
        assert "source"   in chunk
        assert chunk["source"]   == "IPC 1860"
        assert chunk["doc_type"] == "statute"
        assert len(chunk["text"]) > 50
 
 
def test_chunks_json_exists_and_is_valid():
    path = Path("data/processed/chunks.json")
    if not path.exists():
        pytest.skip("chunks.json not yet generated — run preprocessor.py first")
 
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
 
    assert isinstance(chunks, list)
    assert len(chunks) >= 100
 
    required = {"chunk_id", "text", "source", "doc_type"}
    for chunk in chunks[:20]:
        assert required.issubset(chunk.keys())
        assert len(chunk["text"].strip()) > 50