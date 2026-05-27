"""
LexShield AI — Document Analysis API  (Session 7 — CV Pipeline Upgrade)
=========================================================================
POST /api/v1/document/analyze  — full pipeline with proactive rights_alerts
POST /api/v1/document/query    — Q&A on a previously analyzed document

/analyze pipeline
-----------------
1. Extract text  (cv/pipeline.py handles ALL formats — digital + scanned + Indic)
2. Classify      (InLegalBERT -> document type + confidence)
3. NER           (spaCy + regex -> structured entities)
4. Risk Score    (clause-level risk with legal references)
5. Rights Alerts (rule-based proactive rights violation detection)
6. RAG Q&A       (optional: pass extracted text to RAG pipeline)

/query endpoint
---------------
Takes OCR-extracted doc_text + a user question, constructs a Groq prompt
with the document as context, runs multilingual pipeline if needed,
stores turn in session memory.

CHANGES in Session 7 (CV Pipeline Upgrade):
  1. _extract_pdf() rewritten: removed duplicate inline fitz.open() call.
     Now routes ALL PDF extraction through cv.pipeline.extract_text_from_pdf_bytes().
     This means digital PDFs now also benefit from pdfplumber table extraction
     and correct scanned-vs-digital detection — previously the inline fitz call
     bypassed all of that.

  2. _ocr_pdf() removed: was only needed to call cv.pipeline for scanned PDFs.
     extract_text_from_pdf_bytes() now handles both digital and scanned in one call.

  3. _extract_image() updated: now passes source_language to extract_text_from_image()
     so Malayalam/Hindi image documents use the correct Surya language model.

  4. /analyze endpoint: added `language` query parameter (default "en").
     Pass language="ml" for Malayalam, language="hi" for Hindi, etc.
     Propagated to all extraction helpers.

  5. DocumentAnalysisResponse: added two new optional fields:
       - engine_used (str): "pymupdf" | "surya" | "tesseract"
       - ocr_confidence (float): 0.0-1.0 quality signal from Surya.
         1.0 = digital PDF (no OCR). < 0.35 = potentially unreliable.

  NOTHING else changed. All other logic, response models, /query, /save-session,
  _chunk_text, _build_ephemeral_context — all identical to Session 6.

BUG FIX (Session 6 — preserved):
  Replaced hardcoded doc_text[:3000] truncation with an ephemeral
  in-memory ChromaDB vector store scoped to each /query request.

Run:
  uvicorn api.main:app --reload --port 8000

Test Malayalam upload:
  curl -s -X POST http://localhost:8000/api/v1/document/analyze \
    -F "file=@kerala_hc_order.pdf" -F "language=ml" | python -m json.tool
"""

import io
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/document", tags=["document"])

MAX_FILE_SIZE_MB = 10
SUPPORTED_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".txt"}

# ── /query ephemeral RAG constants ────────────────────────────────────────────
_CHUNK_SIZE      = 200   # approximate characters per sentence-boundary chunk
_CHUNK_OVERLAP   = 30    # characters of overlap between consecutive chunks
_TOP_K_CHUNKS    = 5     # how many chunks to pass to the LLM as context
_MAX_CHUNK_CHARS = 400   # hard cap per chunk sent to LLM prompt


# ── Response models ───────────────────────────────────────────────────────────
# ALL UNCHANGED from Session 6 except DocumentAnalysisResponse (2 new fields)

class ClassificationResult(BaseModel):
    label:      int
    label_name: str
    confidence: float
    all_scores: dict[str, float] = {}
    warning:    Optional[str]    = None


class EntityCounts(BaseModel):
    persons: int = 0; organizations: int = 0; dates: int = 0
    locations: int = 0; monetary: int = 0; ipc_sections: int = 0
    case_numbers: int = 0; acts: int = 0


class EntitiesModel(BaseModel):
    persons: list[str] = []; organizations: list[str] = []
    dates: list[str] = []; locations: list[str] = []
    monetary: list[str] = []; ipc_sections: list[str] = []
    case_numbers: list[str] = []; acts: list[str] = []
    entity_counts: EntityCounts = EntityCounts()


class ClauseRiskModel(BaseModel):
    clause_number: int
    clause_text:   str
    score:         int
    risk_level:    str
    flags:         list[str] = []
    legal_refs:    list[str] = []
    explanation:   str       = ""


class RiskModel(BaseModel):
    overall_score:   int
    risk_level:      str
    high_risk_count: int
    summary:         str
    clause_risks:    list[ClauseRiskModel] = []


class CitationModel(BaseModel):
    source_number:    int
    source:           str
    section:          str             = ""
    section_title:    str             = ""
    chapter:          str             = ""
    preview:          str             = ""
    relevance_score:  Optional[float] = None
    retrieval_source: str             = ""
    doc_type:         str             = ""


class LegalExplanationModel(BaseModel):
    answer:            str
    citations:         list[CitationModel] = []
    sources_consulted: int                 = 0
    synthesis_note:    str                 = ""
    grounding_warning: Optional[str]       = None


class RightsAlertModel(BaseModel):
    right:     str
    violation: str
    section:   str
    severity:  str   # "info" | "low" | "medium" | "high" | "critical"


class DocumentAnalysisResponse(BaseModel):
    filename:           str
    text:               str
    word_count:         int
    ocr_used:           bool
    page_count:         int
    classification:     ClassificationResult
    entities:           EntitiesModel
    risk:               RiskModel
    rights_alerts:      list[RightsAlertModel]          = []
    legal_explanation:  Optional[LegalExplanationModel] = None
    warning:            Optional[str]                   = None
    # ── NEW in Session 7 ──────────────────────────────────────────────────────
    engine_used:        str   = "unknown"
    # "pymupdf" = digital PDF (best quality, no OCR)
    # "surya"   = Surya OCR (scanned or Indic language documents)
    # "tesseract" = emergency fallback
    ocr_confidence:     float = 1.0
    # 1.0 = digital PDF (no OCR, perfect).
    # 0.0-1.0 = OCR quality. If < 0.35, analysis may be unreliable.
    # Frontend can show a warning badge when this is below 0.5.


class DocQueryRequest(BaseModel):
    doc_text:   str
    question:   str
    session_id: str
    language:   Optional[str] = "en"


class DocQueryResponse(BaseModel):
    answer:               str
    applicable_sections:  list[str] = []
    risk_note:            str       = ""
    session_id:           str


class DocSaveSessionRequest(BaseModel):
    """
    Lightweight persist of a document analysis result to session memory.
    No LLM call — writes one session row + two turns to SQLite directly.
    """
    filename:    str
    doc_type:    str
    risk_level:  str
    risk_score:  int
    summary:     str                    # risk.summary from /analyze
    confidence:  float
    session_id:  Optional[str] = None  # if None, a new UUID is generated
    user_id:     Optional[str] = None  # pass when user is authenticated


class DocSaveSessionResponse(BaseModel):
    session_id:    str
    first_message: str


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT EXTRACTION HELPERS  (CHANGED in Session 7)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_pdf(
    file_bytes: bytes,
    language:   str = "en",
) -> tuple[str, int, bool, str, float]:
    """
    Extract text from a PDF.

    CHANGED in Session 7:
      - Removed inline fitz.open() (was bypassing cv/pipeline.py for digital PDFs)
      - Now routes ALL PDFs through cv.pipeline.extract_text_from_pdf_bytes()
      - This single function handles: digital → PyMuPDF+pdfplumber,
        scanned → Surya OCR (page-by-page, RAM-safe), fallback → Tesseract
      - Returns: (text, page_count, ocr_used, engine_used, ocr_confidence)

    PREVIOUSLY returned: (text, page_count, ocr_used)  — 3 values
    NOW returns:         (text, page_count, ocr_used, engine_used, confidence) — 5 values
    Only called in /analyze below — no other callers.
    """
    try:
        from cv.pipeline import extract_text_from_pdf_bytes, extract_text

        # Get full result dict with engine metadata
        import tempfile, os
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            result = extract_text(tmp_path, source_language=language)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        text           = result.get("text", "")
        engine_used    = result.get("engine_used", "unknown")
        ocr_confidence = result.get("ocr_confidence", 1.0)
        ocr_used       = engine_used != "pymupdf"

        # Get page count separately (lightweight — just opens PDF header)
        page_count = 1
        try:
            import fitz
            doc        = fitz.open(stream=file_bytes, filetype="pdf")
            page_count = len(doc)
            doc.close()
        except Exception:
            pass

        if not result.get("success") and not text.strip():
            raise HTTPException(
                status_code=422,
                detail="Could not extract text from this PDF. "
                       "The file may be corrupted or password-protected."
            )

        return text, page_count, ocr_used, engine_used, ocr_confidence

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {e}")


def _extract_image(
    file_bytes: bytes,
    language:   str = "en",
) -> tuple[str, int, bool, str, float]:
    """
    Extract text from an image file via OCR.

    CHANGED in Session 7:
      - Now passes source_language to extract_text_from_image()
        so Surya uses the correct language model (was always defaulting to "en")
      - Returns 5 values now (same shape as _extract_pdf for consistency)
        (text, page_count, ocr_used, engine_used, ocr_confidence)

    PREVIOUSLY returned: (text, page_count, ocr_used)
    """
    try:
        from cv.pipeline import extract_text_from_image, preprocess_image
        from PIL import Image
        import numpy as np
        import cv2

        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        cv_img  = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Use Surya directly if available for best multilingual results
        try:
            from cv.pipeline import _SURYA_AVAILABLE, _surya_ocr_image, _clean_extracted_text, _MIN_OCR_CONFIDENCE
            if _SURYA_AVAILABLE:
                text, confidence = _surya_ocr_image(cv_img, language)
                text             = _clean_extracted_text(text)
                engine_used      = "surya"
                return text, 1, True, engine_used, confidence
        except Exception:
            pass

        # Fallback: preprocess + extract_text_from_image (Tesseract)
        preprocessed = preprocess_image(cv_img)
        text         = extract_text_from_image(preprocessed, source_language=language)
        return text, 1, True, "tesseract", 0.5

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image OCR failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# EPHEMERAL RAG HELPERS  (COMPLETELY UNCHANGED from Session 6)
# ═══════════════════════════════════════════════════════════════════════════════

def _chunk_text(
    text: str,
    chunk_size: int = _CHUNK_SIZE,
    overlap:    int = _CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping chunks at sentence boundaries.
    UNCHANGED from Session 6.
    """
    import re

    if not text or not text.strip():
        return []

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    chunks:        list[str] = []
    current:       list[str] = []
    current_len:   int       = 0
    last_sentence: str       = ""

    for sentence in sentences:
        if len(sentence) > chunk_size * 2:
            words     = sentence.split()
            sub_chunk = []
            sub_len   = 0
            for word in words:
                sub_chunk.append(word)
                sub_len += len(word) + 1
                if sub_len >= chunk_size:
                    chunks.append(" ".join(sub_chunk))
                    sub_chunk = []
                    sub_len   = 0
            if sub_chunk:
                sentence = " ".join(sub_chunk)
            else:
                continue

        if current_len + len(sentence) > chunk_size and current:
            chunks.append(" ".join(current))
            last_sentence = current[-1] if current else ""
            current       = [last_sentence, sentence] if last_sentence else [sentence]
            current_len   = len(last_sentence) + len(sentence) + 1
        else:
            current.append(sentence)
            current_len += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]


def _build_ephemeral_context(
    doc_text: str,
    question: str,
    top_k:    int = _TOP_K_CHUNKS,
) -> str:
    """
    Core of the Session 6 Bug Fix.
    Creates a temporary in-memory ChromaDB collection, retrieves top_k chunks
    most relevant to the question, destroys the collection.
    COMPLETELY UNCHANGED from Session 6.
    """
    fallback = doc_text[:3000]

    try:
        from rag.embedder import embedder
        import chromadb

        chunks = _chunk_text(doc_text)

        if not chunks:
            return fallback

        if len(doc_text) <= 3000 and len(chunks) <= top_k:
            return doc_text

        ephemeral_client = chromadb.EphemeralClient()
        collection_name  = f"doc_query_{uuid.uuid4().hex}"
        collection = ephemeral_client.create_collection(
            name     = collection_name,
            metadata = {"hnsw:space": "cosine"},
        )

        try:
            chunk_embeddings = embedder.embed(
                chunks,
                batch_size    = 8,
                show_progress = False,
            )
            collection.add(
                ids        = [f"chunk_{i}" for i in range(len(chunks))],
                embeddings = chunk_embeddings,
                documents  = chunks,
            )
            question_embedding = embedder.embed_single(question)
            results            = collection.query(
                query_embeddings = [question_embedding],
                n_results        = min(top_k, len(chunks)),
                include          = ["documents", "distances"],
            )
            retrieved_chunks: list[str] = (
                results["documents"][0] if results["documents"] else []
            )

            if not retrieved_chunks:
                return fallback

            context_parts = []
            for i, chunk in enumerate(retrieved_chunks, 1):
                safe_chunk = chunk[:_MAX_CHUNK_CHARS]
                context_parts.append(f"[Excerpt {i}]\n{safe_chunk}")

            return "\n\n".join(context_parts)

        finally:
            try:
                ephemeral_client.delete_collection(collection_name)
            except Exception:
                pass

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "Ephemeral RAG fallback to [:3000] for /query — reason: %s", e
        )
        return fallback


# ═══════════════════════════════════════════════════════════════════════════════
# /analyze ENDPOINT  (CHANGED: language param + 5-tuple unpack)
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/analyze", response_model=DocumentAnalysisResponse)
async def analyze_document(
    file:     UploadFile = File(...),
    run_rag:  bool       = Query(
        default     = False,
        description = "Also run RAG pipeline for legal explanation "
                      "(uses Groq API — slower)",
    ),
    # ── NEW in Session 7 ──────────────────────────────────────────────────────
    language: str = Query(
        default     = "en",
        description = (
            "Language of the uploaded document. "
            "Used to select the correct OCR model for scanned/image files. "
            "Options: en (English), ml (Malayalam), hi (Hindi), "
            "ta (Tamil), te (Telugu), kn (Kannada). "
            "Digital PDFs with embedded text do not need this — "
            "it only affects scanned documents and images."
        ),
    ),
):
    """
    Full document analysis pipeline with proactive rights alerting.

    Steps:
      1. Extract text from uploaded file (cv/pipeline.py — all engines unified)
      2. Classify document type (InLegalBERT, 15 categories)
      3. Extract entities (spaCy + custom regex NER)
      4. Score legal risk per clause
      5. Detect rights violations (rule-based, zero LLM cost)
      6. (Optional) RAG legal explanation — set run_rag=true

    Supported: PDF, JPEG, PNG, TIFF, BMP, TXT (max 10 MB)

    Examples:
      # English document (default)
      curl -s -X POST http://localhost:8000/api/v1/document/analyze \\
        -F "file=@contract.pdf" | python -m json.tool

      # Malayalam scanned document
      curl -s -X POST "http://localhost:8000/api/v1/document/analyze?language=ml" \\
        -F "file=@kerala_hc_order.pdf" | python -m json.tool

      # Hindi document with RAG
      curl -s -X POST "http://localhost:8000/api/v1/document/analyze?language=hi&run_rag=true" \\
        -F "file=@fir_hindi.pdf" | python -m json.tool
    """
    # ── Read file ─────────────────────────────────────────────────────────────
    file_bytes = await file.read()
    size_mb    = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code = 413,
            detail      = f"File too large ({size_mb:.1f} MB). Max: {MAX_FILE_SIZE_MB} MB",
        )

    filename = file.filename or "uploaded_file"
    suffix   = Path(filename).suffix.lower()
    warning  = None

    # ── Step 1: Extract text ──────────────────────────────────────────────────
    # All extraction now returns 5 values: (text, page_count, ocr_used, engine_used, ocr_confidence)
    if suffix == ".pdf" or (file.content_type or "").startswith("application/pdf"):
        text, page_count, ocr_used, engine_used, ocr_confidence = _extract_pdf(
            file_bytes, language=language
        )

    elif suffix == ".txt" or (file.content_type or "").startswith("text/"):
        text           = file_bytes.decode("utf-8", errors="replace")
        page_count     = 1
        ocr_used       = False
        engine_used    = "plain_text"
        ocr_confidence = 1.0

    elif suffix in (".jpg", ".jpeg", ".png", ".tiff", ".bmp"):
        text, page_count, ocr_used, engine_used, ocr_confidence = _extract_image(
            file_bytes, language=language
        )

    else:
        raise HTTPException(
            status_code = 415,
            detail      = f"Unsupported type: {suffix}. Use PDF, image, or TXT.",
        )

    # ── OCR quality warning ───────────────────────────────────────────────────
    if not text or len(text.strip()) < 10:
        warning = "Very little text extracted. Document may be blank or image-only."
        text    = text or ""

    elif ocr_used and ocr_confidence < 0.35:
        warning = (
            f"Low OCR confidence ({ocr_confidence:.0%}). "
            "This document may be poorly scanned. "
            "Analysis results may be unreliable. "
            "Try uploading a higher-quality scan."
        )

    # ── Step 2: Classification ────────────────────────────────────────────────
    from models.classifier import classifier
    clf_result     = classifier.predict(text)
    classification = ClassificationResult(
        label      = clf_result.get("label",      -1),
        label_name = clf_result.get("label_name", "unknown"),
        confidence = clf_result.get("confidence", 0.0),
        all_scores = clf_result.get("all_scores", {}),
        warning    = clf_result.get("warning"),
    )
    doc_type = classification.label_name

    # ── Step 3: NER ───────────────────────────────────────────────────────────
    from nlp.ner_pipeline import extract_entities
    ent_result = extract_entities(text)
    ent_dict   = ent_result.to_dict()
    entities   = EntitiesModel(
        persons       = ent_dict.get("persons",       []),
        organizations = ent_dict.get("organizations", []),
        dates         = ent_dict.get("dates",         []),
        locations     = ent_dict.get("locations",     []),
        monetary      = ent_dict.get("monetary",      []),
        ipc_sections  = ent_dict.get("ipc_sections",  []),
        case_numbers  = ent_dict.get("case_numbers",  []),
        acts          = ent_dict.get("acts",          []),
        entity_counts = EntityCounts(**ent_dict.get("entity_counts", {})),
    )

    # ── Step 4: Risk scoring ──────────────────────────────────────────────────
    from models.risk_scorer import risk_scorer, _to_0_100
    doc_risk   = risk_scorer.score_document(text, doc_type=doc_type)
    risk_model = RiskModel(
        overall_score   = _to_0_100(doc_risk.overall_score),
        risk_level      = doc_risk.risk_level,
        high_risk_count = doc_risk.high_risk_count,
        summary         = doc_risk.summary,
        clause_risks    = [
            ClauseRiskModel(
                clause_number = cr.clause_number,
                clause_text   = cr.clause_text,
                score         = _to_0_100(cr.score),
                risk_level    = cr.risk_level,
                flags         = cr.flags,
                legal_refs    = cr.legal_refs,
                explanation   = cr.explanation,
            )
            for cr in doc_risk.clause_risks
            if cr.score > 0
        ],
    )

    # ── Step 5: Proactive rights alerts ───────────────────────────────────────
    from models.risk_scorer import detect_rights_violations
    raw_alerts    = detect_rights_violations(doc_type, ent_dict, text)
    rights_alerts = [RightsAlertModel(**a) for a in raw_alerts]

    # ── Step 6: RAG (optional) ────────────────────────────────────────────────
    legal_explanation = None
    if run_rag and text.strip():
        try:
            from rag.pipeline import rag_pipeline
            doc_label  = doc_type.replace("_", " ")
            acts_found = entities.acts[:2]
            secs_found = entities.ipc_sections[:2]
            focus      = " ".join(acts_found + secs_found)
            rag_query  = (
                f"Explain the key legal provisions and rights in a {doc_label}. "
                f"{focus}"
            ).strip()

            rag_result        = rag_pipeline.query(rag_query)
            legal_explanation = LegalExplanationModel(
                answer            = rag_result.answer_text,
                citations         = [
                    CitationModel(
                        source_number    = c.source_number,
                        source           = c.source,
                        section          = c.section,
                        section_title    = c.section_title,
                        chapter          = c.chapter,
                        preview          = c.preview,
                        relevance_score  = c.relevance_score,
                        retrieval_source = c.retrieval_source,
                        doc_type         = c.doc_type,
                    )
                    for c in rag_result.citations
                ],
                sources_consulted = rag_result.sources_consulted,
                synthesis_note    = rag_result.synthesis_note,
                grounding_warning = rag_result.grounding_warning,
            )
        except Exception as e:
            warning = (warning or "") + f" RAG failed: {e}"

    return DocumentAnalysisResponse(
        filename          = filename,
        text              = text[:5000],    # response payload trim — intentional
        word_count        = len(text.split()),
        ocr_used          = ocr_used,
        page_count        = page_count,
        classification    = classification,
        entities          = entities,
        risk              = risk_model,
        rights_alerts     = rights_alerts,
        legal_explanation = legal_explanation,
        warning           = warning,
        # ── NEW in Session 7 ─────────────────────────────────────────────────
        engine_used       = engine_used,
        ocr_confidence    = round(ocr_confidence, 3),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# /query ENDPOINT  (COMPLETELY UNCHANGED from Session 6)
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/query", response_model=DocQueryResponse)
def document_query(req: DocQueryRequest):
    """
    Ask a question about a previously analyzed document.
    UNCHANGED from Session 6.

    curl -s -X POST http://localhost:8000/api/v1/document/query \\
      -H "Content-Type: application/json" \\
      -d '{"doc_text":"THIS RENTAL AGREEMENT...","question":"Is this notice valid?","session_id":"<sid>"}' \\
      | python -m json.tool
    """
    if not req.doc_text.strip():
        raise HTTPException(status_code=400, detail="doc_text must not be empty")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    question_en   = req.question.strip()
    detected_lang = req.language or "en"

    if detected_lang not in ("en", "english", None, ""):
        try:
            from agents.multilingual_agent import multilingual_agent
            question_en = multilingual_agent.translate_to_english(req.question.strip())
        except Exception:
            pass

    doc_excerpt = _build_ephemeral_context(req.doc_text, question_en)

    prompt = (
        "You are LexShield AI, an expert in Indian law.\n\n"
        "The user has uploaded the following legal document. "
        "Answer their question based on this document and your Indian legal knowledge. "
        "Be specific, cite the relevant clauses from the document where applicable, "
        "and mention any relevant Indian law sections.\n\n"
        f"Document:\n{doc_excerpt}\n\n"
        f"Question: {question_en}\n\n"
        "Provide:\n"
        "1. A clear answer to the question\n"
        "2. The relevant clauses or sections from the document\n"
        "3. Any legal risk the user should be aware of\n\n"
        "Format your response as:\n"
        "ANSWER: <your answer>\n"
        "SECTIONS: <comma-separated list of applicable sections/clauses>\n"
        "RISK NOTE: <any risk or caution>\n"
    )

    answer_text         = ""
    applicable_sections = []
    risk_note           = ""

    try:
        from rag.llm import llm
        raw_response = llm.generate(prompt, max_tokens=600, temperature=0.2)

        lines           = raw_response.strip().split("\n")
        current_section = None
        answer_lines    = []
        sections_lines  = []
        risk_lines      = []

        for line in lines:
            line_upper = line.strip().upper()
            if line_upper.startswith("ANSWER:"):
                current_section = "answer"
                rest = line[line.upper().find("ANSWER:") + 7:].strip()
                if rest:
                    answer_lines.append(rest)
            elif line_upper.startswith("SECTIONS:"):
                current_section = "sections"
                rest = line[line.upper().find("SECTIONS:") + 9:].strip()
                if rest:
                    sections_lines.append(rest)
            elif line_upper.startswith("RISK NOTE:"):
                current_section = "risk"
                rest = line[line.upper().find("RISK NOTE:") + 10:].strip()
                if rest:
                    risk_lines.append(rest)
            elif current_section == "answer":
                answer_lines.append(line)
            elif current_section == "sections":
                sections_lines.append(line)
            elif current_section == "risk":
                risk_lines.append(line)

        answer_text = " ".join(answer_lines).strip() or raw_response.strip()
        if sections_lines:
            raw_secs            = " ".join(sections_lines)
            applicable_sections = [s.strip() for s in raw_secs.split(",") if s.strip()]
        risk_note = " ".join(risk_lines).strip()

    except Exception as e:
        answer_text = f"Unable to process query: {e}"

    if detected_lang not in ("en", "english", None, "") and answer_text:
        try:
            from agents.multilingual_agent import multilingual_agent
            answer_text = multilingual_agent.translate_to_source(answer_text, detected_lang)
        except Exception:
            pass

    try:
        from agents.memory import session_memory
        session_memory.ensure_session(req.session_id)
        session_memory.add_turn(req.session_id, "user",      req.question.strip(), intent="document_query")
        session_memory.add_turn(req.session_id, "assistant", answer_text,          intent="document_query")
    except Exception:
        pass

    return DocQueryResponse(
        answer              = answer_text,
        applicable_sections = applicable_sections,
        risk_note           = risk_note,
        session_id          = req.session_id,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# /save-session ENDPOINT  (COMPLETELY UNCHANGED from Session 6)
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/save-session", response_model=DocSaveSessionResponse)
def document_save_session(req: DocSaveSessionRequest):
    """
    Persist a document analysis result to session memory with zero LLM cost.
    UNCHANGED from Session 6.

    curl -s -X POST http://localhost:8000/api/v1/document/save-session \\
      -H "Content-Type: application/json" \\
      -d '{"filename":"contract.pdf","doc_type":"rental_agreement","risk_level":"high",
           "risk_score":72,"summary":"High-risk contract.","confidence":0.91}' \\
      | python -m json.tool
    """
    from agents.memory import session_memory

    sid = session_memory.ensure_session(req.session_id)

    if req.user_id:
        session_memory.link_session_to_user(sid, req.user_id)

    first_message = f"Document: {req.filename}"

    session_memory.add_turn(
        sid, "user", first_message, intent="document_analysis",
    )

    summary_content = (
        f"[DOCUMENT ANALYSIS]\n"
        f"File: {req.filename}\n"
        f"Type: {req.doc_type.replace('_', ' ').title()}\n"
        f"Risk Level: {req.risk_level.upper()} ({min(req.risk_score, 100)}/100)\n"
        f"Confidence: {round(req.confidence * 100)}%\n\n"
        f"{req.summary}"
    )
    session_memory.add_turn(
        sid, "assistant", summary_content, intent="document_analysis",
    )

    return DocSaveSessionResponse(session_id=sid, first_message=first_message)