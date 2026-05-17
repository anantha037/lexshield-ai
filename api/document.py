"""
LexShield AI — Document Analysis API  (Session 6 — Final)
===========================================================
POST /api/v1/document/analyze  — full pipeline with proactive rights_alerts
POST /api/v1/document/query    — Q&A on a previously analyzed document

/analyze pipeline
-----------------
1. Extract text  (PyMuPDF for digital PDFs, OCR for scanned/images)
2. Classify      (XGBoost → document type + confidence)
3. NER           (spaCy + regex → structured entities)
4. Risk Score    (clause-level risk with legal references)
5. Rights Alerts (rule-based proactive rights violation detection — NEW)
6. RAG Q&A       (optional: pass extracted text to RAG pipeline)

/query endpoint
---------------
Takes OCR-extracted doc_text + a user question, constructs a Groq prompt
with the document as context, runs multilingual pipeline if needed,
stores turn in session memory.

Response additions
-------------------
DocumentAnalysisResponse now includes:
  "rights_alerts": [
    {
      "right":     "Right to Provident Fund",
      "violation": "Contract does not mention EPF contribution",
      "section":   "EPF Act 1952",
      "severity":  "high"
    },
    ...
  ]
"""

import io
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/document", tags=["document"])

MAX_FILE_SIZE_MB = 10
SUPPORTED_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".txt"}


# ── Response models ───────────────────────────────────────────────────────────

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
    section:          str           = ""
    section_title:    str           = ""
    chapter:          str           = ""
    preview:          str           = ""
    relevance_score:  Optional[float] = None
    retrieval_source: str           = ""
    doc_type:         str           = ""


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
    rights_alerts:      list[RightsAlertModel] = []   # NEW — proactive rights violation alerts
    legal_explanation:  Optional[LegalExplanationModel] = None
    warning:            Optional[str]                   = None


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


# ── Text extraction helpers ───────────────────────────────────────────────────

def _extract_pdf(file_bytes: bytes) -> tuple[str, int, bool]:
    try:
        import fitz
        doc        = fitz.open(stream=file_bytes, filetype="pdf")
        pages      = [page.get_text("text") for page in doc]
        page_count = len(pages)
        text       = "\n".join(pages)
        doc.close()
        if len(text.strip().split()) > 20:
            return text, page_count, False
        return _ocr_pdf(file_bytes, page_count), page_count, True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {e}")


def _ocr_pdf(file_bytes: bytes, page_count: int) -> str:
    try:
        from pdf2image import convert_from_bytes
        from cv.pipeline import preprocess_image, extract_text_from_image
        import numpy as np
        images = convert_from_bytes(file_bytes, dpi=200)
        return "\n".join(
            extract_text_from_image(preprocess_image(np.array(img)))
            for img in images[:20]
        )
    except Exception as e:
        return f"[OCR failed: {e}]"


def _extract_image(file_bytes: bytes) -> tuple[str, int, bool]:
    try:
        from cv.pipeline import preprocess_image, extract_text_from_image
        from PIL import Image
        import numpy as np
        img  = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        text = extract_text_from_image(preprocess_image(np.array(img)))
        return text, 1, True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image OCR failed: {e}")


# ── Analyze endpoint ──────────────────────────────────────────────────────────

@router.post("/analyze", response_model=DocumentAnalysisResponse)
async def analyze_document(
    file:           UploadFile = File(...),
    run_rag:        bool       = Query(default=False,
                                       description="Also run RAG pipeline for legal explanation "
                                                   "(uses Groq API — slower)"),
):
    """
    Full document analysis pipeline with proactive rights alerting.

    Steps:
      1. Extract text from uploaded file
      2. Classify document type (XGBoost, 15 categories)
      3. Extract entities (spaCy + custom regex NER)
      4. Score legal risk per clause
      5. Detect rights violations (rule-based, zero LLM cost) — NEW
      6. (Optional) RAG legal explanation — set run_rag=true

    Supported: PDF, JPEG, PNG, TIFF, BMP, TXT (max 10 MB)

    curl -s -X POST http://localhost:8000/api/v1/document/analyze \\
      -F "file=@employment_contract.pdf" | python -m json.tool
    """
    # ── Read file ─────────────────────────────────────────────────────────────
    file_bytes = await file.read()
    size_mb    = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=413,
                            detail=f"File too large ({size_mb:.1f} MB). Max: {MAX_FILE_SIZE_MB} MB")

    filename = file.filename or "uploaded_file"
    suffix   = Path(filename).suffix.lower()
    warning  = None

    # ── Step 1: Extract text ──────────────────────────────────────────────────
    if suffix == ".pdf" or (file.content_type or "").startswith("application/pdf"):
        text, page_count, ocr_used = _extract_pdf(file_bytes)
    elif suffix == ".txt" or (file.content_type or "").startswith("text/"):
        text       = file_bytes.decode("utf-8", errors="replace")
        page_count = 1
        ocr_used   = False
    elif suffix in (".jpg", ".jpeg", ".png", ".tiff", ".bmp"):
        text, page_count, ocr_used = _extract_image(file_bytes)
    else:
        raise HTTPException(status_code=415,
                            detail=f"Unsupported type: {suffix}. Use PDF, image, or TXT.")

    if not text or len(text.strip()) < 10:
        warning = "Very little text extracted. Document may be blank or image-only."
        text    = text or ""

    # ── Step 2: Classification ────────────────────────────────────────────────
    from models.classifier import classifier
    clf_result = classifier.predict(text)
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
    from models.risk_scorer import risk_scorer
    doc_risk   = risk_scorer.score_document(text, doc_type=doc_type)
    risk_model = RiskModel(
        overall_score   = doc_risk.overall_score,
        risk_level      = doc_risk.risk_level,
        high_risk_count = doc_risk.high_risk_count,
        summary         = doc_risk.summary,
        clause_risks    = [
            ClauseRiskModel(**cr.to_dict())
            for cr in doc_risk.clause_risks
            if cr.score > 0   # only include clauses with some risk
        ],
    )

    # ── Step 5: Proactive rights alerts (rule-based, zero cost) ───────────────
    from models.risk_scorer import detect_rights_violations
    raw_alerts    = detect_rights_violations(doc_type, ent_dict, text)
    rights_alerts = [RightsAlertModel(**a) for a in raw_alerts]

    # ── Step 6: RAG (optional — costs Groq API call) ──────────────────────────
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

            rag_result = rag_pipeline.query(rag_query)
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
        text              = text[:5000],
        word_count        = len(text.split()),
        ocr_used          = ocr_used,
        page_count        = page_count,
        classification    = classification,
        entities          = entities,
        risk              = risk_model,
        rights_alerts     = rights_alerts,
        legal_explanation = legal_explanation,
        warning           = warning,
    )


# ── Document Q&A endpoint ─────────────────────────────────────────────────────

@router.post("/query", response_model=DocQueryResponse)
def document_query(req: DocQueryRequest):
    """
    Ask a question about a previously analyzed document.

    The caller provides:
      - doc_text   : the OCR-extracted text from the document (from /analyze response)
      - question   : e.g. "Is this eviction notice valid?" / "What does clause 3 mean?"
      - session_id : existing session to add the turn to
      - language   : optional — pass "ml" for Malayalam, "hi" for Hindi, etc.

    The endpoint:
      1. Builds a Groq prompt with doc_text injected as context (first 3000 chars)
      2. Runs through the multilingual pipeline if language != "en"
      3. Stores user question + assistant answer in session memory
      4. Returns: answer, applicable_sections, risk_note, session_id

    curl -s -X POST http://localhost:8000/api/v1/document/query \\
      -H "Content-Type: application/json" \\
      -d '{"doc_text":"THIS RENTAL AGREEMENT...","question":"Is this notice valid?","session_id":"<sid>"}' | python -m json.tool
    """
    if not req.doc_text.strip():
        raise HTTPException(status_code=400, detail="doc_text must not be empty")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    # ── Multilingual: translate question to English if needed ─────────────────
    question_en = req.question.strip()
    detected_lang = req.language or "en"

    if detected_lang not in ("en", "english", None, ""):
        try:
            from agents.multilingual_agent import multilingual_agent
            question_en = multilingual_agent.translate_to_english(req.question.strip())
        except Exception:
            pass  # fall through with original question

    # ── Build Groq prompt ─────────────────────────────────────────────────────
    doc_excerpt = req.doc_text[:3000]
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

    answer_text        = ""
    applicable_sections = []
    risk_note          = ""

    try:
        from rag.llm import llm
        raw_response = llm.generate(prompt, max_tokens=600, temperature=0.2)

        # Parse structured response
        lines = raw_response.strip().split("\n")
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
            raw_secs = " ".join(sections_lines)
            applicable_sections = [s.strip() for s in raw_secs.split(",") if s.strip()]
        risk_note = " ".join(risk_lines).strip()

    except Exception as e:
        answer_text = f"Unable to process query: {e}"

    # ── Translate answer back to source language if needed ────────────────────
    if detected_lang not in ("en", "english", None, "") and answer_text:
        try:
            from agents.multilingual_agent import multilingual_agent
            answer_text = multilingual_agent.translate_to_source(answer_text, detected_lang)
        except Exception:
            pass

    # ── Store in session memory ────────────────────────────────────────────────
    try:
        from agents.memory import session_memory
        session_memory.ensure_session(req.session_id)
        session_memory.add_turn(req.session_id, "user",      req.question.strip(), intent="document_query")
        session_memory.add_turn(req.session_id, "assistant", answer_text,           intent="document_query")
    except Exception:
        pass  # non-fatal — don't fail the request over memory error

    return DocQueryResponse(
        answer              = answer_text,
        applicable_sections = applicable_sections,
        risk_note           = risk_note,
        session_id          = req.session_id,
    )