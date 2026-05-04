"""
LexShield AI — Document Analysis API  (Week 2, Day 5 update)
=============================================================
POST /api/v1/document/analyze

Full pipeline on every document upload:
  1. Extract text  (PyMuPDF for digital PDFs, OCR for scanned/images)
  2. Classify      (XGBoost → document type + confidence)
  3. NER           (spaCy + regex → structured entities)
  4. Risk Score    (clause-level risk with legal references)
  5. RAG Q&A       (optional: pass extracted text to RAG pipeline for legal explanation)

Response JSON structure:
  {
    "filename":          "rental_agreement.pdf",
    "text":              "THIS RENTAL AGREEMENT...",
    "word_count":        342,
    "ocr_used":          false,
    "page_count":        2,

    "classification": {
      "label":           0,
      "label_name":      "rental_agreement",
      "confidence":      0.97,
      "all_scores":      {...}
    },

    "entities": {
      "persons":         ["Rajesh Kumar", "Priya Sharma"],
      "organizations":   [...],
      "dates":           [...],
      "locations":       [...],
      "monetary":        ["₹15,000", "₹90,000"],
      "ipc_sections":    [],
      "case_numbers":    [],
      "acts":            ["Kerala Buildings (Lease and Rent Control) Act"],
      "entity_counts":   {...}
    },

    "risk": {
      "overall_score":   72,
      "risk_level":      "high",
      "high_risk_count": 2,
      "summary":         "HIGH RISK: 2 clauses contain high-risk terms...",
      "clause_risks": [
        {
          "clause_number":  3,
          "clause_text":    "The deposit shall be non-refundable...",
          "score":          80,
          "risk_level":     "critical",
          "flags":          ["NON_REFUNDABLE_DEPOSIT"],
          "legal_refs":     ["Section 10 of the Kerala Buildings..."],
          "explanation":    "Non-refundable deposit clauses are void..."
        }
      ]
    },

    "legal_explanation": {
      "answer":    "Based on the retrieved sections...",
      "citations": [...],
      "sources_consulted": 3
    },

    "warning": null
  }
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


class DocumentAnalysisResponse(BaseModel):
    filename:           str
    text:               str
    word_count:         int
    ocr_used:           bool
    page_count:         int
    classification:     ClassificationResult
    entities:           EntitiesModel
    risk:               RiskModel
    legal_explanation:  Optional[LegalExplanationModel] = None
    warning:            Optional[str]                   = None


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


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/analyze", response_model=DocumentAnalysisResponse)
async def analyze_document(
    file:           UploadFile = File(...),
    run_rag:        bool       = Query(default=False,
                                       description="Also run RAG pipeline for legal explanation "
                                                   "(uses Groq API — slower)"),
):
    """
    Full document analysis pipeline.

    Steps:
      1. Extract text from uploaded file
      2. Classify document type (XGBoost, 8 categories)
      3. Extract entities (spaCy + custom regex NER)
      4. Score legal risk per clause
      5. (Optional) RAG legal explanation — set run_rag=true

    No Groq API calls by default. Set run_rag=true only when needed.
    Supported: PDF, JPEG, PNG, TIFF, BMP, TXT (max 10 MB)
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
    doc_risk   = risk_scorer.score(text, doc_type=doc_type)
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

    # ── Step 5: RAG (optional — costs Groq API call) ──────────────────────────
    legal_explanation = None
    if run_rag and text.strip():
        try:
            from rag.pipeline import rag_pipeline
            # Build a focused query from doc type + top entities
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
        legal_explanation = legal_explanation,
        warning           = warning,
    )