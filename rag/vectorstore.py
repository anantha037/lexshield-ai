"""
LexShield AI — Vector Store
====================================================
Changes in this version:
  - era field added to ChromaDB metadata in ingest_chunks()
    "legacy" | "current" | "" — used for paired act retrieval in pipeline
  - SOURCE_KEYWORDS and all other logic unchanged from previous version
"""

import os
import time
import gc
from typing import Optional

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import chromadb
from chromadb.config import Settings
from rag.embedder import embedder

COLLECTION_NAME = "legal_documents"
INGEST_BATCH    = 16
BATCH_SLEEP     = 1.5
GC_EVERY_N      = 5

SOURCE_KEYWORDS: dict[str, str] = {
    # Criminal & Justice
    "ipc":                              "Indian Penal Code",
    "indian penal code":               "Indian Penal Code",
    "bns":                              "Bharatiya Nyaya Sanhita",
    "bharatiya nyaya sanhita":         "Bharatiya Nyaya Sanhita",
    "bnss":                             "Bharatiya Nagarik Suraksha Sanhita",
    "bharatiya nagarik suraksha":       "Bharatiya Nagarik Suraksha Sanhita",
    "bsa":                              "Bharatiya Sakshya Adhiniyam",
    "bharatiya sakshya":               "Bharatiya Sakshya Adhiniyam",
    "crpc":                             "Code of Criminal Procedure",
    "code of criminal procedure":       "Code of Criminal Procedure",
    "evidence act":                     "Indian Evidence Act",
    "indian evidence":                  "Indian Evidence Act",
    "pocso":                            "Protection of Children from Sexual Offences",
    "sexual offences":                  "Protection of Children from Sexual Offences",
    "pmla":                             "Prevention of Money Laundering",
    "money laundering":                 "Prevention of Money Laundering",
    "ndps":                             "Narcotic Drugs",
    "narcotic drugs":                   "Narcotic Drugs",
    "narcotic":                         "Narcotic Drugs",
    "uapa":                             "Unlawful Activities",
    "unlawful activities":              "Unlawful Activities",
    "juvenile justice":                 "Juvenile Justice",
    "prevention of corruption":         "Prevention of Corruption",
    # Family & Personal Laws
    "hindu marriage":                   "Hindu Marriage Act",
    "special marriage":                 "Special Marriage Act",
    "muslim personal law":              "Muslim Personal Law",
    "shariat":                          "Muslim Personal Law",
    "indian succession":                "Indian Succession Act",
    "succession act":                   "Indian Succession Act",
    "hindu succession":                 "Hindu Succession Act",
    "domestic violence":                "Protection of Women from Domestic Violence",
    "dv act":                           "Protection of Women from Domestic Violence",
    "family courts":                    "Family Courts Act",
    "senior citizens":                  "Maintenance and Welfare of Parents",
    "maintenance":                      "Maintenance and Welfare of Parents",
    "parents maintenance":              "Maintenance and Welfare of Parents",
    # Corporate, Business & Contract
    "companies act":                    "Companies Act",
    "company law":                      "Companies Act",
    "indian partnership":               "Indian Partnership Act",
    "partnership act":                  "Indian Partnership Act",
    "llp":                              "Limited Liability Partnership",
    "limited liability partnership":    "Limited Liability Partnership",
    "msmed":                            "Micro, Small and Medium Enterprises",
    "msme":                             "Micro, Small and Medium Enterprises",
    "ibc":                              "Insolvency and Bankruptcy Code",
    "insolvency":                       "Insolvency and Bankruptcy Code",
    "bankruptcy":                       "Insolvency and Bankruptcy Code",
    "competition act":                  "Competition Act",
    "negotiable instruments":           "Negotiable Instruments Act",
    "ni act":                           "Negotiable Instruments Act",
    "cheque bounce":                    "Negotiable Instruments Act",
    "dishonour of cheque":              "Negotiable Instruments Act",
    "section 138":                      "Negotiable Instruments Act",
    "contract act":                     "Indian Contract Act",
    "indian contract":                  "Indian Contract Act",
    "arbitration":                      "Arbitration and Conciliation Act",
    "conciliation":                     "Arbitration and Conciliation Act",
    "specific relief":                  "Specific Relief Act",
    "specific performance":             "Specific Relief Act",
    # Taxation & Finance
    "income tax":                       "Income Tax Act",
    "cgst":                             "Central Goods and Services Tax",
    "gst":                              "Central Goods and Services Tax",
    "igst":                             "Integrated Goods and Services Tax",
    "integrated gst":                   "Integrated Goods and Services Tax",
    "customs act":                      "Customs Act",
    "customs duty":                     "Customs Act",
    "fema":                             "Foreign Exchange Management",
    "foreign exchange":                 "Foreign Exchange Management",
    "sebi":                             "Securities and Exchange Board of India",
    "securities exchange":              "Securities and Exchange Board of India",
    "banking regulation":               "Banking Regulation Act",
    # Property, Real Estate & Tenancy
    "transfer of property":             "Transfer of Property Act",
    "top act":                          "Transfer of Property Act",
    "registration act":                 "Registration Act",
    "rera":                             "Real Estate",
    "real estate":                      "Real Estate",
    "kerala":                           "Kerala Buildings",
    "kerala rent":                      "Kerala Buildings",
    "lease and rent":                   "Kerala Buildings",
    # Labour & Employment
    "wages":                            "Code on Wages",
    "code on wages":                    "Code on Wages",
    "industrial relations":             "Industrial Relations Code",
    "social security code":             "Code on Social Security",
    "epf":                              "Code on Social Security",
    "gratuity":                         "Code on Social Security",
    "maternity":                        "Code on Social Security",
    "osh code":                         "Occupational Safety",
    "occupational safety":              "Occupational Safety",
    "posh":                             "Sexual Harassment of Women at Workplace",
    "posh act":                         "Sexual Harassment of Women at Workplace",
    "sexual harassment":                "Sexual Harassment of Women at Workplace",
    "workplace harassment":             "Sexual Harassment of Women at Workplace",
    # Food, Health & Education
    "fssai":                            "Food Safety and Standards",
    "food safety":                      "Food Safety and Standards",
    "food adulteration":                "Food Safety and Standards",
    "rte":                              "Right of Children to Free and Compulsory Education",
    "right to education":               "Right of Children to Free and Compulsory Education",
    "drugs and cosmetics":              "Drugs and Cosmetics Act",
    "pharmacy":                         "Drugs and Cosmetics Act",
    "clinical establishments":          "Clinical Establishments",
    "hospital registration":            "Clinical Establishments",
    # Environment, Forest & Agriculture
    "environment protection":           "Environment (Protection) Act",
    "epa":                              "Environment (Protection) Act",
    "forest conservation":              "Forest (Conservation) Act",
    "indian forest":                    "Indian Forest Act",
    "forest act":                       "Indian Forest Act",
    "wildlife":                         "Wildlife (Protection) Act",
    "wildlife protection":              "Wildlife (Protection) Act",
    "water pollution":                  "Water (Prevention and Control of Pollution)",
    "air pollution":                    "Air (Prevention and Control of Pollution)",
    # Technology, Media & IP
    "it act":                           "Information Technology",
    "information technology":           "Information Technology",
    "cyber":                            "Information Technology",
    "cybercrime":                       "Information Technology",
    "dpdp":                             "Digital Personal Data Protection",
    "data protection":                  "Digital Personal Data Protection",
    "personal data":                    "Digital Personal Data Protection",
    "data breach":                      "Digital Personal Data Protection",
    "copyright":                        "Copyright Act",
    "patents":                          "Patents Act",
    "intellectual property":            "Patents Act",
    "trademarks":                       "Trade Marks Act",
    "trademark":                        "Trade Marks Act",
    # Citizen Rights & Daily Administration
    "constitution":                     "Constitution of India",
    "fundamental rights":               "Constitution of India",
    "article 21":                       "Constitution of India",
    "directive principles":             "Constitution of India",
    "motor vehicles":                   "Motor Vehicles Act",
    "motor vehicle":                    "Motor Vehicles Act",
    "mv act":                           "Motor Vehicles Act",
    "traffic violation":                "Motor Vehicles Act",
    "road accident":                    "Motor Vehicles Act",
    "drunk driving":                    "Motor Vehicles Act",
    "drunk and drive":                  "Motor Vehicles Act",
    "rti":                              "Right to Information",
    "right to information":             "Right to Information",
    "consumer":                         "Consumer Protection",
    "consumer protection":              "Consumer Protection",
    "aadhaar":                          "Aadhaar",
    "cpc":                              "Code of Civil Procedure",
    "civil procedure":                  "Code of Civil Procedure",
    "code of civil procedure":          "Code of Civil Procedure",
}


class LegalVectorStore:

    def __init__(self, persist_dir: str = "data/chroma_db"):
        self.persist_dir = persist_dir
        mode = os.environ.get("CHROMA_MODE", "local").strip().lower()

        if mode == "cloud":
            api_key  = os.environ["CHROMA_API_KEY"]
            tenant   = os.environ["CHROMA_TENANT"]
            database = os.environ["CHROMA_DATABASE"]
            self.client = chromadb.HttpClient(
                ssl=True,
                headers={"x-chroma-token": api_key},
                tenant=tenant,
                database=database,
                settings=Settings(anonymized_telemetry=False),
            )
            print(f"[VectorStore] Mode=cloud  tenant={tenant!r}  db={database!r}")
        else:
            self.client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            print(f"[VectorStore] Mode=local  persist_dir={persist_dir!r}")

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[VectorStore] Collection '{COLLECTION_NAME}' — {self.count()} docs")

    # ── Direct metadata lookup — section fast path ────────────────────────────

    def get_by_section(
        self,
        section_number: str,
        source_hint:    Optional[str] = None,
    ) -> list[dict]:
        """
        Direct ChromaDB metadata query by section number.
        limit= NOT passed to .get() — sliced in Python for ChromaDB compat.
        """
        section_number = section_number.strip().upper()
        try:
            raw = self.collection.get(
                where={"section": {"$eq": section_number}},
                include=["documents", "metadatas"],
            )
            if not raw or not raw.get("ids"):
                return []

            results: list[dict] = []
            for cid, doc, meta in zip(
                raw.get("ids", []), raw.get("documents", []), raw.get("metadatas", [])
            ):
                source = meta.get("source", "")
                if source_hint and source_hint.lower() not in source.lower():
                    continue
                results.append({
                    "chunk_id":         cid,
                    "text":             doc,
                    "source":           source,
                    "doc_type":         meta.get("doc_type",      ""),
                    "section":          meta.get("section",       ""),
                    "section_title":    meta.get("section_title", ""),
                    "chapter":          meta.get("chapter",       ""),
                    "chunk_type":       meta.get("chunk_type",    ""),
                    "category":         meta.get("category",      ""),
                    "era":              meta.get("era",           ""),
                    "score":            1.0,
                    "vector_score":     1.0,
                    "bm25_score":       1.0,
                    "bm25_score_norm":  1.0,
                    "hybrid_score":     1.0,
                    "retrieval_source": "metadata",
                    "rerank_score":     None,
                })
            return results[:20]
        except Exception as e:
            print(f"[VectorStore] get_by_section({section_number!r}) error: {e}")
            return []

    # ── Ingest ────────────────────────────────────────────────────────────────

    def ingest_chunks(self, chunks: list[dict], skip_existing: bool = True) -> int:
        if not chunks:
            return 0
        if skip_existing:
            try:
                existing_ids = set(self.collection.get(include=[])["ids"])
            except Exception:
                existing_ids = set()
            new_chunks = [c for c in chunks if c.get("chunk_id", "") not in existing_ids]
        else:
            new_chunks = chunks

        if not new_chunks:
            print("[VectorStore] All chunks already present, skipping.")
            return 0

        total   = len(new_chunks)
        added   = 0
        batches = [new_chunks[i : i + INGEST_BATCH] for i in range(0, total, INGEST_BATCH)]
        print(f"[VectorStore] Ingesting {total} chunks in {len(batches)} batches ...")

        for batch_idx, batch in enumerate(batches):
            docs      = [c.get("context_text") or c.get("text", "") for c in batch]
            ids       = [c["chunk_id"] for c in batch]
            metadatas = [
                {
                    "source":        str(c.get("source",        "")),
                    "doc_type":      str(c.get("doc_type",      "")),
                    "section":       str(c.get("section",       "")),
                    "section_title": str(c.get("section_title", "")),
                    "chapter":       str(c.get("chapter",       "")),
                    "chunk_type":    str(c.get("chunk_type",    "")),
                    "word_count":    int(c.get("word_count",    0)),
                    "category":      str(c.get("category",      "")),
                    "era":           str(c.get("era",           "")),  # NEW
                }
                for c in batch
            ]
            embeddings = embedder.embed(docs)
            self.collection.add(
                ids=ids, documents=docs,
                embeddings=embeddings, metadatas=metadatas,
            )
            added += len(batch)
            if (batch_idx + 1) % GC_EVERY_N == 0:
                gc.collect()
            time.sleep(BATCH_SLEEP)
            if (batch_idx + 1) % 10 == 0 or batch_idx == len(batches) - 1:
                print(f"  batch {batch_idx + 1}/{len(batches)}  ({added}/{total})")

        gc.collect()
        print(f"[VectorStore] Done. Added {added}. Total: {self.count()}")
        return added

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset_collection(self):
        print(f"[VectorStore] Deleting '{COLLECTION_NAME}' ...")
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print("[VectorStore] Collection reset.")

    # ── Vector search ─────────────────────────────────────────────────────────

    def search(
        self,
        query:           str,
        n_results:       int           = 8,
        category_filter: Optional[str] = None,
    ) -> list[dict]:
        if self.count() == 0:
            return []

        query_embedding = embedder.embed([query])[0]
        where = {"category": {"$eq": category_filter}} if category_filter else None

        kwargs: dict = dict(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self.count()),
            include=["documents", "distances", "metadatas"],
        )
        if where:
            kwargs["where"] = where

        raw = self.collection.query(**kwargs)

        results: list[dict] = []
        for cid, doc, dist, meta in zip(
            raw["ids"][0], raw["documents"][0],
            raw["distances"][0], raw["metadatas"][0],
        ):
            score = max(0.0, 1.0 - dist / 2.0)
            results.append({
                "chunk_id":      cid,
                "text":          doc,
                "source":        meta.get("source",        ""),
                "doc_type":      meta.get("doc_type",      ""),
                "section":       meta.get("section",       ""),
                "section_title": meta.get("section_title", ""),
                "chapter":       meta.get("chapter",       ""),
                "chunk_type":    meta.get("chunk_type",    ""),
                "category":      meta.get("category",      ""),
                "era":           meta.get("era",           ""),
                "score":         round(score, 4),
            })
        return results

    # ── Source-filtered semantic search (for paired act retrieval) ────────────

    def search_by_source(
        self,
        query:          str,
        source_partial: str,
        n_results:      int = 3,
    ) -> list[dict]:
        """
        Semantic search returning only chunks whose source contains source_partial.
        Used by pipeline to retrieve counterpart act chunks (e.g. BNS when IPC queried).
        source_partial is matched case-insensitively against chunk source strings.
        ChromaDB doesn't support partial string match in where clause, so we fetch
        more results and filter in Python.
        """
        raw = self.search(query, n_results=n_results * 8)
        filtered = [
            r for r in raw
            if source_partial.lower() in r.get("source", "").lower()
        ]
        return filtered[:n_results]

    # ── Utilities ─────────────────────────────────────────────────────────────

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            return 0

    def get_by_id(self, chunk_id: str) -> Optional[dict]:
        try:
            r = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
            if not r["ids"]:
                return None
            return {"chunk_id": chunk_id, "text": r["documents"][0], **r["metadatas"][0]}
        except Exception:
            return None


# ── Singleton ─────────────────────────────────────────────────────────────────
vectorstore = LegalVectorStore()