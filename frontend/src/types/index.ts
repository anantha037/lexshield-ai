export type IntentType =
  | "legal_query"
  | "risk_check"
  | "document_draft"
  | "ipc_lookup"
  | "case_search"
  | "general";

export interface NEREntity {
  text: string;
  label: "IPC_SECTION" | "ACT" | "DATE" | "PARTY" | "COURT" | "OTHER";
  start: number;
  end: number;
}

export interface Citation {
  chunk_id: string;
  source: string;
  section: string;
  text: string;
  score: number;
}

export interface RiskScore {
  overall: number;
  breakdown: {
    category: string;
    score: number;
  }[];
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  intent?: IntentType;
  citations?: Citation[];
  timestamp: string;
}

export interface DocumentAnalysisResult {
  document_type: string;
  classification: string;
  risk_score?: RiskScore;
  risk?: { score: number; level: string; factors: string[] };
  entities: NEREntity[];
  citations: Citation[];
  summary: string;
  raw_text?: string;
}

export interface ApiError {
  status: number;
  message: string;
  isRateLimit: boolean;
  isTimeout: boolean;
}