// ── Persona (real SQLite user) ────────────────────────────────────────────────
export type CategoryBias = {
  category: string;
  mean: number;
  count: number;
};

export type PersonaDetail = {
  user_id: string;
  locale: string | null;
  platform: string | null;
  review_count: number | null;
  rating_mean: number | null;
  rating_std: number | null;
  rating_skew: number | null;
  top_categories: string[];
  avg_sent_len: number | null;
  vocab_richness: number | null;
  exclamation_rate: number | null;
  caps_rate: number | null;
  pidgin_freq: number | null;
  emoji_rate: number | null;
  question_rate: number | null;
  naira_mention_rate: number | null;
  price_sensitivity: number | null;
  authenticity_concern: number | null;
  cat_bias: CategoryBias[];
};

// ── Knowledge graph (React Flow) ──────────────────────────────────────────────
export type GraphNode = {
  id: string;
  type: "user" | "review" | "item" | "category";
  label: string;
  text?: string;
  age?: number;
  score?: number;
  isTarget?: boolean;
};

export type GraphEdge = {
  source: string;
  target: string;
  label: string;
  score?: number;
};

export type GraphData = {
  branch: "ppr" | "vector" | "query";
  nodes: GraphNode[];
  edges: GraphEdge[];
};

// ── WebSocket event types ─────────────────────────────────────────────────────
export type StepEvent = {
  type: "step";
  id: string;
  label: string;
  detail?: string;
};

export type TokenEvent = {
  type: "token";
  text: string;
};

export type GraphDataEvent = {
  type: "graph_data";
  branch: "ppr" | "vector" | "query";
  nodes: GraphNode[];
  edges: GraphEdge[];
};

export type FinalEvent = {
  type: "final";
  user_id: string;
  item_id: string;
  review_text: string;
  predicted_rating: number;
  locale: string | null;
  retrieval_branch: string;
};

export type ErrorEvent = {
  type: "error";
  message: string;
};

export type WsEvent = StepEvent | TokenEvent | GraphDataEvent | FinalEvent | ErrorEvent;

// ── Chat thread types ─────────────────────────────────────────────────────────
export type AssistantPayload = {
  type: "review";
  review_text: string;
  predicted_rating: number;
  user_id: string;
  item_id: string;
  locale: string | null;
  retrieval_branch: string;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  createdAt: number;
  payload?: AssistantPayload;
  steps?: { id: string; label: string; detail?: string }[];
  graph?: GraphData;
};

export type Thread = {
  id: string;
  title: string;
  personaId: string;
  updatedAt: number;
  createdAt: number;
  messages: ChatMessage[];
};

export const STORAGE_KEY = "kora.threads.v1";
