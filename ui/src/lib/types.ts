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

// ── Task B: Recommendation types ──────────────────────────────────────────────

export type RecommendItem = {
  item_id: string;
  title: string;
  category: string | null;
  price_usd: number | null;
  price_ngn: number | null;
  avg_rating: number | null;
  rating_count: number | null;
  match_score: number;
  reasons: string[];
};

export type RecommendFinalEvent = {
  type: "recommend_final";
  user_id: string;
  context_query: string;
  items: RecommendItem[];
  locale: string | null;
};

export type ToolCallEvent = {
  type: "tool_call";
  call_id: string;
  tool: string;
  phase: "start" | "done";
  input?: Record<string, unknown>;
  output?: Record<string, unknown>;
};

export type WsEvent = StepEvent | TokenEvent | GraphDataEvent | FinalEvent | RecommendFinalEvent | ErrorEvent | ToolCallEvent;

// ── Chat thread types ─────────────────────────────────────────────────────────
export type ToolCallStep = {
  kind: "tool_call";
  id: string;
  tool: string;
  label: string;
  input: Record<string, unknown>;
  output?: Record<string, unknown>;
  status: "running" | "done";
};

export type RegularStep = {
  kind?: "step";
  id: string;
  label: string;
  detail?: string;
};

export type AnyStep = RegularStep | ToolCallStep;

export type AssistantPayload =
  | {
      type: "review";
      review_text: string;
      predicted_rating: number;
      user_id: string;
      item_id: string;
      locale: string | null;
      retrieval_branch: string;
    }
  | {
      type: "recommend";
      user_id: string;
      context_query: string;
      items: RecommendItem[];
      locale: string | null;
    };

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  createdAt: number;
  payload?: AssistantPayload;
  steps?: AnyStep[];
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
