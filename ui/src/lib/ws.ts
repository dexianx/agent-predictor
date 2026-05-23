import type { WsEvent } from "./types";

export type SimulateRequest = {
  user_id: string;
  item_name: string;
  price_ngn: number;
  item_id?: string;
  item_description?: string;
  item_brand?: string;
  item_category?: string;
  item_features?: string[];
  item_details?: string;
  raw_message?: string;
};

export type PriorTurn = {
  query: string;
  shown_item_ids: string[];
};

export type RecommendWsRequest = {
  user_id: string;
  context_query: string;
  limit?: number;
  prior_turns?: PriorTurn[];
};

const WS_BASE =
  (import.meta.env.VITE_API_URL ?? "http://localhost:8000").replace(/^http/, "ws");

function _makeWsGenerator(url: string, req: object): AsyncGenerator<WsEvent> {
  const ws = new WebSocket(url);
  const queue: WsEvent[] = [];
  let resolve: (() => void) | null = null;
  let done = false;
  let error: Error | null = null;

  ws.onopen  = () => { ws.send(JSON.stringify(req)); };
  ws.onmessage = (ev) => {
    try {
      queue.push(JSON.parse(ev.data) as WsEvent);
      resolve?.();
      resolve = null;
    } catch { /* ignore malformed frames */ }
  };
  ws.onerror = () => { error = new Error("WebSocket connection error"); resolve?.(); resolve = null; };
  ws.onclose = () => { done = true; resolve?.(); resolve = null; };

  return (async function* () {
    while (true) {
      if (queue.length > 0) {
        const event = queue.shift()!;
        yield event;
        if (event.type === "final" || event.type === "recommend_final" || event.type === "error") {
          ws.close();
          return;
        }
      } else if (done) {
        return;
      } else if (error) {
        throw error;
      } else {
        await new Promise<void>((r) => { resolve = r; });
      }
    }
  })();
}

export function connectSimulate(req: SimulateRequest): AsyncGenerator<WsEvent> {
  return _makeWsGenerator(`${WS_BASE}/ws/simulate`, req);
}

export function connectRecommend(req: RecommendWsRequest): AsyncGenerator<WsEvent> {
  return _makeWsGenerator(`${WS_BASE}/ws/recommend`, req);
}
