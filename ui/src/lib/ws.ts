import type { WsEvent } from "./types";

export type SimulateRequest = {
  user_id: string;
  item_name: string;
  price_ngn: number;
  item_id?: string;
  item_description?: string;
  raw_message?: string;
};

const WS_BASE =
  (import.meta.env.VITE_API_URL ?? "http://localhost:8000").replace(/^http/, "ws");

export async function* connectSimulate(
  req: SimulateRequest
): AsyncGenerator<WsEvent> {
  const ws = new WebSocket(`${WS_BASE}/ws/simulate`);

  const queue: WsEvent[] = [];
  let resolve: (() => void) | null = null;
  let done = false;
  let error: Error | null = null;

  ws.onopen = () => {
    ws.send(JSON.stringify(req));
  };

  ws.onmessage = (ev) => {
    try {
      const event = JSON.parse(ev.data) as WsEvent;
      queue.push(event);
      resolve?.();
      resolve = null;
    } catch {
      // ignore malformed frames
    }
  };

  ws.onerror = () => {
    error = new Error("WebSocket connection error");
    resolve?.();
    resolve = null;
  };

  ws.onclose = () => {
    done = true;
    resolve?.();
    resolve = null;
  };

  while (true) {
    if (queue.length > 0) {
      const event = queue.shift()!;
      yield event;
      if (event.type === "final" || event.type === "error") {
        ws.close();
        return;
      }
    } else if (done) {
      return;
    } else if (error) {
      throw error;
    } else {
      await new Promise<void>((r) => {
        resolve = r;
      });
    }
  }
}
