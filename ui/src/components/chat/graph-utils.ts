import type { GraphNode } from "@/lib/types";

// Warm theme palette keyed by node type. Hex values mirror the brand tokens
// in styles.css so the canvas blends with the rest of the chat surface.
export const NODE_COLOR: Record<GraphNode["type"], string> = {
  user: "#d97757",
  review: "#c98a6a",
  item: "#a8b8c8",
  category: "#dedcd1",
};

export function truncate(s: string, n = 22) {
  return s.length > n ? `${s.slice(0, n - 1)}…` : s;
}

// Review nodes from the backend label themselves as "5.0★" / "0.0★" (the
// rating). On the canvas we'd rather surface the review-text snippet, the
// same way Neo4j Browser shows the title; the rating stays available in the
// side panel. All other node types use the backend label directly.
export function captionFor(n: GraphNode): string {
  if (n.type === "review" && n.text && n.text.trim().length > 0) {
    return truncate(n.text, 18);
  }
  return truncate(n.label, 18);
}
