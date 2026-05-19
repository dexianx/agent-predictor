import {
  Suspense,
  lazy,
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";
import { AnimatePresence, motion } from "framer-motion";
import { X } from "lucide-react";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { GraphData, GraphNode, GraphEdge } from "@/lib/types";
import { NODE_COLOR, captionFor, truncate } from "./graph-utils";

const EASE = [0.22, 1, 0.36, 1] as const;

// NVL accesses `document` at module-eval, so it must never be imported during
// SSR. We pay for that with a one-time lazy() round-trip on first render.
const GraphCanvas = lazy(() => import("./GraphCanvas"));

type Props = {
  data: GraphData;
};

export function KnowledgeGraph({ data }: Props) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);

  // Defer NVL mount until after hydration so the document-touching module
  // graph is only ever loaded in the browser.
  useEffect(() => {
    setMounted(true);
  }, []);

  const nodesById = useMemo(() => {
    const m = new Map<string, GraphNode>();
    data.nodes.forEach((n) => m.set(n.id, n));
    return m;
  }, [data.nodes]);

  const selectedNode = selectedId ? (nodesById.get(selectedId) ?? null) : null;

  const handleNodeClick = useCallback((id: string) => setSelectedId(id), []);
  const handleCanvasClick = useCallback(() => setSelectedId(null), []);

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: EASE }}
      className="rounded-2xl border border-border/70 bg-card overflow-hidden shadow-[0_10px_30px_-22px_rgba(217,119,87,0.35)]"
    >
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-border/60">
        <div className="flex items-baseline gap-3 min-w-0">
          <span className="font-serif text-[13.5px] text-foreground">Knowledge Graph</span>
          <span className="text-[11.5px] text-muted-foreground tabular-nums">
            {data.nodes.length} nodes · {data.edges.length} edges
          </span>
        </div>
        <Badge variant="secondary" className="rounded-md text-[10.5px] font-medium">
          LLM Cypher
        </Badge>
      </div>

      <Tabs defaultValue="graph" className="w-full">
        <div className="px-4 pt-2.5">
          <TabsList className="h-7">
            <TabsTrigger value="graph" className="text-[11.5px] h-6 px-2.5">Graph</TabsTrigger>
            <TabsTrigger value="table" className="text-[11.5px] h-6 px-2.5">Table</TabsTrigger>
            <TabsTrigger value="raw" className="text-[11.5px] h-6 px-2.5">RAW</TabsTrigger>
          </TabsList>
        </div>

        <TabsContent value="graph" className="m-0">
          <div
            className="relative h-[360px] w-full overflow-hidden"
            style={{ background: "var(--color-vellum-white, #faf9f5)" }}
          >
            <div className="absolute inset-0">
              <div className="relative w-full h-full">
                {data.nodes.length === 0 ? (
                  <div className="absolute inset-0 grid place-items-center text-xs text-muted-foreground">
                    Querying Neo4j…
                  </div>
                ) : mounted ? (
                  <Suspense
                    fallback={
                      <div className="absolute inset-0 grid place-items-center text-xs text-muted-foreground">
                        Loading visualisation…
                      </div>
                    }
                  >
                    <GraphCanvas
                      nodes={data.nodes}
                      edges={data.edges}
                      onNodeClick={handleNodeClick}
                      onCanvasClick={handleCanvasClick}
                    />
                  </Suspense>
                ) : (
                  <div className="absolute inset-0 grid place-items-center text-xs text-muted-foreground">
                    Preparing graph…
                  </div>
                )}
              </div>

              <AnimatePresence initial={false}>
                {selectedNode && (
                  <motion.aside
                    key="details"
                    initial={{ x: 16, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    exit={{ x: 16, opacity: 0 }}
                    transition={{ duration: 0.2, ease: EASE }}
                    className="absolute right-0 top-0 bottom-0 w-[260px] border-l border-border/60 bg-background/95 backdrop-blur-sm shadow-[-8px_0_24px_-18px_rgba(0,0,0,0.4)] overflow-hidden z-10"
                  >
                    <NodeDetailsPanel
                      node={selectedNode}
                      onClose={() => setSelectedId(null)}
                    />
                  </motion.aside>
                )}
              </AnimatePresence>
            </div>
          </div>
        </TabsContent>

        <TabsContent value="table" className="m-0">
          <div className="max-h-[360px] overflow-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-12 text-[11px]">#</TableHead>
                  <TableHead className="text-[11px]">p</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {data.edges.map((e, i) => (
                  <EdgeRow
                    key={`${e.source}->${e.target}:${e.label}:${i}`}
                    idx={i + 1}
                    edge={e}
                    nodes={data.nodes}
                  />
                ))}
                {data.edges.length === 0 && (
                  <TableRow>
                    <TableCell colSpan={2} className="text-xs text-muted-foreground py-6 text-center">
                      No relationships yet.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </div>
        </TabsContent>

        <TabsContent value="raw" className="m-0">
          <RawView data={data} />
        </TabsContent>
      </Tabs>
    </motion.div>
  );
}

function NodeDetailsPanel({ node, onClose }: { node: GraphNode; onClose: () => void }) {
  // Show every property the node actually carries — including the full
  // (untruncated) text so clicking a review reveals its content.
  const rows: Array<[string, string]> = [];
  rows.push(["id", node.id]);
  rows.push(["type", node.type]);
  rows.push(["label", node.label]);
  if (node.text) rows.push(["text", node.text]);
  if (typeof node.score === "number") rows.push(["score", node.score.toFixed(3)]);
  if (typeof node.age === "number") rows.push(["age (months)", String(node.age)]);
  if (node.isTarget) rows.push(["target", "true"]);

  return (
    <div className="flex flex-col h-full w-[260px]">
      <div className="flex items-center justify-between px-3 py-2 border-b border-border/60">
        <div className="flex items-center gap-2 min-w-0">
          <span
            className="inline-block size-2.5 rounded-full"
            style={{ backgroundColor: NODE_COLOR[node.type] ?? NODE_COLOR.item }}
          />
          <span className="text-[11.5px] font-medium uppercase tracking-wide text-muted-foreground">
            {node.type}
          </span>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="text-muted-foreground hover:text-foreground"
          aria-label="Close"
        >
          <X className="size-3.5" />
        </button>
      </div>
      <div className="overflow-auto flex-1">
        <table className="w-full text-[11px]">
          <thead>
            <tr className="text-muted-foreground">
              <th className="text-left font-medium px-3 py-1.5 w-[80px]">Key</th>
              <th className="text-left font-medium px-3 py-1.5">Value</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([k, v]) => (
              <tr key={k} className="border-t border-border/40">
                <td className="px-3 py-1.5 align-top text-muted-foreground whitespace-nowrap">
                  {k}
                </td>
                <td className="px-3 py-1.5 align-top break-words font-mono text-[10.5px] text-foreground">
                  {v}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function EdgeRow({ idx, edge, nodes }: { idx: number; edge: GraphEdge; nodes: GraphNode[] }) {
  const src = nodes.find((n) => n.id === edge.source);
  const tgt = nodes.find((n) => n.id === edge.target);
  return (
    <TableRow>
      <TableCell className="text-[11px] text-muted-foreground tabular-nums">{idx}</TableCell>
      <TableCell className="text-[11px]">
        <div className="flex items-center gap-1.5 flex-wrap">
          <NodeChip node={src} fallback={edge.source} />
          <span className="text-muted-foreground">─</span>
          <span className="inline-flex items-center rounded-md border border-border/60 bg-background px-2 py-0.5 text-[10px] font-medium tracking-wide uppercase text-muted-foreground">
            {edge.label}
          </span>
          <span className="text-muted-foreground">→</span>
          <NodeChip node={tgt} fallback={edge.target} />
        </div>
      </TableCell>
    </TableRow>
  );
}

function NodeChip({ node, fallback }: { node: GraphNode | undefined; fallback: string }) {
  const color = node ? NODE_COLOR[node.type] : NODE_COLOR.item;
  const display = node ? captionFor(node).replace(/…$/, "") : truncate(fallback, 22);
  const full = node?.text || node?.label || fallback;
  return (
    <span
      className="inline-flex items-center rounded-md px-2 py-0.5 text-[10.5px] font-medium"
      style={{ backgroundColor: color, color: "#1f1e1d" }}
      title={full}
    >
      {display}
    </span>
  );
}

function RawView({ data }: { data: GraphData }) {
  const branch = data.branch;
  const apiBase = (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8000";
  const summary = {
    nodes: data.nodes.length,
    edges: data.edges.length,
    by_type: Object.fromEntries(
      (["user", "item", "review", "category"] as GraphNode["type"][]).map((t) => [
        t,
        data.nodes.filter((n) => n.type === t).length,
      ]),
    ),
    by_label: Object.fromEntries(
      Array.from(new Set(data.edges.map((e) => e.label))).map((lbl) => [
        lbl,
        data.edges.filter((e) => e.label === lbl).length,
      ]),
    ),
  };

  const rows: Array<{ key: string; value: React.ReactNode }> = [
    { key: "Source", value: "FastAPI · simulate_review" },
    { key: "Endpoint", value: apiBase.replace(/^http/, "ws") + "/ws/simulate" },
    { key: "Retrieval branch", value: branch },
    {
      key: "Query",
      value: (
        <code className="font-mono">
          LLM-generated Cypher (branch={branch}) · MATCH (u:User)-[:WROTE]→(r:Review) …
        </code>
      ),
    },
    {
      key: "Result",
      value: (
        <details>
          <summary className="cursor-pointer text-foreground hover:text-primary">
            {data.nodes.length} nodes · {data.edges.length} relationships
          </summary>
          <pre className="mt-2 max-h-[220px] overflow-auto bg-muted/40 rounded-md p-3 font-mono text-[10.5px] leading-relaxed">
{JSON.stringify({ nodes: data.nodes, edges: data.edges }, null, 2)}
          </pre>
        </details>
      ),
    },
    {
      key: "Summary",
      value: (
        <details>
          <summary className="cursor-pointer text-foreground hover:text-primary">
            counts by type and relationship label
          </summary>
          <pre className="mt-2 max-h-[180px] overflow-auto bg-muted/40 rounded-md p-3 font-mono text-[10.5px] leading-relaxed">
{JSON.stringify(summary, null, 2)}
          </pre>
        </details>
      ),
    },
  ];

  return (
    <div className="max-h-[360px] overflow-auto">
      <table className="w-full text-[11.5px]">
        <tbody>
          {rows.map(({ key, value }) => (
            <tr key={key} className="border-t border-border/40 first:border-t-0 align-top">
              <td className="px-4 py-2.5 text-muted-foreground font-medium whitespace-nowrap w-[150px]">
                {key}
              </td>
              <td className="px-4 py-2.5 text-foreground">{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
