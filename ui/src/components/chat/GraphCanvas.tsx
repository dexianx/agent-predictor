import { useCallback, useEffect, useMemo, useRef } from "react";
import { BasicNvlWrapper } from "@neo4j-nvl/react";
import type NVL from "@neo4j-nvl/base";
import type { Node as NvlNode, Relationship as NvlRel } from "@neo4j-nvl/base";
import {
  ClickInteraction,
  DragNodeInteraction,
  HoverInteraction,
  PanInteraction,
  ZoomInteraction,
} from "@neo4j-nvl/interaction-handlers";
import { ZoomIn, ZoomOut, Maximize2 } from "lucide-react";
import type { GraphNode, GraphEdge } from "@/lib/types";
import { NODE_COLOR, captionFor } from "./graph-utils";

type Props = {
  nodes: GraphNode[];
  edges: GraphEdge[];
  onNodeClick: (id: string) => void;
  onCanvasClick: () => void;
};

// This module imports `@neo4j-nvl/*`, which touches `document` at evaluation
// time. It MUST only be loaded in the browser — KnowledgeGraph.tsx imports it
// via React.lazy behind a mounted-on-client guard. Adding new top-level NVL
// imports anywhere else will break SSR.
//
// Why this uses BasicNvlWrapper + manual interaction wiring instead of
// InteractiveNvlWrapper: under React StrictMode (enabled in main.tsx) every
// component mounts → unmounts → remounts on first render. NVL's
// InteractiveNvlWrapper spreads its interaction setup across ~9 separate
// `useInteraction` hook calls, each with its own effect; in the StrictMode
// double-cycle these can land out of order and leave the canvas with a fresh
// NVL instance but no attached ClickInteraction/ZoomInteraction listeners —
// the user sees a graph they can pan but cannot click or wheel-zoom. Wiring
// all interactions inside a single effect we own makes the teardown/recreate
// atomic and StrictMode-safe.
export default function GraphCanvas({ nodes, edges, onNodeClick, onCanvasClick }: Props) {
  const nvlRef = useRef<NVL | null>(null);

  // Memoize translation arrays so NVL doesn't see fresh node/rel references
  // on unrelated parent re-renders (e.g. step events streaming in).
  const nvlNodes: NvlNode[] = useMemo(
    () =>
      nodes.map((n) => ({
        id: n.id,
        caption: captionFor(n),
        color: NODE_COLOR[n.type] ?? NODE_COLOR.item,
        size: n.isTarget ? 44 : 30,
      })),
    [nodes],
  );

  const nvlRels: NvlRel[] = useMemo(
    () =>
      edges.map((e, i) => ({
        id: `${e.source}->${e.target}:${e.label}:${i}`,
        from: e.source,
        to: e.target,
        caption: e.label,
        color: "#9c9a92",
      })),
    [edges],
  );

  // Latest-callback refs so the handlers attached to NVL never need to
  // re-register when the parent passes a fresh callback identity.
  const onNodeClickRef = useRef(onNodeClick);
  const onCanvasClickRef = useRef(onCanvasClick);
  useEffect(() => {
    onNodeClickRef.current = onNodeClick;
    onCanvasClickRef.current = onCanvasClick;
  }, [onNodeClick, onCanvasClick]);

  const nvlOptions = useMemo(
    () => ({
      disableTelemetry: true,
      disableWebGL: true,
      allowDynamicMinZoom: true,
      initialZoom: 1,
      minZoom: 0.05,
      maxZoom: 8,
    }),
    [],
  );

  const wrapperStyle = useMemo(
    () => ({ width: "100%", height: "100%", background: "transparent" }),
    [],
  );

  // Once NVL has constructed itself, wire all interaction handlers in a
  // single atomic block, and also install a defensive trackpad-pinch listener
  // (Mac sends ctrlKey:true wheel events on pinch — NVL's ZoomInteraction
  // already handles this, but we add an explicit fallback in case NVL's
  // listener loses its container reference during a StrictMode cycle).
  const attachInteractions = useCallback((nvl: NVL) => {
    const click = new ClickInteraction(nvl);
    click.updateCallback("onNodeClick", (node: NvlNode) => {
      onNodeClickRef.current(node.id);
    });
    click.updateCallback("onCanvasClick", () => {
      onCanvasClickRef.current();
    });

    const pan = new PanInteraction(nvl);
    const zoom = new ZoomInteraction(nvl);
    const drag = new DragNodeInteraction(nvl);
    const hover = new HoverInteraction(nvl);

    // Defensive trackpad-pinch fallback. Attached to the NVL container so it
    // sits on top of NVL's own wheel listener; if NVL's already handled it
    // (preventDefault'd), this no-ops.
    const container = nvl.getContainer?.() as HTMLElement | undefined;
    const wheelHandler = (evt: WheelEvent) => {
      if (!(evt.ctrlKey || evt.metaKey)) return; // only intercept pinch
      // NVL's own handler runs first (it uses default-phase listener) and
      // already calls preventDefault + setZoomAndPan. If we get here and the
      // event hasn't been defaulted, fall through to a manual setZoom.
      if (evt.defaultPrevented) return;
      evt.preventDefault();
      const cur = nvl.getScale?.() ?? 1;
      const factor = evt.deltaY > 0 ? 0.92 : 1.08;
      nvl.setZoom?.(Math.max(0.05, Math.min(8, cur * factor)));
    };
    container?.addEventListener("wheel", wheelHandler, { passive: false });

    return () => {
      container?.removeEventListener("wheel", wheelHandler);
      try { click.destroy(); } catch { /* no-op */ }
      try { pan.destroy(); } catch { /* no-op */ }
      try { zoom.destroy(); } catch { /* no-op */ }
      try { drag.destroy(); } catch { /* no-op */ }
      try { hover.destroy(); } catch { /* no-op */ }
    };
  }, []);

  // `cleanupRef` holds the dispose function returned by attachInteractions so
  // we can run it on StrictMode synthetic-unmount and on real unmount alike.
  const cleanupRef = useRef<(() => void) | null>(null);

  const tryAttach = useCallback(() => {
    const nvl = nvlRef.current;
    // The BasicNvlWrapper ref is an imperative-handle proxy that exists from
    // first render but only delegates to a real NVL after its internal effect
    // runs. Guard by probing getContainer() — null means "not ready yet".
    const container = nvl?.getContainer?.();
    if (!nvl || !container) return false;
    if (cleanupRef.current) return true; // already attached
    cleanupRef.current = attachInteractions(nvl);
    return true;
  }, [attachInteractions]);

  // Primary attach path: NVL's onInitialization callback.
  const handleInit = useCallback(() => {
    tryAttach();
  }, [tryAttach]);

  // Belt-and-braces: poll for readiness on mount in case onInitialization
  // fires before the ref proxy's inner pointer is populated (a race we've
  // observed under React StrictMode's double-mount in dev).
  useEffect(() => {
    if (tryAttach()) return;
    let cancelled = false;
    const poll = () => {
      if (cancelled) return;
      if (!tryAttach()) {
        setTimeout(poll, 50);
      }
    };
    const t = setTimeout(poll, 16);
    return () => {
      cancelled = true;
      clearTimeout(t);
      cleanupRef.current?.();
      cleanupRef.current = null;
    };
  }, [tryAttach]);

  // Auto-fit after data lands so all nodes are inside the viewport. Two passes
  // because the container can still be growing inside the AgentSteps height
  // animation when the first fit runs.
  useEffect(() => {
    const nvl = nvlRef.current;
    if (!nvl || nodes.length === 0) return;
    const ids = nodes.map((n) => n.id);
    const t1 = setTimeout(() => { try { nvl.fit?.(ids); } catch { /* no-op */ } }, 120);
    const t2 = setTimeout(() => { try { nvl.fit?.(ids); } catch { /* no-op */ } }, 700);
    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
    };
  }, [nodes]);

  const zoomBy = useCallback((factor: number) => {
    const nvl = nvlRef.current;
    if (!nvl) return;
    try {
      const z = nvl.getScale?.() ?? 1;
      nvl.setZoom?.(Math.max(0.05, Math.min(8, z * factor)));
    } catch { /* no-op */ }
  }, []);

  const fitAll = useCallback(() => {
    const nvl = nvlRef.current;
    if (!nvl) return;
    try { nvl.fit?.(nodes.map((n) => n.id)); } catch { /* no-op */ }
  }, [nodes]);

  const setNvlRef = useCallback((r: unknown) => {
    nvlRef.current = r as NVL | null;
  }, []);

  const nvlCallbacks = useMemo(
    () => ({ onInitialization: handleInit }),
    [handleInit],
  );

  return (
    <div className="relative w-full h-full">
      <BasicNvlWrapper
        ref={setNvlRef}
        nodes={nvlNodes}
        rels={nvlRels}
        layout="forceDirected"
        nvlOptions={nvlOptions}
        nvlCallbacks={nvlCallbacks}
        style={wrapperStyle}
      />
      <div className="absolute bottom-3 right-3 flex flex-col gap-1 rounded-md border border-border/60 bg-background/90 backdrop-blur p-1 shadow-sm">
        <CtrlBtn onClick={() => zoomBy(1.25)} label="Zoom in">
          <ZoomIn className="size-3.5" />
        </CtrlBtn>
        <CtrlBtn onClick={() => zoomBy(0.8)} label="Zoom out">
          <ZoomOut className="size-3.5" />
        </CtrlBtn>
        <CtrlBtn onClick={fitAll} label="Fit to view">
          <Maximize2 className="size-3.5" />
        </CtrlBtn>
      </div>
    </div>
  );
}

function CtrlBtn({
  children,
  onClick,
  label,
}: {
  children: React.ReactNode;
  onClick: () => void;
  label: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      aria-label={label}
      className="size-6 grid place-items-center rounded text-foreground/70 hover:text-foreground hover:bg-muted transition-colors"
    >
      {children}
    </button>
  );
}
