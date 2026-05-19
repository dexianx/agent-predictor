import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Sidebar } from "@/components/chat/Sidebar";
import { MobileNav } from "@/components/chat/MobileNav";
import { Composer } from "@/components/chat/Composer";
import type { ProductFields } from "@/components/chat/Composer";
import { EmptyState } from "@/components/chat/EmptyState";
import { MessageView } from "@/components/chat/MessageView";
import { PersonaDetailModal } from "@/components/chat/PersonaDetailModal";
import { appendMessage, newMessageId, updateMessage, useThread } from "@/hooks/useThreads";
import { getPersonaDetail } from "@/lib/api";
import { connectSimulate } from "@/lib/ws";
import { getPersonaDisplayName } from "@/lib/persona-utils";
import type {
  ChatMessage,
  AssistantPayload,
  PersonaDetail,
  GraphData,
  GraphNode,
  GraphEdge,
  GraphDataEvent,
} from "@/lib/types";

function mergeGraph(prev: GraphData | undefined, ev: GraphDataEvent): GraphData {
  const nodeMap = new Map<string, GraphNode>();
  const edgeMap = new Map<string, GraphEdge>();
  if (prev) {
    prev.nodes.forEach((n) => nodeMap.set(n.id, n));
    prev.edges.forEach((e) => edgeMap.set(`${e.source}->${e.target}:${e.label}`, e));
  }
  ev.nodes.forEach((n) => nodeMap.set(n.id, n));
  ev.edges.forEach((e) => edgeMap.set(`${e.source}->${e.target}:${e.label}`, e));
  return {
    branch: ev.branch,
    nodes: [...nodeMap.values()],
    edges: [...edgeMap.values()],
  };
}

export const Route = createFileRoute("/c/$threadId")({
  component: ChatPage,
  head: () => ({
    meta: [
      { title: "Kora — Chat" },
      { name: "description", content: "Persona-grounded review simulation." },
    ],
  }),
});

function ChatPage() {
  const { threadId } = Route.useParams();
  const thread = useThread(threadId);
  const [persona, setPersonaState] = useState<PersonaDetail | null>(null);
  const [streamingId, setStreamingId] = useState<string | null>(null);
  const [detailOpen, setDetailOpen] = useState(false);
  const stopRef = useRef(false);
  const transcriptRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const pid = thread?.personaId;
    if (!pid) {
      setPersonaState(null);
      return;
    }
    let cancelled = false;
    getPersonaDetail(pid)
      .then((p) => {
        if (!cancelled) setPersonaState(p);
      })
      .catch(() => {
        if (!cancelled) setPersonaState(null);
      });
    return () => {
      cancelled = true;
    };
  }, [thread?.personaId]);

  useEffect(() => {
    const el = transcriptRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [thread?.messages.length, streamingId]);

  if (!thread) {
    return (
      <div className="min-h-screen grid place-items-center text-sm text-muted-foreground">
        Loading thread…
      </div>
    );
  }

  const handleSubmit = async (fields: ProductFields) => {
    if (streamingId) return;

    const { item_name, item_description, price_ngn } = fields;
    const displayText = [
      item_name,
      item_description || null,
      price_ngn != null
        ? `${persona?.locale === "US" ? "$" : "₦"}${price_ngn.toLocaleString()}`
        : null,
    ]
      .filter(Boolean)
      .join(" · ");

    const userId = thread.personaId;
    if (!userId) {
      const asstId = newMessageId();
      appendMessage(threadId, { id: newMessageId(), role: "user", text: displayText, createdAt: Date.now() });
      appendMessage(threadId, {
        id: asstId,
        role: "assistant",
        text: "Select a persona from the sidebar first.",
        createdAt: Date.now(),
      });
      return;
    }

    appendMessage(threadId, { id: newMessageId(), role: "user", text: displayText, createdAt: Date.now() });

    const asstId = newMessageId();
    appendMessage(threadId, {
      id: asstId,
      role: "assistant",
      text: "",
      steps: [],
      createdAt: Date.now(),
    });
    setStreamingId(asstId);
    stopRef.current = false;

    let accText = "";
    let accGraph: GraphData | undefined;
    const accSteps: ChatMessage["steps"] = [];
    try {
      for await (const ev of connectSimulate({
        user_id: userId,
        item_name,
        item_description: item_description || undefined,
        price_ngn,
      })) {
        if (stopRef.current) break;

        if (ev.type === "step") {
          accSteps.push({ id: ev.id, label: ev.label, detail: ev.detail });
          updateMessage(threadId, asstId, { steps: [...accSteps] });
        } else if (ev.type === "token") {
          accText += ev.text;
          updateMessage(threadId, asstId, { text: accText });
        } else if (ev.type === "graph_data") {
          accGraph = mergeGraph(accGraph, ev);
          updateMessage(threadId, asstId, { graph: accGraph });
        } else if (ev.type === "final") {
          const payload: AssistantPayload = {
            type: "review",
            review_text: ev.review_text,
            predicted_rating: ev.predicted_rating,
            user_id: ev.user_id,
            item_id: ev.item_id,
            locale: ev.locale,
            retrieval_branch: ev.retrieval_branch,
          };
          updateMessage(threadId, asstId, { text: accText, payload, steps: [...accSteps] });
        } else if (ev.type === "error") {
          updateMessage(threadId, asstId, { text: `Error: ${ev.message}`, steps: [...accSteps] });
        }
      }
    } catch (err) {
      updateMessage(threadId, asstId, {
        text: `Connection error: ${String(err)}`,
        steps: [...accSteps],
      });
    } finally {
      setStreamingId(null);
    }
  };

  const personaLabel = persona
    ? getPersonaDisplayName(persona)
    : thread.personaId
      ? thread.personaId.slice(0, 18)
      : "No persona";

  return (
    <div className="flex h-svh w-full bg-background overflow-hidden">
      <Sidebar />

      <main className="flex-1 flex flex-col min-w-0">
        {/* Mobile header */}
        <header className="md:hidden h-14 border-b border-border/60 px-3 flex items-center gap-2.5 bg-background/90 backdrop-blur-md shrink-0">
          <MobileNav />
          <div className="flex-1 min-w-0">
            <div className="font-serif text-[14.5px] leading-tight truncate">
              {thread.title || "New chat"}
            </div>
            <button
              onClick={() => persona && setDetailOpen(true)}
              disabled={!persona}
              className="text-[11px] text-muted-foreground truncate text-left hover:text-primary transition-colors disabled:cursor-default"
            >
              {personaLabel}
              {persona && <span className="ml-1 opacity-40 text-[9px]">↗</span>}
            </button>
          </div>
        </header>

        {/* Desktop header */}
        <header className="hidden md:flex h-14 border-b border-border/60 px-6 items-center bg-background/85 backdrop-blur-md shrink-0">
          <div className="min-w-0">
            <div className="font-serif text-[15px] leading-tight truncate">
              {thread.title || "New chat"}
            </div>
            <button
              onClick={() => persona && setDetailOpen(true)}
              disabled={!persona}
              className="text-[11.5px] text-muted-foreground truncate mt-0.5 text-left hover:text-primary transition-colors disabled:cursor-default flex items-center gap-1"
            >
              <span>{personaLabel}</span>
              {persona && <span className="opacity-40 text-[10px]">↗ view profile</span>}
            </button>
          </div>
        </header>

        <div ref={transcriptRef} className="flex-1 overflow-y-auto kora-scroll" key={threadId}>
          {thread.messages.length === 0 ? (
            <EmptyState persona={persona} onPick={handleSubmit} />
          ) : (
            <div className="max-w-3xl mx-auto px-4 md:px-6 pt-6 md:pt-10 pb-6 space-y-8">
              {thread.messages.map((m) => (
                <MessageView
                  key={m.id}
                  message={m}
                  persona={persona}
                  streaming={streamingId === m.id}
                />
              ))}
              {streamingId &&
                thread.messages[thread.messages.length - 1]?.id === streamingId &&
                !thread.messages[thread.messages.length - 1]?.text &&
                (thread.messages[thread.messages.length - 1]?.steps?.length ?? 0) === 0 && (
                  <motion.div
                    initial={{ opacity: 0, y: 2 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4 }}
                    className="relative pl-6"
                  >
                    <span className="absolute left-[6px] top-1.5 size-[9px] rounded-full bg-primary shadow-[0_0_0_4px_var(--background)]">
                      <span className="absolute inset-0 rounded-full bg-primary/40 animate-ping" />
                    </span>
                    <span className="text-[14px] shimmer-text font-medium">Thinking</span>
                  </motion.div>
                )}
            </div>
          )}
        </div>

        <Composer
          threadKey={threadId}
          persona={persona}
          busy={!!streamingId}
          onStop={() => {
            stopRef.current = true;
          }}
          onSubmit={handleSubmit}
        />
      </main>

      {/* Persona detail modal — accessible any time including during simulation */}
      <AnimatePresence>
        {detailOpen && persona && (
          <PersonaDetailModal
            userId={persona.user_id}
            open={detailOpen}
            onClose={() => setDetailOpen(false)}
          />
        )}
      </AnimatePresence>
    </div>
  );
}
