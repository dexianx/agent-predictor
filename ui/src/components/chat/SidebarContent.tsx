import { useState, useEffect } from "react";
import { Link, useNavigate, useParams } from "@tanstack/react-router";
import { motion, AnimatePresence } from "framer-motion";
import { Plus, Search as SearchIcon } from "lucide-react";
import koraMark from "@/assets/kora-mark.png";
import { useThreads, createThread, deleteThread, setPersona } from "@/hooks/useThreads";
import { getFeaturedPersonas } from "@/lib/api";
import type { PersonaDetail } from "@/lib/types";
import { getPersonaDisplayName, getPersonaShortId } from "@/lib/persona-utils";
import { PersonaDetailModal } from "./PersonaDetailModal";
import { PersonaSearch } from "./PersonaSearch";

type Props = {
  onClose?: () => void;
};

export function SidebarContent({ onClose }: Props) {
  const threads = useThreads();
  const navigate = useNavigate();
  const params = useParams({ strict: false }) as { threadId?: string };
  const activeId = params.threadId;

  const [featured, setFeatured] = useState<PersonaDetail[]>([]);
  const [pinned, setPinned] = useState<PersonaDetail[]>([]);
  const [showSearch, setShowSearch] = useState(false);
  const [detailUserId, setDetailUserId] = useState<string | null>(null);

  useEffect(() => {
    getFeaturedPersonas()
      .then(setFeatured)
      .catch(() => {/* API not running yet */});
  }, []);

  const allPersonas = [
    ...featured,
    ...pinned.filter((p) => !featured.some((f) => f.user_id === p.user_id)),
  ];

  const newChat = () => {
    const t = createThread(threads.find((x) => x.id === activeId)?.personaId);
    navigate({ to: "/c/$threadId", params: { threadId: t.id } });
    onClose?.();
  };

  const onDelete = (id: string) => {
    deleteThread(id);
    if (id === activeId) {
      const remaining = threads.filter((t) => t.id !== id);
      if (remaining.length > 0) {
        navigate({ to: "/c/$threadId", params: { threadId: remaining[0].id } });
      } else {
        const t = createThread();
        navigate({ to: "/c/$threadId", params: { threadId: t.id } });
      }
    }
  };

  const activeThread = threads.find((t) => t.id === activeId);

  return (
    <>
      <div className="flex flex-col h-full">
        <div className="px-4 pt-5 pb-3 flex items-center gap-2">
          <img src={koraMark} alt="Kora" width={28} height={28} className="rounded-md" />
          <div>
            <div className="font-serif font-normal text-base leading-none tracking-tight">Kora</div>
            <div className="text-[10px] uppercase tracking-widest text-muted-foreground mt-1">
              Persona-aware agent
            </div>
          </div>
        </div>

        <button
          onClick={newChat}
          className="mx-3 mb-3 flex items-center gap-2 px-3 py-2 rounded-lg border border-border/70 bg-card hover:border-primary/50 hover:bg-card/90 transition-all text-sm font-medium"
        >
          <Plus className="size-3.5" />
          New chat
        </button>

        {/* Persona section */}
        <div className="px-4 pt-2 pb-1 flex items-center justify-between">
          <span className="text-[10px] uppercase tracking-widest text-muted-foreground">Persona</span>
          <button
            onClick={() => setShowSearch((v) => !v)}
            className="size-5 rounded grid place-items-center hover:text-primary text-muted-foreground transition-colors"
            title="Search personas"
          >
            <SearchIcon className="size-3" />
          </button>
        </div>

        <div className="px-3 mb-1">
          {showSearch && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              className="mb-2"
            >
              <PersonaSearch onPin={(p) => setPinned((prev) => [p, ...prev])} />
            </motion.div>
          )}
        </div>

        <div className="px-3 mb-4 space-y-1 max-h-48 overflow-y-auto kora-scroll">
          {allPersonas.length === 0 && (
            <div className="text-xs text-muted-foreground px-2 py-1">Loading personas…</div>
          )}
          {allPersonas.map((p) => {
            const active = activeThread?.personaId === p.user_id;
            const isPinned = pinned.some((pp) => pp.user_id === p.user_id);
            return (
              <div key={p.user_id} className="group flex items-center gap-1 px-1 py-1.5 rounded-lg hover:bg-sand transition-colors cursor-pointer">
                <button
                  onClick={() => activeThread && setPersona(activeThread.id, p.user_id)}
                  className="flex-1 text-left min-w-0"
                >
                  <span className={`text-[12.5px] font-medium leading-tight block truncate ${active ? "text-primary" : "text-foreground/85"}`}>
                    {getPersonaDisplayName(p)}
                  </span>
                  <span className="text-[10px] text-muted-foreground font-mono">
                    {getPersonaShortId(p)}{active ? " · Active" : ""}{isPinned ? " · pinned" : ""}
                  </span>
                </button>
                <button
                  onClick={(e) => { e.stopPropagation(); setDetailUserId(p.user_id); }}
                  className="shrink-0 text-[11px] text-muted-foreground/50 hover:text-primary transition-colors px-1.5"
                  title="View profile"
                >↗</button>
              </div>
            );
          })}
        </div>

        {/* Chats section */}
        <div className="px-4 pt-2 pb-1 text-[10px] uppercase tracking-widest text-muted-foreground">
          Chats
        </div>
        <div className="flex-1 overflow-y-auto px-2 pb-3 kora-scroll">
          <AnimatePresence initial={false}>
            {threads.map((t) => {
              const active = t.id === activeId;
              return (
                <motion.div
                  key={t.id}
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.18 }}
                  className="group relative flex items-center"
                >
                  <Link
                    to="/c/$threadId"
                    params={{ threadId: t.id }}
                    onClick={onClose}
                    className={`flex-1 min-w-0 flex items-center pl-2.5 pr-7 py-2 rounded-md text-sm transition-all ${
                      active
                        ? "bg-card text-foreground border border-border/70"
                        : "text-ink-soft hover:bg-sand"
                    }`}
                  >
                    <span className="truncate">{t.title || "New chat"}</span>
                  </Link>
                  <button
                    onClick={() => onDelete(t.id)}
                    aria-label="Delete chat"
                    className="absolute right-1.5 grid place-items-center size-7 rounded-md text-[11px] text-muted-foreground opacity-0 group-hover:opacity-100 hover:bg-destructive/10 hover:text-destructive transition-all font-medium"
                  >
                    ✕
                  </button>
                </motion.div>
              );
            })}
          </AnimatePresence>
        </div>
      </div>

      {/* Persona detail sheet */}
      <AnimatePresence>
        {detailUserId && (
          <PersonaDetailModal
            userId={detailUserId}
            open={!!detailUserId}
            onClose={() => setDetailUserId(null)}
          />
        )}
      </AnimatePresence>
    </>
  );
}
