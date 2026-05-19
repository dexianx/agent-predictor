import { useState, useRef } from "react";
import { Search, UserPlus } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { searchPersonas } from "@/lib/api";
import type { PersonaDetail } from "@/lib/types";

type Props = {
  onPin: (persona: PersonaDetail) => void;
};

export function PersonaSearch({ onPin }: Props) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<PersonaDetail[]>([]);
  const [loading, setLoading] = useState(false);
  const debounce = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleInput = (value: string) => {
    setQuery(value);
    if (debounce.current) clearTimeout(debounce.current);
    if (!value.trim()) {
      setResults([]);
      return;
    }
    debounce.current = setTimeout(async () => {
      setLoading(true);
      try {
        const r = await searchPersonas(value.trim(), undefined, 10);
        setResults(r);
      } finally {
        setLoading(false);
      }
    }, 350);
  };

  return (
    <div className="space-y-1">
      <div className="relative">
        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 size-3.5 text-muted-foreground" />
        <input
          value={query}
          onChange={(e) => handleInput(e.target.value)}
          placeholder="Search 25K personas…"
          className="w-full pl-8 pr-3 py-1.5 rounded-lg bg-sand border border-border/60 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary/40"
        />
      </div>

      <AnimatePresence>
        {(results.length > 0 || loading) && (
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            className="rounded-lg border border-border/60 bg-card shadow-md overflow-hidden"
          >
            {loading && (
              <div className="px-3 py-2 text-xs text-muted-foreground animate-pulse">
                Searching…
              </div>
            )}
            {results.map((p) => (
              <div
                key={p.user_id}
                className="flex items-center gap-2 px-3 py-2 hover:bg-sand transition-colors group"
              >
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-mono truncate">{p.user_id.slice(0, 16)}…</div>
                  <div className="text-[11px] text-muted-foreground">
                    {p.locale ?? "??"} · {p.review_count} reviews
                    {p.top_categories[0] && ` · ${p.top_categories[0]}`}
                  </div>
                </div>
                <button
                  onClick={() => { onPin(p); setQuery(""); setResults([]); }}
                  className="opacity-0 group-hover:opacity-100 transition-opacity size-7 rounded-md grid place-items-center hover:bg-primary/10 hover:text-primary"
                  title="Add to session"
                >
                  <UserPlus className="size-3.5" />
                </button>
              </div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
