import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useState } from "react";
import { Globe, ChevronDown, Loader2 } from "lucide-react";
import type { ToolCallStep } from "@/lib/types";

const EASE = [0.22, 1, 0.36, 1] as const;

type Props = { step: ToolCallStep };

export function ToolCallCard({ step }: Props) {
  const [open, setOpen] = useState(true);

  // Auto-collapse when tool call completes
  useEffect(() => {
    if (step.status === "done") {
      const t = setTimeout(() => setOpen(false), 1200);
      return () => clearTimeout(t);
    }
  }, [step.status]);

  const query = (step.input as { query?: string })?.query ?? "";
  const results = (step.output as { results?: Array<{ title: string; url: string; description: string }> })?.results ?? [];

  return (
    <div className="mt-1 mb-1 rounded-lg border border-border/60 bg-muted/30 overflow-hidden text-[13px] min-w-0">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center gap-2 px-3 py-2 hover:bg-muted/50 transition-colors text-left min-w-0"
      >
        <Globe className="size-3.5 text-primary shrink-0" />
        <span className="font-medium text-foreground/90 flex-1 truncate">
          {step.label}
          {query && (
            <span className="ml-1.5 font-normal text-muted-foreground">— {query}</span>
          )}
        </span>
        {step.status === "running" ? (
          <Loader2 className="size-3.5 text-primary animate-spin shrink-0" />
        ) : (
          <motion.span
            animate={{ rotate: open ? 0 : -90 }}
            transition={{ duration: 0.2, ease: EASE }}
            className="text-muted-foreground shrink-0"
          >
            <ChevronDown className="size-3.5" />
          </motion.span>
        )}
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            key="body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.28, ease: EASE }}
            className="overflow-hidden"
          >
            <div className="px-3 pb-3 pt-1 space-y-2">
              {/* Input */}
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-1">
                  Query
                </p>
                <p className="text-foreground/80 bg-background/60 rounded px-2 py-1 border border-border/40">
                  {query || JSON.stringify(step.input)}
                </p>
              </div>

              {/* Output */}
              {step.status === "done" && results.length > 0 && (
                <div>
                  <p className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground mb-1">
                    Results ({results.length})
                  </p>
                  <ul className="space-y-1.5">
                    {results.map((r, i) => (
                      <li key={i} className="bg-background/60 rounded px-2 py-1.5 border border-border/40 min-w-0 overflow-hidden">
                        <p className="font-medium text-foreground/90 truncate min-w-0">{r.title}</p>
                        {r.url && (
                          <p className="text-primary/70 text-[11px] truncate min-w-0 overflow-hidden">{r.url}</p>
                        )}
                        {r.description && (
                          <p className="text-muted-foreground text-[12px] mt-0.5 line-clamp-2 break-words">
                            {r.description}
                          </p>
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {step.status === "running" && (
                <p className="text-muted-foreground italic">Fetching results…</p>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
