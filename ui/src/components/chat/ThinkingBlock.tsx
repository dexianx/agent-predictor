import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useState } from "react";
import { Clock, ChevronRight, Loader2 } from "lucide-react";
import ReactMarkdown from "react-markdown";

const EASE = [0.22, 1, 0.36, 1] as const;

type Props = {
  text: string;
  streaming: boolean;
  done: boolean;
};

export function ThinkingBlock({ text, streaming, done }: Props) {
  const [open, setOpen] = useState(true);

  // Auto-collapse 700ms after recommendations are ready
  useEffect(() => {
    if (done) {
      const t = setTimeout(() => setOpen(false), 700);
      return () => clearTimeout(t);
    }
  }, [done]);

  if (!text) return null;

  // Summarise the first ~60 chars for the collapsed label
  const preview = text.replace(/\s+/g, " ").trim().slice(0, 72);

  return (
    <div className="mb-3 rounded-xl border border-border/40 bg-muted/20 overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-muted/30 transition-colors"
      >
        <motion.span
          animate={{ rotate: open ? 90 : 0 }}
          transition={{ duration: 0.2, ease: EASE }}
          className="text-muted-foreground/60 shrink-0"
        >
          <ChevronRight className="size-3.5" />
        </motion.span>

        <Clock className="size-3.5 text-muted-foreground/60 shrink-0" />

        <span className="text-[13px] text-muted-foreground/80 flex-1 min-w-0">
          {open ? (
            <span className="font-medium">
              {streaming ? "Thinking" : "Thought"}
            </span>
          ) : (
            <span className="truncate block">
              {preview}
              {preview.length < text.trim().length && "…"}
            </span>
          )}
        </span>

        {streaming && open && (
          <Loader2 className="size-3.5 text-muted-foreground/60 animate-spin shrink-0" />
        )}
      </button>

      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            key="body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: EASE }}
            className="overflow-hidden"
          >
            <div className="px-4 pt-1 pb-3 border-t border-border/30">
              <div
                className={`text-[13.5px] leading-[1.65] text-muted-foreground/75 prose prose-sm max-w-none
                  prose-p:my-1 prose-strong:text-muted-foreground/90 prose-strong:font-semibold
                  [overflow-wrap:break-word] [word-break:break-word]${streaming ? " prose-streaming" : ""}`}
              >
                <ReactMarkdown>{text}</ReactMarkdown>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
