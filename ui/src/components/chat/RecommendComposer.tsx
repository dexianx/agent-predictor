import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { ArrowUp, Square } from "lucide-react";
import type { PersonaDetail } from "@/lib/types";

type Props = {
  onSubmit: (contextQuery: string) => void;
  onStop?: () => void;
  busy: boolean;
  persona: PersonaDetail | null;
  threadKey: string;
};

export function RecommendComposer({ onSubmit, onStop, busy, persona, threadKey }: Props) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [canSend, setCanSend] = useState(false);

  useEffect(() => { textareaRef.current?.focus(); }, [threadKey]);
  useEffect(() => { if (!busy) textareaRef.current?.focus(); }, [busy]);

  const checkCanSend = () => {
    setCanSend(!!(textareaRef.current?.value.trim()));
  };

  const send = () => {
    const query = textareaRef.current?.value.trim() ?? "";
    if (!query) return;
    onSubmit(query);
    if (textareaRef.current) {
      textareaRef.current.value = "";
      textareaRef.current.style.height = "auto";
    }
    setCanSend(false);
  };

  const autosize = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 140) + "px";
  };

  return (
    <div className="px-3 md:px-4 pb-[max(16px,env(safe-area-inset-bottom,16px))] md:pb-5 pt-2 shrink-0">
      <form
        onSubmit={(e) => { e.preventDefault(); send(); }}
        className="md:mx-auto md:max-w-3xl rounded-[18px] md:rounded-[20px] border border-border bg-card transition-[border-color,box-shadow] duration-300 focus-within:border-foreground/25 focus-within:shadow-[0_2px_24px_-12px_rgba(20,20,19,0.18)]"
      >
        <div className="px-4 md:px-5 pt-3.5 md:pt-4 pb-2">
          <textarea
            ref={textareaRef}
            rows={2}
            placeholder="What would you like recommended? e.g. a budget power bank for travel…"
            onInput={() => { autosize(); checkCanSend(); }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
            }}
            className="w-full bg-transparent text-[15px] md:text-[15.5px] leading-[1.55] outline-none placeholder:text-muted-foreground/60 resize-none max-h-[140px]"
            disabled={busy}
          />
        </div>

        <div className="px-3 pb-2.5 md:pb-3 pt-1 flex items-center justify-between gap-2">
          <div className="pl-1 md:pl-2 text-[11.5px] md:text-[12px] text-muted-foreground/80 truncate min-w-0">
            Recommending for{" "}
            <span className="text-foreground/80 font-medium font-mono text-[11px]">
              {persona ? persona.user_id.slice(0, 16) : "—"}
            </span>
            {persona?.locale && (
              <>
                <span className="mx-1.5 text-muted-foreground/50">·</span>
                <span className="hidden sm:inline">{persona.locale}</span>
              </>
            )}
          </div>

          {busy ? (
            <motion.button
              key="stop"
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              type="button"
              onClick={onStop}
              className="shrink-0 grid place-items-center size-8 rounded-full bg-primary text-primary-foreground hover:bg-primary/85 transition-colors"
              aria-label="Stop"
            >
              <Square className="size-3 fill-current" />
            </motion.button>
          ) : (
            <motion.button
              key="send"
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              whileTap={{ scale: 0.94 }}
              type="submit"
              disabled={!canSend}
              className="shrink-0 grid place-items-center size-8 rounded-full bg-primary text-primary-foreground transition-all disabled:bg-primary/20 disabled:text-primary-foreground/50"
              aria-label="Send"
            >
              <ArrowUp className="size-4" strokeWidth={2.4} />
            </motion.button>
          )}
        </div>
      </form>
    </div>
  );
}
