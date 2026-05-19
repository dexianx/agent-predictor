import { motion } from "framer-motion";
import ReactMarkdown from "react-markdown";
import { Star, AlertTriangle } from "lucide-react";
import type { ChatMessage } from "@/lib/types";
import type { PersonaDetail } from "@/lib/types";
import { useMemo } from "react";
import { AgentSteps } from "./AgentSteps";
import { KnowledgeGraph } from "./KnowledgeGraph";

type Props = {
  message: ChatMessage;
  persona: PersonaDetail | null;
  streaming?: boolean;
};

const EASE = [0.22, 1, 0.36, 1] as const;

function StarRating({ rating }: { rating: number }) {
  return (
    <div className="flex items-center gap-1.5 mt-3">
      {[1, 2, 3, 4, 5].map((s) => (
        <Star
          key={s}
          className={`size-5 ${s <= rating ? "fill-primary text-primary" : "text-border"}`}
        />
      ))}
      <span className="font-serif text-xl font-semibold tabular-nums ml-1">{rating}★</span>
    </div>
  );
}

export function MessageView({ message, persona, streaming }: Props) {
  if (message.role === "user") {
    return (
      <motion.div
        initial={{ opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, ease: EASE }}
        className="flex justify-end"
      >
        <div className="max-w-[88%] sm:max-w-[78%] rounded-2xl rounded-tr-md bg-primary text-primary-foreground px-4 py-2.5 text-[15px] leading-relaxed shadow-[0_8px_22px_-18px_rgba(217,119,87,0.55)]">
          {message.text}
        </div>
      </motion.div>
    );
  }

  const hasSteps = (message.steps?.length ?? 0) > 0;
  const hasText = !!message.text;
  const expandable = useMemo(
    () =>
      message.graph
        ? { retrieval: <KnowledgeGraph data={message.graph} /> }
        : undefined,
    [message.graph],
  );

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4, ease: EASE }}
      className="min-w-0"
    >
      {hasSteps && (
        <div className="mb-4">
          <AgentSteps
            steps={message.steps!}
            running={!!streaming}
            streamingText={!!streaming && hasText}
            expandable={expandable}
          />
        </div>
      )}

      {hasText &&
        (() => {
          // Backend currently overwrites `text` with "Error: <msg>" on the
          // error WS event. Render those as a proper notice card instead of
          // raw markdown so they don't look like a UI glitch — and so an empty
          // `ev.message` doesn't surface as a bare "Error:" line.
          const errorMatch = /^Error:\s*(.*)$/s.exec(message.text);
          if (errorMatch && !message.payload && !streaming) {
            const detail = errorMatch[1].trim();
            return (
              <motion.div
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, ease: EASE }}
                className="mt-1 flex gap-3 rounded-2xl border border-destructive/40 bg-destructive/5 px-4 py-3"
              >
                <AlertTriangle className="size-4 text-destructive mt-0.5 shrink-0" />
                <div className="min-w-0">
                  <div className="text-[13px] font-medium text-destructive">
                    Review generation failed
                  </div>
                  <div className="text-[12.5px] text-muted-foreground mt-0.5 break-words">
                    {detail
                      ? detail
                      : "The backend returned an error without a message. Check the API logs and try again."}
                  </div>
                </div>
              </motion.div>
            );
          }
          return (
            <div
              className={`text-[15px] md:text-[15.5px] leading-[1.7] text-foreground prose prose-sm max-w-none prose-p:my-0 prose-strong:text-foreground prose-strong:font-semibold${
                streaming ? " prose-streaming" : ""
              }`}
            >
              <ReactMarkdown>{message.text}</ReactMarkdown>
            </div>
          );
        })()}

      {message.payload?.type === "review" && !streaming && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: EASE }}
          className="mt-3 rounded-2xl border border-border/70 bg-card p-5 shadow-[0_10px_30px_-22px_rgba(217,119,87,0.35)]"
        >
          <div className="text-xs text-muted-foreground mb-2">
            Simulated review · {message.payload.locale ?? "??"} · via{" "}
            {message.payload.retrieval_branch}
          </div>
          <blockquote className="pl-3 border-l-2 border-primary/40 text-foreground leading-relaxed">
            {message.payload.review_text}
          </blockquote>
          <StarRating rating={message.payload.predicted_rating} />
        </motion.div>
      )}

    </motion.div>
  );
}
