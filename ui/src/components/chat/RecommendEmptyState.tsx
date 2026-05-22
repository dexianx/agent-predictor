import { motion } from "framer-motion";
import type { PersonaDetail } from "@/lib/types";
import { getPersonaDisplayName } from "@/lib/persona-utils";

const SAMPLE_QUERIES = [
  {
    label: "Budget power bank for travel",
    query: "Recommend me a good budget power bank for travel",
    hint: "40000mAh · wireless charging · under $50",
  },
  {
    label: "Wireless keyboard for remote work",
    query: "Find me a reliable wireless keyboard for remote work setup",
    hint: "Multi-device · backlit · USB-C charging",
  },
  {
    label: "Air fryer for a small family",
    query: "What air fryer should I get for a small family of 4?",
    hint: "5-6qt · easy clean · one-touch presets",
  },
  {
    label: "Phone case for Galaxy S25",
    query: "Suggest a durable phone case for my Samsung Galaxy S25",
    hint: "Military-grade · slim · kickstand",
  },
  {
    label: "Camera gimbal for travel",
    query: "What is a good camera gimbal for travel vlogging?",
    hint: "Foldable · stabilised · lightweight",
  },
  {
    label: "Best value kitchen appliance under $100",
    query: "Recommend the best value kitchen appliance under $100",
    hint: "High rating · popular · versatile",
  },
];

const EASE = [0.22, 1, 0.36, 1] as const;

export function RecommendEmptyState({
  persona,
  onPick,
}: {
  persona: PersonaDetail | null;
  onPick: (query: string) => void;
}) {
  const displayName = persona ? getPersonaDisplayName(persona) : "this persona";
  return (
    <div className="max-w-3xl mx-auto px-4 md:px-6 pt-12 md:pt-24 pb-10">
      <motion.div
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.55, ease: EASE }}
      >
        <h1 className="font-serif text-[clamp(28px,7vw,40px)] leading-[1.1] tracking-tight text-foreground">
          What can I find for you<span className="text-primary">?</span>
        </h1>
        <p className="mt-3 text-ink-soft text-[14.5px] md:text-[15.5px] leading-relaxed max-w-xl">
          Kora will search the knowledge graph and reason through the best picks for{" "}
          <span className="text-foreground font-medium">{displayName}</span>'s taste profile.
        </p>
      </motion.div>

      <motion.div
        initial="hidden"
        animate="show"
        variants={{
          hidden: {},
          show: { transition: { staggerChildren: 0.04, delayChildren: 0.2 } },
        }}
        className="mt-8 md:mt-10 flex flex-col"
      >
        {SAMPLE_QUERIES.map((q) => (
          <motion.button
            key={q.label}
            variants={{
              hidden: { opacity: 0, y: 4 },
              show: { opacity: 1, y: 0, transition: { duration: 0.4, ease: EASE } },
            }}
            onClick={() => onPick(q.query)}
            className="group text-left py-3 border-t border-border/60 last:border-b hover:text-foreground active:text-foreground transition-colors flex items-center justify-between gap-4 min-h-[52px]"
          >
            <span className="flex flex-col gap-0.5 min-w-0">
              <span className="text-[14.5px] md:text-[15px] text-ink-soft group-hover:text-foreground transition-colors truncate">
                {q.label}
              </span>
              <span className="text-[12px] text-muted-foreground/60 truncate">
                {q.hint}
              </span>
            </span>
            <span
              aria-hidden
              className="shrink-0 text-muted-foreground/40 group-hover:text-foreground/60 group-hover:translate-x-0.5 transition-all"
            >
              ↗
            </span>
          </motion.button>
        ))}
      </motion.div>
    </div>
  );
}
