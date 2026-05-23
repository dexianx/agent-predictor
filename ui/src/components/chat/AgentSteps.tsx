import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Bot, ChevronDown } from "lucide-react";
import { ToolCallCard } from "./ToolCallCard";
import type { AnyStep, ToolCallStep } from "@/lib/types";

export type AgentStepData = {
  id: string;
  label: string;
  detail?: string;
};

type Props = {
  steps: AnyStep[];
  running: boolean;
  streamingText?: boolean;
  /**
   * Map of `step.id` → expandable content rendered beneath that step. Steps
   * with expandable content auto-open while the step is active and auto-collapse
   * once the trace moves on. Users can toggle either state manually; manual
   * overrides reset when the step becomes active again.
   */
  expandable?: Record<string, ReactNode>;
};

const EASE = [0.22, 1, 0.36, 1] as const;

type ManualState = "open" | "closed" | undefined;

export function AgentSteps({ steps, running, streamingText, expandable }: Props) {
  // activeIndex points at the currently-running step, or one past the last
  // step when nothing is running anymore.
  const activeIndex = running && !streamingText ? steps.length - 1 : steps.length;

  // Manual open/close overrides per step id. Cleared automatically when the
  // step becomes the active one again so the auto-open rule wins on revisit.
  const [manual, setManual] = useState<Record<string, ManualState>>({});
  const prevActiveStepId = useRef<string | undefined>(undefined);
  const currentActiveId = activeIndex < steps.length ? steps[activeIndex]?.id : undefined;

  useEffect(() => {
    if (
      currentActiveId &&
      currentActiveId !== prevActiveStepId.current &&
      manual[currentActiveId] !== undefined
    ) {
      setManual((m) => {
        const next = { ...m };
        delete next[currentActiveId];
        return next;
      });
    }
    prevActiveStepId.current = currentActiveId;
    // We intentionally exclude `manual` so re-entering a step resets its override
    // exactly once on activation.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentActiveId]);

  const isOpen = (_i: number, id: string) => {
    const state = manual[id];
    if (state === "open") return true;
    if (state === "closed") return false;
    // Default: expandable sections stay open. The user can collapse manually.
    return true;
  };

  const toggle = (i: number, id: string) => {
    setManual((m) => ({
      ...m,
      [id]: isOpen(i, id) ? "closed" : "open",
    }));
  };

  // Track per-step row heights so the rail and dot stay aligned even when
  // expandable content is open.
  const rowRefs = useRef<Array<HTMLLIElement | null>>([]);
  const [heights, setHeights] = useState<number[]>([]);
  const recomputeHeights = () => {
    setHeights(rowRefs.current.map((el) => (el?.offsetHeight ?? 28)));
  };

  useEffect(() => {
    recomputeHeights();
  }, [steps.length, manual, expandable]);

  // Re-measure when expandable content (the graph) animates open/closed.
  useEffect(() => {
    const ro = new ResizeObserver(() => recomputeHeights());
    rowRefs.current.forEach((el) => el && ro.observe(el));
    return () => ro.disconnect();
  }, [steps.length]);

  const dotY = useMemo(() => {
    const idx = Math.min(activeIndex, steps.length - 1);
    let y = 0;
    for (let i = 0; i < idx; i++) y += heights[i] ?? 28;
    return y + 10;
  }, [activeIndex, steps.length, heights]);

  const totalHeight = heights.reduce((a, b) => a + b, 0) || steps.length * 28;

  if (steps.length === 0) return null;

  return (
    <div className="flex gap-3 min-w-0 overflow-hidden">
      <div className="shrink-0 mt-0.5">
        <motion.div
          animate={running && !streamingText ? { scale: [1, 1.12, 1] } : { scale: 1 }}
          transition={{
            duration: 1.4,
            repeat: running && !streamingText ? Infinity : 0,
            ease: "easeInOut",
          }}
          className="size-6 rounded-md bg-primary/10 grid place-items-center"
        >
          <Bot className="size-3.5 text-primary" />
        </motion.div>
      </div>

      <div className="relative pl-6 flex-1 min-w-0 overflow-hidden" style={{ minHeight: totalHeight }}>
        <div
          className="absolute left-[10px] top-2 bottom-2 w-px bg-border/70"
          aria-hidden
        />
        <motion.div
          aria-hidden
          className="absolute left-[10px] top-2 w-px bg-primary/60 origin-top"
          initial={{ height: 0 }}
          animate={{ height: Math.max(dotY - 4, 0) }}
          transition={{ duration: 0.55, ease: EASE }}
        />

        <motion.div
          aria-hidden
          className="absolute left-[6px] size-[9px] rounded-full bg-primary shadow-[0_0_0_4px_var(--background)]"
          initial={false}
          animate={{ y: dotY - 4 }}
          transition={{ type: "spring", stiffness: 260, damping: 30, mass: 0.6 }}
        >
          {running && !streamingText && (
            <motion.span
              className="absolute inset-0 rounded-full bg-primary/50"
              animate={{ scale: [1, 2.2, 1], opacity: [0.5, 0, 0.5] }}
              transition={{ duration: 1.6, repeat: Infinity, ease: "easeOut" }}
            />
          )}
        </motion.div>

        <ol className="space-y-0 min-w-0">
          {steps.map((s, i) => {
            const done = i < activeIndex;
            const active = i === activeIndex && running && !streamingText;

            // Tool call steps render as a self-contained card
            if (s.kind === "tool_call") {
              return (
                <motion.li
                  key={s.id + i}
                  ref={(el) => { rowRefs.current[i] = el; }}
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4, ease: EASE, delay: i * 0.03 }}
                  className="min-w-0 overflow-hidden"
                >
                  <ToolCallCard step={s as ToolCallStep} />
                </motion.li>
              );
            }

            const extra = expandable?.[s.id];
            const open = !!extra && isOpen(i, s.id);
            return (
              <motion.li
                key={s.id + i}
                ref={(el) => {
                  rowRefs.current[i] = el;
                }}
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.4, ease: EASE, delay: i * 0.03 }}
                className="flex flex-col"
              >
                <div className="flex items-baseline gap-2 leading-[28px]">
                  <button
                    type="button"
                    onClick={extra ? () => toggle(i, s.id) : undefined}
                    disabled={!extra}
                    className={`flex items-baseline gap-2 text-left ${
                      extra ? "cursor-pointer hover:text-primary transition-colors" : "cursor-default"
                    }`}
                  >
                    <span
                      className={
                        active
                          ? "text-foreground font-medium tracking-tight"
                          : done
                            ? "text-foreground/85 tracking-tight"
                            : "text-muted-foreground tracking-tight"
                      }
                    >
                      {active ? (
                        <span className="shimmer-text font-medium">{s.label}</span>
                      ) : (
                        s.label
                      )}
                    </span>
                    {(s as AgentStepData).detail && (
                      <span className="text-muted-foreground text-[13px] truncate">
                        {(s as AgentStepData).detail}
                      </span>
                    )}
                    {extra && (
                      <motion.span
                        animate={{ rotate: open ? 0 : -90 }}
                        transition={{ duration: 0.25, ease: EASE }}
                        className="text-muted-foreground self-center ml-0.5"
                      >
                        <ChevronDown className="size-3.5" />
                      </motion.span>
                    )}
                  </button>
                </div>
                {extra && (
                  <AnimatePresence initial={false}>
                    {open && (
                      <motion.div
                        key="extra"
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.35, ease: EASE }}
                        className="overflow-hidden"
                      >
                        <div className="pt-2 pb-3">{extra}</div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                )}
              </motion.li>
            );
          })}
        </ol>
      </div>
    </div>
  );
}
