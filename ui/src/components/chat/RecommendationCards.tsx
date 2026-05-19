import { motion } from "framer-motion";
import type { Recommendation } from "@/lib/agent";
import { Star } from "lucide-react";

export function RecommendationCards({ items }: { items: Recommendation[] }) {
  return (
    <div className="grid gap-2.5 mt-3">
      {items.map((r, i) => (
        <motion.article
          key={r.item.id}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35, delay: i * 0.07, ease: [0.22, 1, 0.36, 1] }}
          className="group rounded-2xl border border-border/70 bg-card hover:border-primary/40 hover:shadow-[0_8px_28px_-18px_rgba(217,119,87,0.35)] transition-all p-4"
        >
          <div className="flex items-start gap-3">
            <div className="grid place-items-center shrink-0 size-11 rounded-xl bg-sand text-2xl">
              {r.item.emoji}
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-baseline gap-2 flex-wrap">
                <h3 className="font-serif text-base font-semibold text-foreground truncate">
                  {r.item.title}
                </h3>
                <span className="text-xs text-muted-foreground">{r.item.subtitle}</span>
                {r.item.price && (
                  <span className="text-xs text-ink-soft font-medium">· {r.item.price}</span>
                )}
              </div>
              <p className="text-sm text-ink-soft mt-1 leading-relaxed">{r.item.blurb}</p>

              <ul className="mt-2 space-y-0.5">
                {r.reasons.map((why, j) => (
                  <li key={j} className="text-xs text-muted-foreground flex gap-1.5">
                    <span className="text-primary mt-0.5">›</span>
                    <span>{why}</span>
                  </li>
                ))}
              </ul>

              <div className="mt-3 flex items-center gap-3">
                <div className="flex items-center gap-1 text-xs">
                  <Star className="size-3 fill-primary text-primary" />
                  <span className="font-semibold text-foreground">{r.item.avgRating.toFixed(1)}</span>
                  <span className="text-muted-foreground">({r.item.reviews.toLocaleString()})</span>
                </div>
                <div className="flex-1 flex items-center gap-2">
                  <div className="flex-1 h-1.5 rounded-full bg-sand-deep overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.round(r.match * 100)}%` }}
                      transition={{ duration: 0.9, delay: 0.2 + i * 0.07, ease: "easeOut" }}
                      className="h-full bg-gradient-to-r from-primary/80 to-primary rounded-full"
                    />
                  </div>
                  <span className="text-xs font-semibold text-primary tabular-nums">
                    {Math.round(r.match * 100)}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        </motion.article>
      ))}
    </div>
  );
}
