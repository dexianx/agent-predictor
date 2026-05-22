import { motion } from "framer-motion";
import { Star } from "lucide-react";
import type { RecommendItem } from "@/lib/types";

export function RecommendResultCards({ items }: { items: RecommendItem[] }) {
  return (
    <div className="grid gap-2.5 mt-3">
      {items.map((item, i) => {
        const price = item.price_usd
          ? `$${item.price_usd.toFixed(0)}`
          : item.price_ngn
          ? `₦${item.price_ngn.toLocaleString()}`
          : null;

        return (
          <motion.article
            key={item.item_id}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35, delay: i * 0.07, ease: [0.22, 1, 0.36, 1] }}
            className="group rounded-2xl border border-border/70 bg-card hover:border-primary/40 hover:shadow-[0_8px_28px_-18px_rgba(217,119,87,0.35)] transition-all p-4"
          >
            <div className="flex items-start gap-3">
              {/* Rank badge */}
              <div className="grid place-items-center shrink-0 size-11 rounded-xl bg-sand text-lg font-serif font-semibold text-primary">
                {i + 1}
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-baseline gap-2 flex-wrap">
                  <h3 className="font-serif text-base font-semibold text-foreground line-clamp-2">
                    {item.title}
                  </h3>
                  {item.category && (
                    <span className="text-xs text-muted-foreground shrink-0">{item.category}</span>
                  )}
                  {price && (
                    <span className="text-xs text-ink-soft font-medium shrink-0">· {price}</span>
                  )}
                </div>

                <ul className="mt-2 space-y-0.5">
                  {item.reasons.map((why, j) => (
                    <li key={j} className="text-xs text-muted-foreground flex gap-1.5">
                      <span className="text-primary mt-0.5 shrink-0">›</span>
                      <span>{why}</span>
                    </li>
                  ))}
                </ul>

                <div className="mt-3 flex items-center gap-3">
                  <div className="flex items-center gap-1 text-xs">
                    {item.avg_rating != null ? (
                      <>
                        <Star className="size-3 fill-primary text-primary" />
                        <span className="font-semibold text-foreground">{item.avg_rating.toFixed(1)}</span>
                        {item.rating_count != null && (
                          <span className="text-muted-foreground">({item.rating_count.toLocaleString()})</span>
                        )}
                      </>
                    ) : (
                      <span className="text-muted-foreground italic">web result</span>
                    )}
                  </div>
                  <div className="flex-1 flex items-center gap-2">
                    <div className="flex-1 h-1.5 rounded-full bg-sand-deep overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${Math.round(item.match_score * 100)}%` }}
                        transition={{ duration: 0.9, delay: 0.2 + i * 0.07, ease: "easeOut" }}
                        className="h-full bg-gradient-to-r from-primary/80 to-primary rounded-full"
                      />
                    </div>
                    <span className="text-xs font-semibold text-primary tabular-nums">
                      {Math.round(item.match_score * 100)}%
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </motion.article>
        );
      })}
    </div>
  );
}
