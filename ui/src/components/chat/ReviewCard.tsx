import { motion } from "framer-motion";
import type { ReviewResult } from "@/lib/agent";
import { Star } from "lucide-react";

export function ReviewCard({ review }: { review: ReviewResult }) {
  const full = Math.floor(review.rating);
  const half = review.rating - full >= 0.5;
  const empty = 5 - full - (half ? 1 : 0);

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
      className="mt-3 rounded-2xl border border-border/70 bg-card p-5 shadow-[0_10px_30px_-22px_rgba(217,119,87,0.35)]"
    >
      <div className="flex items-start gap-3">
        <div className="grid place-items-center shrink-0 size-12 rounded-xl bg-sand text-2xl">
          {review.item.emoji}
        </div>
        <div className="min-w-0 flex-1">
          <h3 className="font-serif text-lg font-normal leading-tight">{review.item.title}</h3>
          <p className="text-xs text-muted-foreground">{review.item.subtitle}</p>
        </div>
      </div>

      <div className="mt-4 flex items-center gap-2">
        <div className="flex items-center gap-0.5">
          {[...Array(full)].map((_, i) => (
            <motion.span
              key={`f${i}`}
              initial={{ scale: 0, rotate: -45, opacity: 0 }}
              animate={{ scale: 1, rotate: 0, opacity: 1 }}
              transition={{ duration: 0.35, delay: 0.1 + i * 0.07, type: "spring", stiffness: 280 }}
            >
              <Star className="size-5 fill-primary text-primary" />
            </motion.span>
          ))}
          {half && (
            <motion.span
              initial={{ scale: 0, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.35, delay: 0.1 + full * 0.07, type: "spring" }}
              className="relative inline-flex size-5"
            >
              {/* empty star underneath */}
              <Star className="absolute inset-0 size-5 text-sand-deep" />
              {/* filled star clipped to left half */}
              <span className="absolute inset-0 overflow-hidden" style={{ width: "50%" }}>
                <Star className="size-5 fill-primary text-primary" />
              </span>
            </motion.span>
          )}
          {[...Array(empty)].map((_, i) => (
            <Star key={`e${i}`} className="size-5 text-sand-deep" />
          ))}
        </div>
        <span className="font-serif text-xl font-semibold text-foreground tabular-nums">
          {review.rating.toFixed(1)}
        </span>
        <span
          className={`ml-1 text-xs px-1.5 py-0.5 rounded-md font-medium ${
            review.ratingDelta >= 0
              ? "bg-primary/10 text-primary"
              : "bg-secondary text-secondary-foreground"
          }`}
        >
          {review.ratingDelta >= 0 ? "+" : ""}
          {review.ratingDelta.toFixed(1)} vs avg
        </span>
      </div>

      <blockquote className="mt-4 pl-3 border-l-2 border-primary/40 text-foreground leading-relaxed">
        {review.text}
      </blockquote>

      <div className="mt-4 flex flex-wrap gap-1.5">
        {review.toneTags.map((t) => (
          <motion.span
            key={t}
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 + Math.random() * 0.2 }}
            className="text-[11px] uppercase tracking-wide font-medium px-2 py-0.5 rounded-full bg-sand text-cocoa"
          >
            {t}
          </motion.span>
        ))}
      </div>
    </motion.div>
  );
}
