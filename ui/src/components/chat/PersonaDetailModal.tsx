import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { getPersonaDetail } from "@/lib/api";
import type { PersonaDetail } from "@/lib/types";

type Props = {
  userId: string;
  open: boolean;
  onClose: () => void;
};

function pidginLabel(freq: number | null) {
  if (!freq) return "None";
  if (freq > 0.05) return "Frequent";
  if (freq > 0.01) return "Occasional";
  return "None";
}

function tendencyLabel(mean: number | null) {
  if (!mean) return "—";
  if (mean < 3.3) return "Tough critic";
  if (mean < 3.8) return "Slightly harsh";
  if (mean < 4.3) return "Balanced";
  return "Generous";
}

function Bar({ value }: { value: number }) {
  const pct = Math.round(Math.min(1, Math.max(0, value)) * 100);
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 rounded-full bg-border/60 overflow-hidden">
        <motion.div
          className="h-full rounded-full bg-primary"
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.5, ease: "easeOut" }}
        />
      </div>
      <span className="text-[11px] tabular-nums text-muted-foreground w-7 text-right">{pct}%</span>
    </div>
  );
}

export function PersonaDetailModal({ userId, open, onClose }: Props) {
  const [persona, setPersona] = useState<PersonaDetail | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!open || !userId) return;
    setLoading(true);
    setPersona(null);
    getPersonaDetail(userId)
      .then(setPersona)
      .finally(() => setLoading(false));
  }, [userId, open]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-end md:items-center justify-end md:justify-end"
      onClick={onClose}
    >
      <div className="absolute inset-0 bg-black/30 backdrop-blur-sm" />
      <motion.div
        className="relative z-10 w-full md:w-[420px] h-[85vh] md:h-full bg-background border-l border-border shadow-2xl overflow-y-auto"
        initial={{ x: "100%" }}
        animate={{ x: 0 }}
        exit={{ x: "100%" }}
        transition={{ type: "spring", stiffness: 320, damping: 35 }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="sticky top-0 z-10 bg-background/95 backdrop-blur border-b border-border/60 px-5 py-4 flex items-center justify-between">
          <div>
            <div className="font-mono text-sm font-medium truncate max-w-[260px]">{userId}</div>
            <div className="text-[11px] text-muted-foreground mt-0.5">Persona profile</div>
          </div>
          <button
            onClick={onClose}
            className="size-8 rounded-lg grid place-items-center hover:bg-sand text-muted-foreground transition-colors"
          >
            ✕
          </button>
        </div>

        {loading && (
          <div className="p-5 space-y-4 animate-pulse">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="h-4 rounded bg-border/40" style={{ width: `${60 + (i % 3) * 15}%` }} />
            ))}
          </div>
        )}

        {!loading && persona && (
          <div className="p-5 space-y-6">
            {/* Identity */}
            <section>
              <h3 className="text-[10px] uppercase tracking-widest text-muted-foreground mb-3">Identity</h3>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div><span className="text-muted-foreground text-xs">Locale</span><div className="font-medium">{persona.locale ?? "—"}</div></div>
                <div><span className="text-muted-foreground text-xs">Platform</span><div className="font-medium">{persona.platform ?? "—"}</div></div>
                <div><span className="text-muted-foreground text-xs">Reviews</span><div className="font-medium">{persona.review_count?.toLocaleString() ?? "—"}</div></div>
                <div><span className="text-muted-foreground text-xs">Tendency</span><div className="font-medium">{tendencyLabel(persona.rating_mean)}</div></div>
              </div>
            </section>

            {/* Rating profile */}
            <section>
              <h3 className="text-[10px] uppercase tracking-widest text-muted-foreground mb-3">Rating Profile</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between"><span className="text-muted-foreground">Mean</span><span className="tabular-nums">{persona.rating_mean?.toFixed(2) ?? "—"}★</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Std dev</span><span className="tabular-nums">{persona.rating_std?.toFixed(2) ?? "—"}</span></div>
                <div className="flex justify-between"><span className="text-muted-foreground">Skew</span><span className="tabular-nums">{persona.rating_skew?.toFixed(2) ?? "—"}</span></div>
              </div>
            </section>

            {/* Writing style */}
            <section>
              <h3 className="text-[10px] uppercase tracking-widest text-muted-foreground mb-3">Writing Style</h3>
              <div className="space-y-3 text-sm">
                <div>
                  <div className="flex justify-between mb-1"><span className="text-muted-foreground">Vocabulary richness</span><span className="tabular-nums text-xs">{((persona.vocab_richness ?? 0) * 100).toFixed(0)}%</span></div>
                  <Bar value={persona.vocab_richness ?? 0} />
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Avg sentence length</span>
                  <span className="tabular-nums">{persona.avg_sent_len?.toFixed(1) ?? "—"} words</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Pidgin frequency</span>
                  <span className={`text-[11px] px-2 py-0.5 rounded-full font-medium ${
                    pidginLabel(persona.pidgin_freq) === "Frequent"
                      ? "bg-primary/15 text-primary"
                      : pidginLabel(persona.pidgin_freq) === "Occasional"
                      ? "bg-amber-500/15 text-amber-600"
                      : "bg-border/50 text-muted-foreground"
                  }`}>{pidginLabel(persona.pidgin_freq)}</span>
                </div>
              </div>
            </section>

            {/* Cultural markers */}
            <section>
              <h3 className="text-[10px] uppercase tracking-widest text-muted-foreground mb-3">Cultural Markers</h3>
              <div className="flex flex-wrap gap-2">
                {(persona.naira_mention_rate ?? 0) > 0.01 && (
                  <span className="text-xs px-2.5 py-1 rounded-full bg-sand text-cocoa border border-border/50">₦ mentions</span>
                )}
                {(persona.authenticity_concern ?? 0) > 0.2 && (
                  <span className="text-xs px-2.5 py-1 rounded-full bg-sand text-cocoa border border-border/50">Authenticity concern</span>
                )}
                {(persona.price_sensitivity ?? 0) > 0.2 && (
                  <span className="text-xs px-2.5 py-1 rounded-full bg-sand text-cocoa border border-border/50">Price sensitive</span>
                )}
                {(persona.emoji_rate ?? 0) > 0.02 && (
                  <span className="text-xs px-2.5 py-1 rounded-full bg-sand text-cocoa border border-border/50">Emoji user</span>
                )}
                {(persona.exclamation_rate ?? 0) > 0.03 && (
                  <span className="text-xs px-2.5 py-1 rounded-full bg-sand text-cocoa border border-border/50">Expressive</span>
                )}
              </div>
            </section>

            {/* Top categories */}
            {persona.top_categories.length > 0 && (
              <section>
                <h3 className="text-[10px] uppercase tracking-widest text-muted-foreground mb-3">Top Categories</h3>
                <div className="flex flex-wrap gap-1.5">
                  {persona.top_categories.map((c) => (
                    <span key={c} className="text-xs px-2 py-0.5 rounded-md bg-primary/8 text-primary font-medium">
                      {c}
                    </span>
                  ))}
                </div>
              </section>
            )}

            {/* Category rating bias */}
            {persona.cat_bias.length > 0 && (
              <section>
                <h3 className="text-[10px] uppercase tracking-widest text-muted-foreground mb-3">Category Rating Bias</h3>
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-left text-[11px] text-muted-foreground border-b border-border/50">
                      <th className="pb-1.5 font-medium">Category</th>
                      <th className="pb-1.5 font-medium text-right">Avg ★</th>
                      <th className="pb-1.5 font-medium text-right">Reviews</th>
                    </tr>
                  </thead>
                  <tbody>
                    {persona.cat_bias.slice(0, 10).map((b) => (
                      <tr key={b.category} className="border-b border-border/20">
                        <td className="py-1.5 text-foreground/80 truncate max-w-[160px]">{b.category}</td>
                        <td className="py-1.5 text-right tabular-nums">{b.mean.toFixed(2)}</td>
                        <td className="py-1.5 text-right tabular-nums text-muted-foreground">{b.count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </section>
            )}
          </div>
        )}
      </motion.div>
    </div>
  );
}
