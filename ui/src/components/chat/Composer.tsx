import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowUp, Square, ChevronDown } from "lucide-react";
import type { PersonaDetail } from "@/lib/types";

export type ProductFields = {
  item_name: string;
  item_description: string;
  price_ngn: number;
  item_brand?: string;
  item_category?: string;
  item_features?: string;
  item_details?: string;
};

type Props = {
  onSubmit: (fields: ProductFields) => void;
  onStop?: () => void;
  busy: boolean;
  persona: PersonaDetail | null;
  threadKey: string;
};

export function Composer({ onSubmit, onStop, busy, persona, threadKey }: Props) {
  const nameRef     = useRef<HTMLInputElement>(null);
  const descRef     = useRef<HTMLTextAreaElement>(null);
  const priceRef    = useRef<HTMLInputElement>(null);
  const brandRef    = useRef<HTMLInputElement>(null);
  const categoryRef = useRef<HTMLInputElement>(null);
  const featuresRef = useRef<HTMLTextAreaElement>(null);
  const detailsRef  = useRef<HTMLTextAreaElement>(null);

  const [canSend, setCanSend]   = useState(false);
  const [expanded, setExpanded] = useState(false);

  useEffect(() => { nameRef.current?.focus(); }, [threadKey]);
  useEffect(() => { if (!busy) nameRef.current?.focus(); }, [busy]);

  const checkCanSend = () => {
    const hasName  = !!(nameRef.current?.value.trim());
    const hasDesc  = !!(descRef.current?.value.trim());
    const hasPrice = !!(priceRef.current?.value.trim());
    setCanSend(hasName && hasDesc && hasPrice);
  };

  const send = () => {
    const item_name        = nameRef.current?.value.trim()     ?? "";
    const item_description = descRef.current?.value.trim()     ?? "";
    const priceRaw         = priceRef.current?.value.trim();
    if (!item_name || !item_description || !priceRaw) return;
    const price_ngn = parseFloat(priceRaw);
    if (isNaN(price_ngn)) return;

    const item_brand    = brandRef.current?.value.trim()    || undefined;
    const item_category = categoryRef.current?.value.trim() || undefined;
    const item_features = featuresRef.current?.value.trim() || undefined;
    const item_details  = detailsRef.current?.value.trim()  || undefined;

    onSubmit({ item_name, item_description, price_ngn, item_brand, item_category, item_features, item_details });

    nameRef.current!.value = "";
    if (descRef.current) { descRef.current.value = ""; descRef.current.style.height = "auto"; }
    priceRef.current!.value = "";
    if (brandRef.current)    brandRef.current.value = "";
    if (categoryRef.current) categoryRef.current.value = "";
    if (featuresRef.current) { featuresRef.current.value = ""; featuresRef.current.style.height = "auto"; }
    if (detailsRef.current)  { detailsRef.current.value  = ""; detailsRef.current.style.height  = "auto"; }
    setCanSend(false);
  };

  const autosize = (el: HTMLTextAreaElement | null, maxPx = 120) => {
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, maxPx) + "px";
  };

  const inputClass =
    "w-full bg-transparent text-[15px] md:text-[15.5px] leading-[1.55] outline-none placeholder:text-muted-foreground/60";

  return (
    <div className="px-3 md:px-4 pb-[max(16px,env(safe-area-inset-bottom,16px))] md:pb-5 pt-2 shrink-0">
      <form
        onSubmit={(e) => { e.preventDefault(); send(); }}
        className="md:mx-auto md:max-w-3xl rounded-[18px] md:rounded-[20px] border border-border bg-card transition-[border-color,box-shadow] duration-300 focus-within:border-foreground/25 focus-within:shadow-[0_2px_24px_-12px_rgba(20,20,19,0.18)]"
      >
        <div className="px-4 md:px-5 pt-3.5 md:pt-4 pb-2 space-y-2">
          {/* Required: name */}
          <input
            ref={nameRef}
            type="text"
            placeholder="Product name…"
            onInput={checkCanSend}
            onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); descRef.current?.focus(); } }}
            className={inputClass}
            disabled={busy}
          />
          <div className="h-px bg-border/40" />

          {/* Required: description */}
          <textarea
            ref={descRef}
            rows={1}
            placeholder="Product details…"
            onInput={() => { autosize(descRef.current); checkCanSend(); }}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
            className={`${inputClass} resize-none max-h-[120px]`}
            disabled={busy}
          />
          <div className="h-px bg-border/40" />

          {/* Required: price */}
          <input
            ref={priceRef}
            type="number"
            min="0"
            step="any"
            placeholder={`Price in ${persona?.locale === "US" ? "$" : "₦"}…`}
            onInput={checkCanSend}
            className={`${inputClass} [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none`}
            disabled={busy}
          />

          {/* Optional section toggle */}
          <button
            type="button"
            onClick={() => setExpanded((v) => !v)}
            disabled={busy}
            className="flex items-center gap-1 text-[11.5px] text-muted-foreground/60 hover:text-muted-foreground transition-colors mt-0.5 select-none"
          >
            <motion.span
              animate={{ rotate: expanded ? 180 : 0 }}
              transition={{ duration: 0.2 }}
              className="inline-flex"
            >
              <ChevronDown className="size-3" />
            </motion.span>
            {expanded ? "Hide details" : "More details"}
          </button>

          {/* Optional: brand + category + features + details */}
          <AnimatePresence initial={false}>
            {expanded && (
              <motion.div
                key="extra"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.22, ease: [0.22, 1, 0.36, 1] }}
                className="overflow-hidden"
              >
                <div className="space-y-2 pt-1">
                  <div className="h-px bg-border/40" />

                  {/* Brand + Category side-by-side */}
                  <div className="grid grid-cols-2 gap-3">
                    <input
                      ref={brandRef}
                      type="text"
                      placeholder="Brand (e.g. Anker)…"
                      className={`${inputClass} text-[13.5px]`}
                      disabled={busy}
                    />
                    <input
                      ref={categoryRef}
                      type="text"
                      placeholder="Category (e.g. Electronics)…"
                      className={`${inputClass} text-[13.5px]`}
                      disabled={busy}
                    />
                  </div>
                  <div className="h-px bg-border/40" />

                  {/* Features */}
                  <textarea
                    ref={featuresRef}
                    rows={2}
                    placeholder="Features — one per line…"
                    onInput={() => autosize(featuresRef.current, 100)}
                    className={`${inputClass} text-[13.5px] resize-none max-h-[100px]`}
                    disabled={busy}
                  />
                  <div className="h-px bg-border/40" />

                  {/* Technical details */}
                  <textarea
                    ref={detailsRef}
                    rows={2}
                    placeholder="Technical details / specs…"
                    onInput={() => autosize(detailsRef.current, 100)}
                    className={`${inputClass} text-[13.5px] resize-none max-h-[100px]`}
                    disabled={busy}
                  />
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div className="px-3 pb-2.5 md:pb-3 pt-1 flex items-center justify-between gap-2">
          <div className="pl-1 md:pl-2 text-[11.5px] md:text-[12px] text-muted-foreground/80 truncate min-w-0">
            Thinking as{" "}
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
