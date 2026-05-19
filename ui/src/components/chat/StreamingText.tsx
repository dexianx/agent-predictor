import { motion } from "framer-motion";
import { memo } from "react";

type Props = { text: string; className?: string };

export const StreamingText = memo(function StreamingText({ text, className }: Props) {
  const tokens = text.split(/(\s+)/);
  return (
    <span className={className}>
      {tokens.map((tok, i) =>
        /^\s+$/.test(tok) ? (
          <span key={i}>{tok}</span>
        ) : (
          <motion.span
            key={i}
            initial={{ opacity: 0, y: 3, filter: "blur(2px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            transition={{ duration: 0.32, ease: [0.22, 1, 0.36, 1] }}
            style={{ display: "inline-block" }}
          >
            {tok}
          </motion.span>
        ),
      )}
    </span>
  );
});
