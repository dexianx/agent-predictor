import type { PersonaDetail } from "./types";

export function getPersonaDisplayName(p: PersonaDetail): string {
  const mean = p.rating_mean ?? 3.5;
  const tendency =
    mean < 3.3 ? "Tough Critic" :
    mean < 3.8 ? "Measured Reviewer" :
    mean < 4.3 ? "Balanced Reviewer" : "Generous Reviewer";

  const cat =
    p.top_categories[0]
    ?? (p.platform?.replace("amazon_us", "Amazon").replace("jumia_ng", "Jumia"))
    ?? "Shopper";

  const locale = p.locale ?? "??";
  return `${tendency} · ${cat} · ${locale}`;
}

export function getPersonaShortId(p: PersonaDetail): string {
  return p.user_id.slice(0, 12) + "…";
}
