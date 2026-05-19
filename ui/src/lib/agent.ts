import { Persona } from "./mock/personas";
import { Item, itemsByDataset, allItems } from "./mock/items";

export type Dataset = Item["dataset"];

export type AgentStep =
  | { kind: "step"; id: string; label: string; detail?: string; data?: unknown }
  | { kind: "token"; text: string }
  | { kind: "final"; payload: AgentResult };

export type Recommendation = {
  item: Item;
  match: number; // 0-1
  reasons: string[];
};

export type ReviewResult = {
  item: Item;
  rating: number; // 1-5 (allow .5)
  text: string;
  toneTags: string[];
  ratingDelta: number; // vs persona avg
};

export type AgentResult =
  | { type: "recommend"; persona: Persona; query: string; dataset: Dataset; items: Recommendation[]; intro: string }
  | { type: "review"; persona: Persona; result: ReviewResult; intro: string }
  | { type: "chat"; text: string };

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export type Intent = "recommend" | "review" | "chat";

export function classifyIntent(text: string): Intent {
  const t = text.toLowerCase();
  if (/\b(recommend|suggest|what should i|find me|pick|propose|any good)\b/.test(t)) return "recommend";
  if (/\b(review|rate|rating|how would .* feel|write a .*review|stars?)\b/.test(t)) return "review";
  return "chat";
}

export function detectDataset(text: string, persona: Persona): Dataset {
  const t = text.toLowerCase();
  if (/\b(book|read|novel|author|literature|chapter)\b/.test(t)) return "goodreads";
  if (/\b(restaurant|food|eat|jollof|suya|cafe|coffee|bar|brunch|dinner|lunch)\b/.test(t)) return "yelp";
  if (/\b(buy|product|gadget|headphone|kindle|mouse|charger|amazon)\b/.test(t)) return "amazon";
  // fall back to persona's first domain
  const first = persona.domains[0];
  return first === "books" ? "goodreads" : first === "shopping" ? "amazon" : "yelp";
}

function findItem(query: string): Item | null {
  const t = query.toLowerCase();
  let best: { item: Item; score: number } | null = null;
  for (const it of allItems) {
    const hay = `${it.title} ${it.subtitle} ${it.tags.join(" ")}`.toLowerCase();
    let s = 0;
    for (const word of t.split(/\W+/).filter((w) => w.length > 2)) {
      if (hay.includes(word)) s += 1;
    }
    if (!best || s > best.score) best = { item: it, score: s };
  }
  return best && best.score > 0 ? best.item : null;
}

function scoreFor(persona: Persona, item: Item): { match: number; reasons: string[] } {
  const reasons: string[] = [];
  let m = 0.55 + (item.avgRating - 3.5) * 0.06;
  const tagText = item.tags.join(" ").toLowerCase();
  for (const trait of persona.traits) {
    const t = trait.toLowerCase();
    if (
      (t.includes("foodie") && item.dataset === "yelp") ||
      (t.includes("literary") && item.dataset === "goodreads") ||
      (t.includes("tech") && item.dataset === "amazon")
    ) {
      m += 0.08;
      reasons.push(`Aligns with your ${trait} streak`);
    }
    if (t.includes("value") && (item.price === "₦" || item.price === "$" || /value/.test(tagText))) {
      m += 0.07;
      reasons.push("Strong value signal — fits your price sensitivity");
    }
    if (t.includes("warm") && /warm|cozy|outdoor|garden|live/.test(tagText)) {
      m += 0.05;
      reasons.push("Ambience tracks with reviews you've left fondly before");
    }
  }
  if (item.dataset === "yelp" && persona.city === "Lagos" && /lagos|lekki|yaba|ikoyi|vi|surulere/i.test(item.subtitle)) {
    m += 0.06;
    reasons.push(`In your home city, ${persona.city}`);
  }
  if (reasons.length === 0) reasons.push("Broadly matches your historical taste profile");
  m = Math.max(0.42, Math.min(0.98, m + (Math.random() * 0.04 - 0.02)));
  return { match: m, reasons: reasons.slice(0, 3) };
}

const ICE = (s: string) => s; // identity, kept for future Nigerianisation

function reviewText(persona: Persona, item: Item, rating: number): { text: string; toneTags: string[] } {
  const toneTags: string[] = [];
  const friendly = persona.tone.includes("warm") || persona.tone.includes("playful");
  const dry = persona.tone.includes("dry") || persona.tone.includes("analytical");
  const punchy = persona.tone.includes("concise") || persona.tone.includes("energetic");
  if (friendly) toneTags.push("warm");
  if (dry) toneTags.push("measured");
  if (punchy) toneTags.push("punchy");
  toneTags.push(persona.city === "Lagos" || persona.city === "Ibadan" ? "Nigerian-English" : "neutral-English");
  if (rating >= 4.5) toneTags.push("enthused");
  else if (rating <= 2.5) toneTags.push("disappointed");
  else toneTags.push("balanced");

  const opener =
    rating >= 4.5
      ? friendly
        ? `Okay, ${item.title} did not come to play.`
        : punchy
        ? `${item.title}? Underrated.`
        : `A surprisingly assured outing from ${item.title}.`
      : rating >= 3.5
      ? friendly
        ? `${item.title} is honestly solid — not perfect, but worth your time.`
        : `${item.title} works on most counts, with the usual caveats.`
      : rating >= 2.5
      ? `${item.title} is a mixed bag for me.`
      : `I wanted to love ${item.title}. I did not.`;

  const middle =
    item.dataset === "yelp"
      ? friendly
        ? ` The ${item.tags[0]} hit the spot — generous portions, and the staff were sweet.`
        : ` The ${item.tags[0]} was competent; service had moments of charm and moments of distraction.`
      : item.dataset === "goodreads"
      ? dry
        ? ` Adichie-adjacent in places, but the prose carries its own weight; the middle act drags briefly before recovering.`
        : ` The pacing finds its rhythm by the second act, and I underlined more sentences than I'd planned.`
      : punchy
      ? ` Build feels premium for the price. Battery is the headline. Setup took two minutes.`
      : ` Build is sturdier than I expected at this price point, and the everyday use cases hold up to scrutiny.`;

  const closer =
    rating >= 4
      ? friendly
        ? ` I'd come back. I'll bring people.`
        : ` Recommended, with mild reservations noted.`
      : rating >= 3
      ? ` Worth a try if you're in the area, manage expectations.`
      : ` Hard pass at this price; consider alternatives.`;

  return { text: ICE(opener + middle + closer), toneTags };
}

function chatReply(text: string): string {
  const t = text.toLowerCase();
  if (/help|what can/.test(t)) {
    return "I can do two things really well right now: **simulate a review** for any item the way your chosen persona would write it, or **recommend** items from Yelp, Amazon, or Goodreads tuned to that persona. Try one of the suggestions below, or just tell me what you want.";
  }
  if (/who are you|what are you/.test(t)) {
    return "I'm **Kora** — a persona-aware agent built for the DSN × BCT LLM Agent Challenge. Pick a persona on the left, then ask me to recommend or review.";
  }
  return "Tell me what you'd like — for example, *recommend a quiet spot in Lagos for a first date*, or *write a 4-star review of Half of a Yellow Sun in this persona's voice*.";
}

export async function* runAgent(opts: {
  persona: Persona;
  query: string;
  dataset?: Dataset;
}): AsyncGenerator<AgentStep, void, unknown> {
  const { persona, query } = opts;
  const intent = classifyIntent(query);

  if (intent === "chat") {
    await sleep(300);
    const reply = chatReply(query);
    for (const word of reply.split(/(\s+)/)) {
      yield { kind: "token", text: word };
      await sleep(18);
    }
    yield { kind: "final", payload: { type: "chat", text: reply } };
    return;
  }

  yield { kind: "step", id: "persona", label: "Reading persona", detail: `${persona.name} · ${persona.city}`, data: persona.traits };
  await sleep(450);

  const dataset = opts.dataset ?? detectDataset(query, persona);

  yield { kind: "step", id: "retrieve", label: `Retrieving from ${dataset}`, detail: "candidate set k=8" };
  await sleep(550);
  const candidates = itemsByDataset(dataset);

  yield { kind: "step", id: "score", label: "Scoring & reasoning", detail: "weighting traits × item signals", data: candidates.length };
  await sleep(600);

  if (intent === "review") {
    const target = findItem(query) ?? candidates[0];
    // base rating around persona avg, lifted by item quality
    const base = persona.avgRating + (target.avgRating - 3.8) * 0.6;
    const noise = (Math.random() - 0.5) * 0.4;
    const rating = Math.max(1, Math.min(5, Math.round((base + noise) * 2) / 2));
    const { text, toneTags } = reviewText(persona, target, rating);
    yield { kind: "step", id: "compose", label: "Composing review", detail: `${rating.toFixed(1)}★ in ${persona.name}'s voice` };
    await sleep(450);

    const intro = `Here's how **${persona.name}** would likely review **${target.title}**:`;
    for (const word of intro.split(/(\s+)/)) {
      yield { kind: "token", text: word };
      await sleep(14);
    }
    const result: ReviewResult = {
      item: target,
      rating,
      text,
      toneTags,
      ratingDelta: rating - persona.avgRating,
    };
    yield { kind: "final", payload: { type: "review", persona, result, intro } };
    return;
  }

  // recommend
  const scored = candidates
    .map((item) => ({ item, ...scoreFor(persona, item) }))
    .sort((a, b) => b.match - a.match)
    .slice(0, 5);

  yield { kind: "step", id: "compose", label: "Ranking & composing", detail: `top ${scored.length} of ${candidates.length}` };
  await sleep(420);

  const intro = `Based on **${persona.name}**'s history, here are ${scored.length} picks from **${dataset}** — ordered by match:`;
  for (const word of intro.split(/(\s+)/)) {
    yield { kind: "token", text: word };
    await sleep(12);
  }

  yield {
    kind: "final",
    payload: { type: "recommend", persona, query, dataset, items: scored, intro },
  };
}
