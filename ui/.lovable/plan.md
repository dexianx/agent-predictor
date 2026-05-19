# Plan: Kora — Persona-Aware Agent UI

A single chat agent that fulfills both DSN×BCT hackathon tasks:

- **Task A — Review Simulation**: given a persona + item, generate a star rating + written review.
- **Task B — Recommendation**: given a persona + intent, rank and recommend items with reasoning.

The agent auto-detects intent from the user's message and runs the matching agentic workflow, visibly. All data is mocked (Yelp / Amazon / Goodreads-style fixtures). Threaded chats are saved per-browser via localStorage.

## Design direction

- **Palette**: clean white base, warm terracotta/amber accent (`oklch(0.68 0.16 50)` primary), soft sand surfaces, deep cocoa ink for text. Subtle Nigerian-warm feel without being literal.
- **Type**: a distinctive serif for display (Fraunces) + Inter for UI. Avoids generic AI look.
- **Motion**: framer-motion throughout — staggered token-by-token text reveal for streamed responses, soft fade/slide for tool steps, spring transitions on thread switch, animated star rating fill.
- **Surfaces**: assistant messages flat on the page (no bubble); user messages in a warm tinted pill with high-contrast foreground.

## Information architecture

```
/                    Landing → auto-creates a thread, navigates to /c/:id
/c/:threadId         Chat workspace (sidebar + transcript + composer)
```

- **Sidebar**: thread list, "New chat", persona switcher (mock personas like *Adaeze, 28, Lagos foodie* / *Tunde, lit-fiction reader*), and a small "What can Kora do?" panel.
- **Empty state**: persona card + 6 suggestion chips split across the two tasks ("Recommend a jollof spot in Yaba", "Write a 4★ review for *Half of a Yellow Sun*", etc.).
- **Composer**: AI Elements PromptInput with persona pill, dataset selector (Yelp / Amazon / Goodreads), and a Send button; textarea autofocuses.

## Agent behavior (mocked, but feels real)

A lightweight client-side router classifies the message:

- contains "recommend / suggest / what should I…" → **Recommendation flow**
- contains "review / rate / how would I rate…" → **Review-simulation flow**
- otherwise → conversational reply with quick-action buttons to pick a flow

Each flow streams a sequence of visible **agent steps** (collapsed accordion, AI Elements `Tool` style):

1. *Reading persona* → shows persona traits chip cloud
2. *Retrieving candidates from {dataset}* → shows 5–8 mock items sliding in
3. *Scoring & reasoning* → shows NDCG-ish score bars animating
4. *Composing response* → final answer streams token-by-token

**Recommendation output**: animated ranked cards (cover, title, why-it-fits, match %, tags). Click a card → expands inline detail.

**Review output**: animated star rating fills 0→N, then the review text streams. Below: small panel showing tone tags (e.g., "warm, observational, Nigerian-English") and a predicted RMSE delta vs. persona's history — selling the "behavioural fidelity" angle from the rubric.

## Persistence (localStorage, threaded)

One idempotent bootstrap path reads `kora.threads.v1`. Schema:

```ts
type Thread = { id: string; title: string; personaId: string; updatedAt: number; messages: UIMessage[] }
```

- Visiting `/` selects most recent thread or creates one, then navigates to `/c/:id`.
- New chat → create record, navigate. Switching threads = navigation.
- Chat window keyed by `threadId` so messages never bleed.
- Delete button is a sibling of the row select button (no nested buttons).

## Mock data

`src/lib/mock/` with three small fixtures:

- `yelp.ts` (restaurants, mostly Lagos/Abuja flavored)
- `amazon.ts` (electronics + lifestyle)
- `goodreads.ts` (books incl. African literature)
- `personas.ts` (4 personas with history, preferences, tone)

A small `agent.ts` module exposes `runRecommend(persona, query)` and `runReview(persona, item)` returning an async generator of step events + a final payload — used by the chat UI to drive the streaming animation.

## Routes / files to create

- `src/routes/index.tsx` (replace placeholder; bootstraps a thread, redirects)
- `src/routes/c.$threadId.tsx` (chat workspace)
- `src/components/chat/{Sidebar,Transcript,Composer,AgentSteps,RecommendationCards,ReviewCard,PersonaPill,EmptyState}.tsx`
- `src/components/ai-elements/*` (installed via `bun x ai-elements@latest add conversation message prompt-input shimmer tool`)
- `src/hooks/useThreads.ts` (localStorage-backed)
- `src/lib/agent.ts`, `src/lib/mock/*`
- `src/styles.css` updated with terracotta palette + Fraunces/Inter `@font-face`/Google import

## Technical notes

- Pure frontend, no Lovable Cloud, no server functions — agent is mocked locally.
- Framer-motion for all animation (token reveal via splitting streamed text into spans with staggered `animate`).
- AI Elements primitives wrap our chat surfaces; tool accordions stay collapsed by default.
- Logo: a small generated terracotta "K" mark, not Sparkles.
- Strict adherence to semantic tokens in `styles.css`.

## Out of scope

- Real LLM calls, real datasets, auth, deployment-time env vars.
- Database persistence (chosen: localStorage).