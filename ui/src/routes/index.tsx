import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useEffect } from "react";
import { ensureFirstThread } from "@/hooks/useThreads";

export const Route = createFileRoute("/")({
  component: Index,
  head: () => ({
    meta: [
      { title: "Kora — Persona-Aware Agent" },
      {
        name: "description",
        content:
          "Kora is a persona-aware LLM agent that simulates user reviews and delivers contextual recommendations across Yelp, Amazon, and Goodreads.",
      },
      { property: "og:title", content: "Kora — Persona-Aware Agent" },
      {
        property: "og:description",
        content:
          "Built for the DSN × BCT LLM Agent Challenge. Review simulation and recommendation in one warm, conversational surface.",
      },
    ],
  }),
});

function Index() {
  const navigate = useNavigate();
  useEffect(() => {
    const t = ensureFirstThread();
    navigate({ to: "/c/$threadId", params: { threadId: t.id }, replace: true });
  }, [navigate]);

  return (
    <div className="min-h-screen grid place-items-center bg-background">
      <div className="text-sm text-muted-foreground">Opening Kora…</div>
    </div>
  );
}
