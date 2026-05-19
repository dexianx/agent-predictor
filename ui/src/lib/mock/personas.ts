export type Persona = {
  id: string;
  name: string;
  age: number;
  city: string;
  bio: string;
  traits: string[];
  tone: string;
  avgRating: number; // historical average
  domains: ("food" | "books" | "shopping")[];
  emoji: string;
};

export const personas: Persona[] = [
  {
    id: "adaeze",
    name: "Adaeze",
    age: 28,
    city: "Lagos",
    bio: "Product designer who weekends as a jollof critic. Loves smoky party rice and quiet corners.",
    traits: ["foodie", "warm critic", "values ambience", "Yoruba-Igbo bilingual"],
    tone: "warm, observational, light Nigerian-English",
    avgRating: 4.1,
    domains: ["food", "shopping"],
    emoji: "🍲",
  },
  {
    id: "tunde",
    name: "Tunde",
    age: 34,
    city: "Ibadan",
    bio: "Lecturer in comparative literature. Reads African fiction with a translator's eye.",
    traits: ["literary", "skeptical", "long-form reviewer", "favours sparse prose"],
    tone: "measured, analytical, occasional dry humour",
    avgRating: 3.7,
    domains: ["books"],
    emoji: "📚",
  },
  {
    id: "zainab",
    name: "Zainab",
    age: 24,
    city: "Abuja",
    bio: "Med-student-turned-tech-Twitter. Buys gadgets fast, reviews them faster.",
    traits: ["tech-forward", "value-conscious", "punchy", "emoji-friendly"],
    tone: "energetic, concise, lightly playful",
    avgRating: 4.0,
    domains: ["shopping"],
    emoji: "⚡",
  },
  {
    id: "kelechi",
    name: "Kelechi",
    age: 41,
    city: "Port Harcourt",
    bio: "Engineer and dad of two. Pragmatic, never stars anything five unless it changed his life.",
    traits: ["pragmatic", "value first", "skeptical of hype", "family lens"],
    tone: "plain-spoken, direct, occasionally fatherly",
    avgRating: 3.4,
    domains: ["food", "shopping", "books"],
    emoji: "🛠️",
  },
];

export const personaById = (id: string) =>
  personas.find((p) => p.id === id) ?? personas[0];
