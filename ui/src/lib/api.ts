import type { PersonaDetail } from "./types";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

async function apiFetch<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`API ${res.status}: ${path}`);
  return res.json() as Promise<T>;
}

export async function getFeaturedPersonas(): Promise<PersonaDetail[]> {
  return apiFetch<PersonaDetail[]>("/personas/featured");
}

export async function getPersonaDetail(userId: string): Promise<PersonaDetail> {
  return apiFetch<PersonaDetail>(`/personas/${encodeURIComponent(userId)}`);
}

export async function searchPersonas(
  query: string,
  locale?: string,
  limit = 25,
  offset = 0
): Promise<PersonaDetail[]> {
  const params = new URLSearchParams({ search: query, limit: String(limit), offset: String(offset) });
  if (locale) params.set("locale", locale);
  return apiFetch<PersonaDetail[]>(`/personas?${params.toString()}`);
}
