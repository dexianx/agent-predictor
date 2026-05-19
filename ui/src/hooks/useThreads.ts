import { useEffect, useSyncExternalStore } from "react";
import { STORAGE_KEY, Thread, ChatMessage } from "@/lib/types";

const DEFAULT_PERSONA_ID = "";

const isClient = typeof window !== "undefined";

function read(): Thread[] {
  if (!isClient) return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as Thread[];
    if (!Array.isArray(parsed)) return [];
    return parsed;
  } catch {
    return [];
  }
}

const listeners = new Set<() => void>();
let cache: Thread[] | null = null;

function getSnapshot(): Thread[] {
  if (cache === null) cache = read();
  return cache;
}

function getServerSnapshot(): Thread[] {
  return [];
}

function write(next: Thread[]) {
  cache = next;
  if (isClient) localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  listeners.forEach((l) => l());
}

function subscribe(cb: () => void) {
  listeners.add(cb);
  const onStorage = (e: StorageEvent) => {
    if (e.key === STORAGE_KEY) {
      cache = read();
      listeners.forEach((l) => l());
    }
  };
  if (isClient) window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(cb);
    if (isClient) window.removeEventListener("storage", onStorage);
  };
}

export function useThreads() {
  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
}

export function useThread(id: string | undefined) {
  const all = useThreads();
  return id ? all.find((t) => t.id === id) : undefined;
}

function uid() {
  return Math.random().toString(36).slice(2, 10);
}

export function createThread(personaId?: string): Thread {
  const all = read();
  const t: Thread = {
    id: uid(),
    title: "New chat",
    personaId: personaId ?? all[0]?.personaId ?? DEFAULT_PERSONA_ID,
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: [],
  };
  write([t, ...all]);
  return t;
}

export function ensureFirstThread(): Thread {
  const all = read();
  if (all.length > 0) return all[0];
  return createThread();
}

export function deleteThread(id: string) {
  write(read().filter((t) => t.id !== id));
}

export function renameThread(id: string, title: string) {
  write(
    read().map((t) =>
      t.id === id ? { ...t, title, updatedAt: Date.now() } : t,
    ),
  );
}

export function setPersona(id: string, personaId: string) {
  write(
    read().map((t) =>
      t.id === id ? { ...t, personaId, updatedAt: Date.now() } : t,
    ),
  );
}

export function appendMessage(id: string, msg: ChatMessage) {
  write(
    read().map((t) =>
      t.id === id
        ? {
            ...t,
            updatedAt: Date.now(),
            title:
              t.messages.length === 0 && msg.role === "user"
                ? msg.text.slice(0, 48)
                : t.title,
            messages: [...t.messages, msg],
          }
        : t,
    ),
  );
}

export function updateMessage(
  id: string,
  msgId: string,
  patch: Partial<ChatMessage>,
) {
  write(
    read().map((t) =>
      t.id === id
        ? {
            ...t,
            updatedAt: Date.now(),
            messages: t.messages.map((m) =>
              m.id === msgId ? { ...m, ...patch } : m,
            ),
          }
        : t,
    ),
  );
}

export const newMessageId = uid;

// dev helper: noop on server, ensures hydration consistency
export function useHydrated() {
  // simple flag via effect
  if (!isClient) return false;
  return true;
}
