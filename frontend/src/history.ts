// Recent questions, kept in this browser only. Storage can be blocked (private windows), so every
// access is wrapped and the page works without it.
import type { Mode } from "./App";

export interface HistoryItem {
  q: string;
  mode: Mode;
}

const KEY = "secqa-history";
const MAX = 15;

export function loadHistory(): HistoryItem[] {
  try {
    const items = JSON.parse(localStorage.getItem(KEY) ?? "[]");
    return Array.isArray(items) ? items.filter((i) => typeof i?.q === "string") : [];
  } catch {
    return [];
  }
}

export function addToHistory(items: HistoryItem[], item: HistoryItem): HistoryItem[] {
  const next = [item, ...items.filter((i) => !(i.q === item.q && i.mode === item.mode))].slice(0, MAX);
  try {
    localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    /* not saved; still shown for this visit */
  }
  return next;
}

export function clearHistory(): HistoryItem[] {
  try {
    localStorage.removeItem(KEY);
  } catch {
    /* nothing to clear */
  }
  return [];
}
