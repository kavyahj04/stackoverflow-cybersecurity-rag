import type { AnswerResult } from "./api";

/** The answer as Markdown with numbered references, for notes or a paper draft. */
export function answerToMarkdown(a: AnswerResult): string {
  const lines = [`## ${a.question}`, "", a.text, ""];
  if (a.age_note) lines.push(`> ${a.age_note}`, "");
  if (a.sources.length) {
    lines.push("**Sources** (Security Stack Exchange, CC BY-SA 4.0)", "");
    for (const s of a.sources) {
      const flags = [s.accepted ? "accepted" : null, `${s.votes >= 0 ? "+" : ""}${s.votes} votes`, s.year]
        .filter(Boolean)
        .join(", ");
      lines.push(`[${s.n}] ${s.thread_title} (${flags}) ${s.url}`);
    }
  }
  return lines.join("\n");
}

export function downloadJson(data: unknown, filename: string) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

export function slug(text: string): string {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 50) || "answer";
}
