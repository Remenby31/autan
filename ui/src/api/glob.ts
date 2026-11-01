/** Whether `name` matches the zone pattern `pattern` as the SDK matches it (case-schema.md:
 * `*`, `?`, `[…]`, case-sensitive, the whole name; Python's fnmatchcase). */
export function globMatches(pattern: string, name: string): boolean {
  return new RegExp(`^(?:${translate(pattern)})$`, "su").test(name);
}

const SPECIAL = /[\\^$.*+?()[\]{}|]/g;

// Python 3.13's fnmatch.translate, over code points: `[` opens a class only when a `]` closes
// it, a `]` right after `[` or `[!` belongs to the class; consecutive `*` are one. Between two
// stars, the fixed part is matched lazily and atomically as Python does (`(?>.*?fixed)`, here
// `(?=(.*?fixed))\N`), so no pattern backtracks without bound.
function translate(pattern: string): string {
  const pat = Array.from(pattern);
  const parts: string[] = [];
  const stars: number[] = [];
  for (let i = 0; i < pat.length; i++) {
    const c = pat[i] as string;
    if (c === "*") {
      stars.push(parts.length);
      parts.push(".*");
      while (pat[i + 1] === "*") i++;
    } else if (c === "?") parts.push(".");
    else if (c === "[") {
      let j = i + 1;
      if (pat[j] === "!") j++;
      if (pat[j] === "]") j++;
      while (j < pat.length && pat[j] !== "]") j++;
      if (j >= pat.length) parts.push("\\[");
      else {
        parts.push(charClass(pat.slice(i + 1, j)));
        i = j;
      }
    } else parts.push(c.replace(SPECIAL, "\\$&"));
  }
  if (!stars.length) return parts.join("");
  let out = parts.slice(0, stars[0]).join("");
  let group = 0;
  for (let s = 1; s < stars.length; s++) {
    group++;
    out += `(?=(.*?${parts.slice((stars[s - 1] as number) + 1, stars[s]).join("")}))\\${group}`;
  }
  return `${out}.*${parts.slice((stars[stars.length - 1] as number) + 1).join("")}`;
}

// The class body as fnmatch rewrites it, then read as Python's re reads a class: split at each
// `-` that can end a range, drop reversed ranges by merging their chunks, join the chunks back
// with range dashes; negation is read from the rewritten class.
function charClass(body: string[]): string {
  let chunks: string[][] = [body];
  if (body.includes("-")) {
    chunks = [];
    let from = 0;
    let k = body[0] === "!" ? 2 : 1;
    for (;;) {
      k = body.indexOf("-", k);
      if (k < 0) break;
      chunks.push(body.slice(from, k));
      from = k + 1;
      k += 3;
    }
    const last = body.slice(from);
    if (last.length) chunks.push(last);
    else chunks[chunks.length - 1]?.push("-");
    for (let n = chunks.length - 1; n > 0; n--) {
      const before = chunks[n - 1] as string[];
      const after = chunks[n] as string[];
      if (codePoint(before[before.length - 1]) > codePoint(after[0])) {
        chunks[n - 1] = [...before.slice(0, -1), ...after.slice(1)];
        chunks.splice(n, 1);
      }
    }
  }
  // Each chunk's own `-` is literal; the joints are ranges, read left to right as re does.
  const tokens: { c: string; joint: boolean }[] = [];
  chunks.forEach((chunk, n) => {
    if (n > 0) tokens.push({ c: "-", joint: true });
    for (const c of chunk) tokens.push({ c, joint: false });
  });
  const negates = tokens[0]?.c === "!" && !tokens[0].joint;
  if (tokens.length === 0) return "(?!)";
  if (tokens.length === 1 && negates) return ".";
  const items: string[] = [];
  for (let t = negates ? 1 : 0; t < tokens.length; ) {
    const lo = tokens[t] as { c: string };
    const hi = tokens[t + 2];
    if (tokens[t + 1]?.joint && hi !== undefined) {
      items.push(`${inClass(lo.c)}-${inClass(hi.c)}`);
      t += 3;
    } else {
      items.push(inClass(lo.c));
      t += 1;
    }
  }
  return `[${negates ? "^" : ""}${items.join("")}]`;
}

const codePoint = (c: string | undefined) => c?.codePointAt(0) ?? -1;
const inClass = (c: string) => c.replace(/[\\\][^-]/g, "\\$&");

/** A zone name as a pattern matching that name only (`[`, `*`, `?` made literal). */
export function globEscape(name: string): string {
  return name.replace(/[[*?]/g, "[$&]");
}
