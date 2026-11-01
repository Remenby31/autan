/** Whether `name` matches the zone pattern `pattern` as the SDK matches it (case-schema.md:
 * `*`, `?`, `[…]`, case-sensitive, the whole name; Python's fnmatchcase). */
export function globMatches(pattern: string, name: string): boolean {
  return new RegExp(`^${translate(pattern)}$`, "su").test(name);
}

const SPECIAL = /[\\^$.*+?()[\]{}|]/g;

// Python's fnmatch._translate (3.12+): `[` opens a class only when a `]` closes it, a `]` right
// after `[` or `[!` belongs to the class, `!` first negates it.
function translate(pattern: string): string {
  let out = "";
  for (let i = 0; i < pattern.length; i++) {
    const c = pattern[i] as string;
    if (c === "*") out += ".*";
    else if (c === "?") out += ".";
    else if (c === "[") {
      let j = i + 1;
      if (pattern[j] === "!") j++;
      if (pattern[j] === "]") j++;
      while (j < pattern.length && pattern[j] !== "]") j++;
      if (j >= pattern.length) {
        out += "\\[";
        continue;
      }
      out += charClass(pattern.slice(i + 1, j));
      i = j;
    } else out += c.replace(SPECIAL, "\\$&");
  }
  return out;
}

// The class body as Python's fnmatch rewrites it, then read as Python's re reads a class.
function charClass(body: string): string {
  const negated = body.startsWith("!");
  // fnmatch: split at each `-` that can end a range, drop reversed ranges by merging their
  // chunks, and join the chunks back with range dashes.
  let chunks: string[] = [];
  if (body.includes("-")) {
    let from = 0;
    let k = negated ? 2 : 1;
    for (;;) {
      k = body.indexOf("-", k);
      if (k < 0) break;
      chunks.push(body.slice(from, k));
      from = k + 1;
      k += 3;
    }
    const last = body.slice(from);
    if (last) chunks.push(last);
    else chunks[chunks.length - 1] += "-";
    for (let k2 = chunks.length - 1; k2 > 0; k2--) {
      const before = chunks[k2 - 1] as string;
      const after = chunks[k2] as string;
      if ((before[before.length - 1] as string) > (after[0] as string)) {
        chunks[k2 - 1] = before.slice(0, -1) + after.slice(1);
        chunks.splice(k2, 1);
      }
    }
  } else chunks = [body];
  // Each chunk's own `-` is literal; the joints are ranges, read left to right as re does.
  const tokens: { c: string; dash: boolean }[] = [];
  chunks.forEach((chunk, n) => {
    if (n > 0) tokens.push({ c: "-", dash: true });
    for (const c of chunk) tokens.push({ c, dash: false });
  });
  // Negation is read from the rewritten class: merging a reversed range can bring `!` first.
  const negates = tokens[0]?.c === "!" && !tokens[0].dash;
  if (tokens.length === 0) return "(?!)";
  if (tokens.length === 1 && negates) return ".";
  let t = negates ? 1 : 0;
  const items: string[] = [];
  while (t < tokens.length) {
    const lo = tokens[t] as { c: string; dash: boolean };
    const dash = tokens[t + 1];
    const hi = tokens[t + 2];
    if (dash?.dash && hi !== undefined) {
      items.push(`${escapeInClass(lo.c)}-${escapeInClass(hi.c)}`);
      t += 3;
    } else {
      items.push(escapeInClass(lo.c));
      t += 1;
    }
  }
  return `[${negates ? "^" : ""}${items.join("")}]`;
}

const escapeInClass = (c: string) => c.replace(/[\\\][^-]/g, "\\$&");

/** A zone name as a pattern matching that name only (`[`, `*`, `?` made literal). */
export function globEscape(name: string): string {
  return name.replace(/[[*?]/g, "[$&]");
}
