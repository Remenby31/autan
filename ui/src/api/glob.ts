/** Whether `name` matches the zone pattern `pattern` as the SDK matches it (case-schema.md:
 * `*`, `?`, `[…]`, case-sensitive, the whole name; Python's fnmatchcase). */
export function globMatches(pattern: string, name: string): boolean {
  return new RegExp(`^${translate(pattern)}$`, "su").test(name);
}

const SPECIAL = /[\\^$.*+?()[\]{}|]/g;

// Python's fnmatch.translate: `[` opens a class only when a `]` closes it, a `]` right after
// `[` or `[!` belongs to the class, `!` first negates it.
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

// A character class as Python reads it: `!` first negates, `a-b` is a range, a reversed range
// (`z-a`) is dropped, and a class left empty matches nothing, or any character when negated.
function charClass(body: string): string {
  const negated = body.startsWith("!");
  const stuff = negated ? body.slice(1) : body;
  const items: string[] = [];
  for (let k = 0; k < stuff.length; k++) {
    const c = stuff[k] as string;
    const end = stuff[k + 2];
    if (stuff[k + 1] === "-" && end !== undefined) {
      if (c <= end) items.push(`${escapeInClass(c)}-${escapeInClass(end)}`);
      k += 2;
    } else items.push(escapeInClass(c));
  }
  if (!items.length) return negated ? "." : "(?!)";
  return `[${negated ? "^" : ""}${items.join("")}]`;
}

const escapeInClass = (c: string) => c.replace(/[\\\][^-]/g, "\\$&");

/** A zone name as a pattern matching that name only (`[`, `*`, `?` made literal). */
export function globEscape(name: string): string {
  return name.replace(/[[*?]/g, "[$&]");
}
