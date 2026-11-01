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
      let body = pattern.slice(i + 1, j).replace(/[\\\]]/g, "\\$&");
      if (body.startsWith("!")) body = `^${body.slice(1)}`;
      else if (body.startsWith("^")) body = `\\${body}`;
      out += `[${body}]`;
      i = j;
    } else out += c.replace(SPECIAL, "\\$&");
  }
  return out;
}
