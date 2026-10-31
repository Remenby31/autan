# Writing style

How to write a page of this site. The model is the Vercel documentation, read on 2026-10-31 in headless Chrome: twelve pages across getting started, concepts, guides and CLI reference ([sources](#sources)). This page covers prose; layout and theme are in [docs-site.md](design/docs-site.md), the rules for every contributor in [AGENTS.md](../AGENTS.md#docs).

## Pick the page type first

Every page is one of four types. Do not mix them on one page; link instead.

| Type | Answers | Title | Lives in |
|---|---|---|---|
| Getting started | "Show me it working." | `Getting started with <thing>` | `docs/getting-started/` |
| Concept | "How does it work, and why?" | Noun, or the task it enables: `Meshes`, `Runs` | `docs/concepts/` |
| Guide | "How do I do X?" | Imperative verb: `Check a case before you run it` | `docs/guides/` |
| Reference | "What exactly is X?" | The exact name: `autan plan`, `Contract: results` | `docs/contracts/`, `docs/sdk/` |

Pages for contributors (plan, research, design) are a fifth type, the project record: they keep their own form and language.

Vercel keeps the same split: *What is Compute?* explains, *Getting started with Vercel Functions* walks through one example, `vercel deploy` lists every option under its own heading.

## Open with what the reader gets

The first sentence under the title says what the thing is or what the page lets you do. No preamble, no "This page describes".

Vercel: "Use the Node.js runtime to run Vercel Functions written in JavaScript or TypeScript." "Environment variables are key-value pairs configured outside your source code…"

> Before: `autan plan CASE --mesh-info MESHINFO [--gpus N] [--json]` checks a case against its mesh before any GPU is reserved: it resolves every boundary and cell zone and shows where each condition comes from (…), reports coded diagnostics (…) and estimates device memory.
>
> After: `autan plan` checks a case against its mesh before you reserve a GPU. It shows the condition of every zone and where it comes from, reports coded diagnostics and estimates device memory.

A guide then lists its prerequisites, its numbered steps, and ends on **Next steps** with links. A concept page ends on links to the reference that defines it.

## Talk to the reader

- **Second person, active voice, present tense.** On the twelve pages, "you" and "your" occur 230 times in 7 100 words; "we" twice, "our" never.
- **Name the actor.** The subject is the reader, Autan, or a named component (`autan plan`, the mesh reader), never an unnamed process.
- **Short sentences, short paragraphs.** Vercel's median sentence is 15 words (90th percentile 28); the median paragraph is one sentence of 15 words, the 90th percentile 42 words.
- **One idea per paragraph.** Lists for parallel items, tables for anything with two or more attributes.

> Before: Units are parsed by Pint with its default registry.
>
> After: Autan parses units with Pint's default registry.

> Before: Until the C++ mesh readers exist, the MeshInfo is passed explicitly.
>
> After: Until the mesh readers ship, pass the MeshInfo yourself with `--mesh-info`.

## Headings say what the section does

- In a concept page: `How boundary conditions are resolved`, `Why ghost cells exist`. Vercel: "How the Vercel Functions lifecycle works", "Why functions should run near your data source".
- In a guide: imperative, one step per heading. Vercel: "Install the Vercel CLI", "Deploy your project".
- In reference: the exact name of the option, field or code, one heading each, so the table of contents lists them. Vercel gives each `vercel deploy` option its own heading.
- Sentence case. No numbering, no trailing punctuation.

> Before: `## Building a mesh`
>
> After (concept): `## How readers build a mesh`

## Code carries the example

- Show code early. Vercel's *Functions* page opens on a ten-line handler before any prose.
- Every command or snippet must run against `main` as written. If it does not run yet, do not show it: describe the target in prose and link to the plan.
- Under a command, one sentence says what it did, and the output follows when it matters.
- Give the full command, flags included. No `...` in a command the reader will paste.

## Notes are rare and short

A blockquote renders as a note. Use it for a constraint or side effect the reader would otherwise miss, in one or two sentences. Vercel: "The first deployment of a new project is always a production deployment, even when you omit `--prod`."

Never put the main content, a summary or a quotation in a note.

## Diagrams replace text

Add a diagram only when it removes a paragraph; the prose around it does not repeat what it shows. Diagrams are SVG files in `docs/assets/diagrams/`, linked as images (`![Owner and neighbour of a face](../assets/diagrams/mesh-faces.svg)`), and the site inlines them.

- Colours come from [tokens.css](design/ui-inspiration/tokens.css) with a fallback, `fill="var(--fg-2, #52525b)"`, so the diagram follows the theme toggle. A `<style>` block may redefine the variables under `prefers-color-scheme: dark` for GitHub; the site drops it.
- The logo's language: polyhedral cells in outline, flow in the Cp palette (`--series-1` blue for low pressure, `--fg-3` neutral, `--series-2` red for high), `--accent` for the one thing the eye should land on.
- Labels in Geist, 12–13 px, a word or an identifier each. No sentences inside a diagram.
- `viewBox` without `width` or `height`; `<title>` and `<desc>` first, referenced by `aria-labelledby`; ids prefixed with the file name, since inlined diagrams share one document.
- Hand-written, under 6 kB, no embedded fonts or raster images.

## What we never write

- "Simply", "just", "easily", "obviously", "please", "powerful", "seamless". None occurs on the twelve Vercel pages.
- History: "previously", "now supports", "new". The page describes the present.
- Future promises in user pages. A planned feature is stated once, with its milestone, and linked to [the plan](PLAN.md).
- The same fact on two pages. State it once, link to it elsewhere ([AGENTS.md](../AGENTS.md#docs)).
- Questions as headings, exclamation marks, emoji.

## French pages

A user page exists in English first. Its French version sits next to it as `<page>.fr.md` and starts with the hash of the English file it translates:

```yaml
---
source_hash: 3f9c2a7b1d04
---
```

`bun scripts/translations.ts hash <file>` in `website/` prints the value; CI fails when it no longer matches the English page. Update the translation in the same change as its source. The same rules apply in French: *vous*, active voice, short sentences. Keep code, identifiers, error codes and field names in English.

Project pages written in French (the plan, the state of the art) carry `lang: fr` instead and are shown as they are on both sites.

## Sources

Read on 2026-10-31 at <https://vercel.com/docs>: the home page, `getting-started-with-vercel`, `functions`, `functions/quickstart`, `fundamentals/what-is-compute`, `cli/deploy`, `deployments`, `project-configuration`, `builds/build-image`, `environment-variables/managing-environment-variables`, `functions/runtimes/node-js`, `incremental-static-regeneration` and `rolling-releases`. Counts are over the paragraph text of these pages (438 sentences).
