# Documentation site

The site at <https://remenby31.github.io/autan/> renders the Markdown of this repository. It lives in `website/`, owns no content and is rebuilt from `docs/` and `README.md` on every push to `main`.

## Writing for the site

- A page is any `.md` or `.mdx` file under `docs/`, plus `README.md`, which becomes the introduction. Its URL is its path under `docs/`, lower-cased, without the extension: `docs/contracts/results.md` → `/autan/contracts/results`.
- The first `#` heading is the page title. The sidebar shortens it: a one-word prefix before `: ` is dropped ("Contract: results" → "Results"), and so is a subtitle after ` — `.
- The folder is the sidebar section. Known sections come in this order: Contracts, SDK, Validation, Design, Research; a new folder gets its own section after them, so a page never needs a site change to appear.
- Link to other files with relative paths, as on GitHub. The build rewrites a link to a page into a site link, a link to an HTML document into its published copy, and any other link (source files, `AGENTS.md`) into a GitHub link. A link to a missing file is reported by the build as `[docs] <file>: link to missing file <target>`.
- Standalone HTML documents under `docs/` are published unchanged at their repository path (`docs/design/dx.html` → `/autan/docs/design/dx.html`) and listed in the sidebar with an external-link icon, under their `<title>`.
- Blockquotes render as a note box: use them for status and warnings, not for quotations.

Preview locally from `website/`: `bun install --frozen-lockfile`, then `bun run dev`, or `bun run build && bun run preview` for the exact GitHub Pages output.

## What we take from the Vercel documentation

Studied on 2026-10-31 on <https://vercel.com/docs>, headless Chrome, light and dark, 1440 px and 390 px wide.

| Vercel | Here |
|---|---|
| Three columns: sidebar (≈ 290 px), content (≈ 770 px measure), "On this page" (≈ 230 px) with the current section marked by a bar | Same layout; Fumadocs' docs layout |
| Sidebar search box with a `⌘K` hint; the dialog lists pages, then matching sections | Same; the index is built at compile time and searched in the browser, so it works offline |
| Breadcrumb above the title (`Functions / Runtimes / Node.js`) | Section above the title |
| Title at 40–56 px, weight 600, tight negative tracking; body 16 px, line height 1.7 | Same proportions |
| Code block: thin border, 8 px radius, file name or `terminal` header, copy button; package-manager tabs (`pnpm` `yarn` `npm` `bun`) | Bordered blocks with a copy button; tabs available in `.mdx` (`<Tabs>`) for when the SDK has install commands |
| Inline code in a bordered chip | Same |
| Notes as bordered boxes with an icon, not italic quotations | Blockquotes styled as notes; `<Callout>` in `.mdx` |
| Reference pages (CLI): one page per command, commands in monospace in the sidebar, one heading per option so the TOC lists them | Model for the SDK and CLI reference |
| "Copy page" as Markdown, "View as Markdown", `llms.txt` | "Copy Markdown" and "Open" on every page; `/autan/llms.txt`, `/autan/llms-full.txt` |
| Previous/next links at the bottom | Same |
| Dark theme follows the system; near-black background | Same, plus a manual toggle |
| Mobile: sidebar and TOC collapse into menus | Same |

We do not take the marketing header, sign-up buttons, "Ask AI" or the feedback form: nothing behind them exists here.

## Stack

Fumadocs 16 on React Router 8 (Vite 8), prerendered to static files, Bun 1.3, Tailwind 4, Biome; the same toolchain as the web app in `ui/` (Bun, Vite, Tailwind, Biome). Colours, fonts and radii come only from `docs/design/ui-inspiration/tokens.css`, which the app also uses; fonts are Geist and Geist Mono (SIL OFL 1.1), self-hosted.

We built the three official Fumadocs templates on 2026-10-31 and compared them:

| | Next.js 16, `output: 'export'` | React Router 8, prerender | TanStack Start 1.169, SPA prerender |
|---|---|---|---|
| Static export for GitHub Pages | Yes | Yes: every loader is prerendered to a `.data` file, so client navigation stays static | **No**: client navigation calls `/_serverFn/…`, which 404s on a static host (reproduced) |
| Base path `/autan/` | `basePath` | `basename` + Vite `base` | — |
| Offline search | Static index | Static index | Static index |
| Template build | 7 s | 4 s | 3 s |
| `node_modules` | 617 MB | 390 MB | 391 MB |
| Shares the app's toolchain | No (webpack/Turbopack, Next) | Vite | Vite and TanStack Router |

React Router wins on the static constraint and on maintenance: it is the Vite option Fumadocs supports end to end in static mode, and it shares Vite, Tailwind and Biome with `ui/`. Next.js would work as well but brings a second build system and twice the dependencies. On 2026-10-31 the full build took 4.3 s and produced 6.8 MB, of which 1.3 MB is the search index, fetched on the first search.

## Build and deploy

`.github/workflows/docs.yml` builds the site on every pull request that touches `website/`, `docs/` or `README.md`, and deploys it to GitHub Pages on push to `main` only. The build writes `website/dist/`: the prerendered pages, the search index, the Markdown exports, the HTML documents and a `404.html`.
