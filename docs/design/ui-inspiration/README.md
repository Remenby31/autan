# UI inspiration and visual direction

The web UI (milestone M10, `docs/PLAN.md`) has five screens: **Home** (runs and studies), **Setup** (preparing a case), **Live** (a run in progress), **Analysis** (3D results) and **Compare** (two or more runs). The current mockup is [`../ux-mockup.html`](../ux-mockup.html). This page records what we take from tools that do this well, then fixes the visual and interaction direction. Tokens live in [`tokens.css`](tokens.css); the UI and the docs site import that file.

The brief: **as little text as possible**. Shape, state, motion and the keyboard carry the meaning. Screens are full-bleed and immersive, not grids of cards.

## References

Each reference links to the public page it was studied on (2026-10-31). Third-party screenshots are not redistributed here.

### Vercel

[Deployments](https://vercel.com/docs/deployments) · [Managing deployments](https://vercel.com/docs/deployments/managing-deployments) · [Runtime logs](https://vercel.com/docs/logs/runtime) · [Instant rollback](https://vercel.com/docs/instant-rollback) · [Observability](https://vercel.com/docs/observability) · [Speed Insights](https://vercel.com/docs/speed-insights)

The closest model for the run lifecycle: a deployment is a run, a project is a case.

- **Run list as a dense table, one row per run** (Home). A row reads left to right: what changed, state dot with duration, target, short hash, branch, relative date, author. No card, no description. We use: case diff summary · state + wall time · GPUs and precision · mesh hash · relative date.
- **Filters are chips, and the state filter shows its colours** (Home). `Status 4/5` with the coloured dots in the trigger tells you what is hidden without opening it.
- **Search is anchored to the sidebar, not a centred modal.** In the signed-in dashboard, `Find` sits at the top of the sidebar and opens as a dropdown below it, mixing entities (projects, settings) and a natural-language question. We keep a centred ⌘K for actions and put entity search where the eye already is.
- **Detail page: a metadata grid, then collapsible sections** (Live, Analysis). Created · Status · Duration · Environment in one row; Summary, Checks, Domains folded below with a count or a state icon on the right. Our run header: Started · State · Iteration · GPUs · ETA; sections fold for Plan, Monitors, Outputs.
- **Live logs: a histogram timeline above a streamed table, and a Live toggle** (Live). The histogram shows where the errors are before you read a line. Ours plots residual and warnings per iteration above the solver log; scrolling up pauses Live, `End` resumes.
- **Empty states offer the next source** (all). Vercel's empty runtime-logs view points at request logs and CDN requests instead of explaining why it is empty.
- **Compare as "current vs. other", two stacked rows with a role badge** (Compare). `Current` / `Previous` tags beat a two-column table when you compare two runs.
- **Deployment summary rows: icon, monospace path, right-aligned metrics and a region chip** (Setup). The same anatomy fits boundary zones: zone name, BC type chip, face count, provenance.
- Also: the resources tab (search, segmented filter, row actions menu); the production tile (preview thumbnail beside its metadata); Observability and Speed Insights (one big chart, a metric rail on the left, threshold lines); the docs ⌘K menu (matched term highlighted, `Esc` hint).

### Linear

[linear.app](https://linear.app/)

- **Keyboard first, everywhere** (all). Every action has a key, shown in menus and in the palette; `j`/`k` move, `x` selects, `Space` peeks.
- **Optimistic updates.** The UI changes before the server answers and reconciles quietly. For us: relabel a BC and the viewport recolours at once; `plan` re-runs in the background.
- **Low-contrast chrome, high-contrast content.** Sidebar and toolbars sit at `--fg-2`; the work is at `--fg`. The board: columns with counts, no borders around cards beyond one hairline.

### Raycast

[raycast.com](https://www.raycast.com/)

- **The palette is the app** (all). A list on the left, a detail pane on the right, an action bar at the bottom with the primary action and `⌘K` for more. Ours: type a case, run or command; the right pane previews (residual sparkline, thumbnail); `↵` opens, `⌘↵` runs.
- **Footer hints instead of help text.** Shortcuts teach themselves.

### Arc

Arc is being retired (arc.net now shows a farewell page), so no screenshot. We keep two ideas: the sidebar is the only navigation chrome, and a command bar replaces menus.

### Figma

[figma.com/design](https://www.figma.com/design/)

- **Canvas first, panels float** (Setup, Analysis). Inspector on the right reflects the selection; nothing selected shows the document properties.
- **Selection drives everything.** Click a surface in the viewport: the inspector shows its zone and BC, the code drawer highlights the matching line.

### Stripe

[Workbench](https://docs.stripe.com/workbench)

- **Workbench: a developer drawer docked to the bottom of every page** (all). Overview, Events, Logs, without leaving the screen. Our equivalent: `` ` `` opens a drawer with `case.py`, the solver log and the SDK call that reproduces the view.
- **Errors carry a code and a fix.** Matches the CLI's `E0412` style (`../dx.html`).

### Grafana

[Stats](https://play.grafana.org/d/Zb3f4veGk/stats) · [Gradient area fills](https://play.grafana.org/d/37Dq903mk/graph-gradient-area-fills) · [play.grafana.org](https://play.grafana.org/) (⌘K)

- **Stat with sparkline** (Live, Home). One big number, the trend behind it, a threshold colour. Used for CL, CD, residual drop and ns/cell/iteration.
- **Shared time cursor** across panels (Live, Compare): hover one chart, every chart shows the same iteration.
- **Time range and refresh as one control** (Live): `Last 6 hours · Refresh 30s`. Ours: iteration window + Live.
- The ⌘K palette: results grouped by type (pages, folders, dashboards) with the breadcrumb in muted text.

### Weights & Biases

[Public workspace](https://wandb.ai/stacey/deep-drive/workspace) · [Experiment tracking](https://wandb.ai/site/experiment-tracking/)

The nearest analogue to studies and comparisons.

- **Runs table on the left, panels on the right, one colour per run** (Compare). The swatch next to the run name is the legend for every chart; the eye toggles visibility. We use the `--series-*` colours the same way.
- **Overlay curves of many runs on one chart** (Compare, Home for studies): residuals, CL vs. iteration.
- **Parallel coordinates for sweeps** (Home > study): Mach, AoA, mesh level and turbulence model on the axes, the objective last.

### Onshape

[onshape.com](https://www.onshape.com/en/)

- **A full CAD tool in the browser with no install**, and a feature tree beside the model. Our Setup tree follows the case: Geometry, Mesh, Physics, Conditions, Run, Results, each with a state glyph.

### SimScale

[Product](https://www.simscale.com/product/)

- **The setup tree uses state glyphs: green filled, red ring, grey empty** (Setup). Without reading a word you see that `Boundary conditions` and the material are incomplete. We take exactly this.
- **Colour legend bottom-right, small, with units** (Analysis).
- Avoid: the top toolbar of thirty labelled icons. Our tools live in the palette and on keys.

### ParaView Glance

[Glance app](https://kitware.github.io/glance/app/), sample "Formula 1"

- **Pipeline as a stack of collapsible blocks per dataset** (Analysis): representation, colour by, opacity. Ours: one block per field layer (surface, slice, streamlines, iso-surface).
- **Camera presets on a thin rail at the viewport edge** (X, Y, Z, reset). Ours, with `⌥1`–`⌥6` for the six axis views and `f` to frame the selection.

### Not captured

Luminary Cloud: no authenticated workspace was available to study; its public pages show marketing and a prediction demo. We keep its principle from `../dx.html`: the SDK and the UI show the same case.

## Direction

### Principles

1. **Minimal text.** A label only when a glyph, position or colour cannot carry the meaning. No sentences in the chrome, no explanatory paragraphs, no tooltips as a crutch: if a control needs one, redesign it. Units and numbers stay; prose goes.
2. **The work fills the screen.** The viewport, the chart or the run table is the page. Panels float over the viewport on `--glass` with backdrop blur and fold away (`\` hides all chrome). No dashboard of cards.
3. **Keyboard first.** Everything is reachable from ⌘K; every frequent action has a single key, shown next to it in menus and the palette. `?` shows the full map. The mouse is for the viewport. Rules: `⌘` on macOS is `Ctrl` elsewhere; the app claims ⌘K, `/` and its single keys only while the page has focus and no editable field (input, code drawer, palette) does; the innermost open layer (palette, popover, drawer) handles a key first and `Esc` closes it; irreversible actions take a modifier (`⇧S` stops a run); single-key shortcuts can be turned off.
4. **Instant feedback.** Optimistic changes; local recomputation (`plan`, BC coverage) under 100 ms; no blocking spinner. A run starts only after `plan` passes on the current revision of the case: `⌘↵` while `plan` is pending queues the launch and shows its progress on the button; a failing `plan` cancels it and shows the codes. Long work shows progress in place (the state dot, the row, the tab title).
5. **State is visible.** Every run, zone and setup step carries a state glyph. Colour is reserved for state and data; the chrome is greyscale.
6. **Motion explains change.** Motion says where something came from or what changed: a panel slides from its trigger, a changed value flashes once, a new log line fades in. Nothing loops except the live pulse. Under `prefers-reduced-motion` all durations drop to zero and the running glyph stays static.
7. **Code is the source of truth.** Every change to the simulation (geometry, mesh, physics, conditions, run settings) edits `case.py`, and the drawer shows the diff. View state (camera, displayed field, layout, filters) never touches the case and lives in the URL.

### Tokens

Defined in [`tokens.css`](tokens.css), light and dark. Dark is the default for Live and Analysis (viewport contrast); Home and Compare follow the system theme.

| Group | Choice | Why |
|---|---|---|
| Type | Geist Sans for UI, Geist Mono for numbers, paths, code (both SIL OFL 1.1, self-hosted). A compact base size, three weights. `font-variant-numeric: tabular-nums` on every number. | Dense UI needs a compact sans; numbers must align in columns and not jitter while they update. |
| Neutrals | `--bg-0…3`, `--line`, `--line-strong`, `--fg`, `--fg-2`, `--fg-3`. Every text colour, `--fg-3` included, reaches 4.5:1 on `--bg-0` and `--bg-1` in both themes. | Chrome stays out of the way of the colour map. |
| Accent | One blue, `--accent`: focus, selection, primary action, running state. | A single accent keeps state colours readable. |
| Semantic | `--success` converged, `--warning` stalled or risky (W-codes), `--danger` diverged or error (E/D-codes), `--info` = accent. Each has a `-soft` fill. | Mirrors the CLI's coded messages. |
| Run states | `--state-queued`, `-planning`, `-running`, `-converged`, `-stalled`, `-diverged`, `-stopped`. | One glyph language for rows, tabs, tree and favicon. |
| Series | `--series-1…8`, assigned in order to compared runs and kept for the session. | Same run, same colour on every chart and in the 3D legend. |
| Field colour maps | Diverging: cool–warm (Moreland 2009) for signed fields (Cp, Δ). Sequential: viridis for magnitudes. Never jet/rainbow. | Perceptually uniform and readable by colour-blind users. Lightness alone does not carry the sign, so signed fields are normalised around zero and the legend marks 0 and the sign. |
| Density | A 4 px grid; regular and compact row heights; one control height, a taller one for primary actions. | Tables of hundreds of runs fit on one screen. |
| Radii | Four steps: chips and inputs · buttons and rows · panels and popovers · dialogs and palette. | Rounded enough to read as interactive, not soft. |
| Elevation | Borders and background steps first; shadows only on popovers and the palette. In dark mode the shadow includes a 1 px ring. | Shadows vanish on dark backgrounds. |
| Motion | Four durations (press, hover, popover, panel), `--ease-out` for entry, one live pulse. | Fast enough to feel instant, slow enough to follow. |

### Components

- **State glyph.** An 8 px shape, so state reads without colour. Each glyph has an accessible name equal to the state.

  | State | Glyph | Colour | Reduced motion |
  |---|---|---|---|
  | queued | hollow circle | `--state-queued` | same |
  | planning | dashed ring | `--state-planning` | same |
  | running | filled circle with an expanding halo | `--state-running` | filled circle with a static halo |
  | converged | filled circle | `--state-converged` | same |
  | stalled | filled circle with a horizontal bar | `--state-stalled` | same |
  | diverged | ring with `!` | `--state-diverged` | same |
  | stopped | filled square | `--state-stopped` | same |

  Used in rows, tabs, the setup tree and the tab favicon.
- **Run row.** Dot · name · case diff summary (mono, muted) · wall time · GPUs × precision chip · mesh hash · relative date. `Space` peeks (residual sparkline + thumbnail), `↵` opens, `x` selects, `c` compares the selection.
- **Study row.** A run row that expands into its member runs; the right edge shows the objective as a sparkline across the sweep.
- **Filter chips.** `State`, `Case`, `Mesh`, `Precision`, `Author`; the trigger shows the active values as dots or a count.
- **Command palette (⌘K).** Centred, 640 px, top third. Grouped results (cases, runs, commands, docs), shortcut on each row, a preview pane on wide screens, an action bar at the bottom. Natural-language requests go to the agent, whose answer is a code diff to accept (as in the mockup).
- **Setup rail.** Steps with state glyphs (SimScale tree); `1`–`6` jump to a step.
- **Inspector.** Floating, right or left of the viewport, reflects the selection; collapsible blocks (ParaView Glance).
- **Code drawer.** `` ` `` toggles; shows `case.py` with the last change flashed, the log and a reproduce snippet (Stripe Workbench).
- **Metric tile.** Big mono number + unit + sparkline + threshold colour (Grafana stat).
- **Live chart.** Residuals (log scale) and monitors, shared iteration cursor, Live toggle; scrolling back pauses.
- **Log stream.** Histogram timeline over a virtualised table; coded messages link to the cell in the viewport (`D0107` → `--at-cell`).
- **Viewport chrome.** Axis-view rail, colour legend bottom-right with units and range, a minimal status line (cells, iteration, FPS) in mono.
- **Compare header.** Runs as coloured chips with a role (`A`/`B`, or reference); 3D compare shows A | B | Δ with synced cameras.

### Screens

| Screen | Layout | Keys |
|---|---|---|
| Home | Sidebar (Find at top) + full-width run/study table, filter chips above, stat strip for the selection. | `/` filter, `j`/`k`, `Space`, `↵`, `x`, `c`, `n` new case |
| Setup | Full-bleed viewport, setup rail left, inspector floating, code drawer right. Missing BCs glow on the model. | `1`–`6`, `b` assign BC, `p` plan, `⌘↵` run |
| Live | Run header grid, residual chart and metric tiles over the viewport (fields update as snapshots arrive), log stream in the drawer. | `l` live, `⇧S` stop, `End` resume follow |
| Analysis | Viewport only; inspector with field layers; legend; probes and slices from the palette. | `f` frame, `⌥1`–`⌥6` views, `[`/`]` cycle field |
| Compare | Runs table with series swatches left, overlaid charts right; `v` switches to 3D A | B | Δ. | `v`, `d` case diff, `t` toggle run |

### What we avoid

- Grids of cards, hero headers, sentences explaining a control.
- Colour used for decoration, coloured chrome, gradients on surfaces; rainbow colour maps.
- Modal wizards and confirmation dialogs; prefer undo (`⌘Z`) and inline state.
- Spinners without progress, toasts for what should be inline state.
- Toolbars of labelled icons; hidden shortcuts that only a cheat sheet reveals.
- Simulation settings that exist only in the UI and not in `case.py`.
