# Repository standards: research and decisions

Checked 2026-10-31. We read the files of 12 reference repositories on their default branch, through `gh api` and the GitLab API. We also read the primary specifications. Every decision below cites the source that justifies it. "Unverified" marks a claim no primary source confirmed.

Reference repositories: [PyFR](https://github.com/PyFR/PyFR), [SU2](https://github.com/su2code/SU2), [Kokkos](https://github.com/kokkos/kokkos), [PETSc](https://gitlab.com/petsc/petsc), [deal.II](https://github.com/dealii/dealii), [DOLFINx](https://github.com/FEniCS/dolfinx), [Trixi.jl](https://github.com/trixi-framework/Trixi.jl), [JAX](https://github.com/jax-ml/jax), [PyTorch](https://github.com/pytorch/pytorch), [uv](https://github.com/astral-sh/uv), [ruff](https://github.com/astral-sh/ruff), [pydantic](https://github.com/pydantic/pydantic).

## Decisions

| Area | Decision | Main sources |
|---|---|---|
| README | State the maturity first, then why the project exists, done versus planned, architecture, how correctness and speed are proven, citing, licence | [GitHub: About READMEs](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes), [Lee 2018](https://doi.org/10.1371/journal.pcbi.1006561), [§1](#1-readme) |
| Community files | `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` (Contributor Covenant 3.0), `SECURITY.md` (private vulnerability reporting), `CITATION.cff` 1.2.0, `CHANGELOG.md`, all at the root | [§2](#2-community-health-files) |
| Issue forms | Bug, numerical discrepancy, feature; blank issues disabled | [§3](#3-issue-and-pull-request-templates) |
| PR template | Impact, change, verification evidence (tests, MMS orders, benchmarks, reproducibility, Codex review), agent disclosure | [§3](#3-issue-and-pull-request-templates) |
| Ownership | `CODEOWNERS` with the lead on every path; lanes stay in `AGENTS.md` | [§4](#4-ownership-and-labels) |
| Labels | Declared in `.github/labels.yml`, synced by a workflow, additive only; bug and feature use GitHub issue types | [§4](#4-ownership-and-labels) |
| Format and lint | `.editorconfig`, `.clang-format` (pinned clang-format 23), `.clang-tidy` (allow-list, warnings as errors), `ruff.toml`, `_typos.toml`, gersemi for CMake, pre-commit with hooks pinned by SHA | [§5](#5-format-and-lint) |
| CI | `ci.yml` runs every pre-commit hook on Ubuntu; actions pinned by SHA, `permissions: {}` by default, zizmor and actionlint on workflows | [§6](#6-ci-and-supply-chain) |
| Supply chain | OpenSSF Scorecard workflow; Dependabot for actions and pre-commit hooks, grouped, 7-day cooldown | [§6](#6-ci-and-supply-chain) |
| Releases | release-please from squashed Conventional Commit titles; SemVer 0.y.z, breaking change bumps the minor, everything else the patch | [§7](#7-versioning-releases-and-changelog) |
| Agents | `AGENTS.md` keeps its non-negotiables; it adds the pre-commit gate, the agent disclosure and three rules taken from uv, ruff and Trixi.jl | [§8](#8-agent-rules) |

## 1. README

What the guides ask:

- GitHub: a README says "What the project does; Why the project is useful; How users can get started; Where users can get help; Who maintains and contributes". It "should only contain information necessary for developers to get started using and contributing" ([About READMEs](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes)).
- [makeareadme.com](https://www.makeareadme.com/) lists a "Project status" section and asks to "show the expected output if you can".
- Lee, *Ten simple rules for documenting scientific software* ([PLoS Comput Biol 2018](https://doi.org/10.1371/journal.pcbi.1006561)): show examples, and put the citation information in the README.
- OpenSSF Best Practices `description_good` and `interact`: the README says what the software does, how to get it, give feedback and contribute ([criteria](https://www.bestpractices.dev/en/criteria/0)).

What the reference repositories do:

- None of the seven scientific codes states its maturity. DOLFINx comes closest with "actively developed" ([README](https://github.com/FEniCS/dolfinx/blob/main/README.md)). A pre-alpha project that says so up front fills a real gap.
- deal.II shows regression-tester and performance-tester badges, not just "build passing" ([README](https://github.com/dealii/dealii/blob/master/README.md)).
- Trixi.jl has a "Referencing" section with BibTeX and a software DOI ([README](https://github.com/trixi-framework/Trixi.jl/blob/main/README.md)).
- Kokkos keeps every section short and links out ([README](https://github.com/kokkos/kokkos/blob/develop/README.md)). pydantic's README is about 70 lines and holds one runnable example ([README](https://github.com/pydantic/pydantic/blob/main/README.md)).
- uv and ruff open with a tagline and evidence: a benchmark chart ([uv](https://github.com/astral-sh/uv/blob/main/README.md)). JAX shows a platform × accelerator support matrix ([README](https://github.com/jax-ml/jax/blob/main/README.md)).
- PyFR's README has no install, no quickstart, no citation and no status ([README](https://github.com/PyFR/PyFR/blob/develop/README.md)): the counter-example.

Decision: the README states "pre-alpha, nothing runs yet" before anything else. It shows a done-versus-planned table by milestone, the hardware targets, the evidence every change must carry and how to cite. It has no install section and no example until there is something to install: an example that does not run would be worse than none. Badges come in two kinds. Static badges state the stack, the licence and the project's origin, which are facts a visitor scans for (the lead's choice). Status badges appear only when they report something real: CI, and Scorecard once it has published a score; test and benchmark badges once they exist.

## 2. Community health files

### Which files count

The community profile API reports README, LICENSE, CODE_OF_CONDUCT, CONTRIBUTING, issue template and PR template ([REST: community metrics](https://docs.github.com/en/rest/metrics/community)). The repository scored 42 % on 2026-10-31 with a description, README and LICENSE: 3 of 7 items. The 7-item denominator is our inference, as GitHub does not publish the formula. `SECURITY.md` does not count towards the score, but GitHub's checklist and Scorecard both look for it ([About community profiles](https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/about-community-profiles-for-public-repositories)).

GitHub looks for health files in `.github/`, then the root, then `docs/` ([default community health file](https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/creating-a-default-community-health-file)). We put them at the root, where a reader of the file tree sees them.

### CONTRIBUTING

- Best Practices `contribution` (MUST) asks to "explain the contribution process"; `test_policy` (MUST) asks for a policy that new functionality gets tests ([criteria](https://www.bestpractices.dev/en/criteria/0)).
- JAX's contributing page lists the exact lint and test commands ([docs/contributing.md](https://github.com/jax-ml/jax/blob/main/docs/contributing.md)); uv lists every lint command ([CONTRIBUTING.md](https://github.com/astral-sh/uv/blob/main/CONTRIBUTING.md)).
- Contribution licensing: PETSc, deal.II, Trixi.jl and Kokkos use the Developer Certificate of Origin ([deal.II](https://github.com/dealii/dealii/blob/master/CONTRIBUTING.md), [Trixi.jl](https://github.com/trixi-framework/Trixi.jl/blob/main/CONTRIBUTING.md)). None uses a CLA.

Decision: `CONTRIBUTING.md` holds the setup, the hooks, the PR flow and where each rule lives. The rules themselves stay in `AGENTS.md`, which applies to humans too, so each fact is stated once. We do not add a DCO sign-off: Apache-2.0 section 5 already licenses every contribution under the project licence ([Apache-2.0 §5](https://www.apache.org/licenses/LICENSE-2.0#contributions)), and agents cannot certify origin on their own. Revisit if outside contributors arrive.

### Code of conduct

- Contributor Covenant 3.0 is the current version ([3.0](https://www.contributor-covenant.org/version/3/0/code_of_conduct/)). The reference repositories use older versions: 1.4 (SU2, deal.II, DOLFINx, Kokkos, PyTorch) or 2.0 (Trixi.jl, Astral). PETSc uses the NumFOCUS code.
- The text asks to fill in a reporting channel and to review the enforcement section. A private channel is the norm.

Decision: Contributor Covenant 3.0, unchanged except the reporting channel. Reports go privately to the maintainer through the repository's private reporting form, which only the maintainer reads. The repository publishes no personal email address.

### Security policy

- Best Practices `vulnerability_report_process` and `vulnerability_report_private` (MUST): publish the process and say how to report privately. "Projects hosted on GitHub SHOULD consider enabling privately reporting a security vulnerability" ([criteria](https://www.bestpractices.dev/en/criteria/0)). `vulnerability_report_response` (MUST) requires a first response within 14 days.
- Scorecard Security-Policy gives points for a link or email, and for disclosure timelines in text ([checks.md](https://github.com/ossf/scorecard/blob/main/docs/checks.md#security-policy)).
- Kokkos: acknowledgement "within five business days", "90-day disclosure timeline" ([SECURITY.md](https://github.com/kokkos/kokkos/blob/develop/docs/SECURITY.md)). PyTorch, pydantic and uv state a scope: crashes from invalid arguments are bugs, not vulnerabilities ([PyTorch](https://github.com/pytorch/pytorch/blob/main/SECURITY.md), [pydantic](https://github.com/pydantic/pydantic/blob/main/SECURITY.md)).

Decision: report through GitHub private vulnerability reporting, which we enable. Acknowledge within 5 business days, disclose within 90 days. The scope is what matters for a solver: readers parse untrusted mesh and case files, so memory-safety bugs in readers are in scope. A wrong numerical answer is not a vulnerability; it goes to the numerical discrepancy form.

### Citation

- CFF 1.2.0 requires `cff-version`, `message`, `title` and `authors` ([schema](https://github.com/citation-file-format/citation-file-format/blob/main/schema-guide.md)). GitHub shows "Cite this repository" from a root `CITATION.cff` ([About CITATION files](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files)).
- Kokkos, deal.II, DOLFINx, PyTorch and pydantic ship a `CITATION.cff`. DOLFINx's release updates `version` and `date-released` in it ([RELEASE.md](https://github.com/FEniCS/dolfinx/blob/main/RELEASE.md)).
- If a `.zenodo.json` exists, Zenodo ignores `CITATION.cff` ([Zenodo](https://help.zenodo.org/docs/github/describe-software/citation-file/)).

Decision: a `CITATION.cff` validated against the CFF 1.2.0 schema in pre-commit (the check-jsonschema hook `check-citation-file-format`; cffconvert 2.0.0 predates its own hook), with no `version` until the first release and no `.zenodo.json`. Zenodo archiving is a lead decision at the first release.

### Changelog

- Keep a Changelog 1.1.0: "Changelogs are for humans", "Don't let your friends dump git logs into changelogs" ([1.1.0](https://keepachangelog.com/en/1.1.0/)).
- Best Practices `release_notes` (MUST): release notes are a human-readable summary, "not the raw output of a version control log" ([criteria](https://www.bestpractices.dev/en/criteria/0)).
- release-please builds the changelog from commit subjects, with sections "Features", "Bug Fixes" and so on; it has no `Unreleased` section, so it is not Keep a Changelog ([release-please](https://github.com/googleapis/release-please), [changelog snapshot](https://github.com/googleapis/release-please/blob/main/__snapshots__/default-changelog-notes.js)).
- The Python tooling repos generate the changelog from PR labels and edit it by hand (uv and ruff with rooster, pydantic with `.github/release.yml`). SU2 forces one `changelog:*` label per PR ([require-labels.yml](https://github.com/su2code/SU2/blob/master/.github/workflows/require-labels.yml)).

Decision: release-please writes `CHANGELOG.md`. Our squash keeps only the PR title, which `AGENTS.md` already requires to read on its own, so each entry is a written summary, not a raw log. We rename the sections to Keep a Changelog words (Added, Fixed, Performance, Documentation). The release PR is where the lead edits the text before merging.

## 3. Issue and pull request templates

- Issue forms need `name`, `description` and `body`, and live in `.github/ISSUE_TEMPLATE/`. `type:` sets the issue type ([syntax](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/syntax-for-issue-forms)). This user-owned repository exposes the types Task, Bug and Feature (`GET /repos/Remenby31/autan/issue-types`).
- JAX and PyTorch require the output of an environment script and disable blank issues ([JAX bug form](https://github.com/jax-ml/jax/blob/main/.github/ISSUE_TEMPLATE/bug-report.yml), [PyTorch config](https://github.com/pytorch/pytorch/blob/main/.github/ISSUE_TEMPLATE/config.yml)). DOLFINx asks for the git commit and a minimal example ([bug_report.yml](https://github.com/FEniCS/dolfinx/blob/main/.github/ISSUE_TEMPLATE/bug_report.yml)). Kokkos asks for the generated configuration header ([bug_report.md](https://github.com/kokkos/kokkos/blob/develop/.github/ISSUE_TEMPLATE/bug_report.md)). SU2 asks for config, mesh and commands ([bug_report.md](https://github.com/su2code/SU2/blob/master/.github/ISSUE_TEMPLATE/bug_report.md)).
- None of the seven scientific codes has a form for numerical or accuracy problems. Ours asks for what makes such a report reproducible: case and mesh with their hashes, precision, backend, the expected value and its independent source, the observed value, and the observed order when a mesh family is involved. The hashes match the reproducibility rule of `AGENTS.md`.
- PR templates: uv and ruff ask for "Summary" and "Test Plan" ([uv](https://github.com/astral-sh/uv/blob/main/.github/PULL_REQUEST_TEMPLATE.md)). PyTorch adds "Included benchmark results (for PRs impacting perf)" and an "Agent details" section ([template](https://github.com/pytorch/pytorch/blob/main/.github/PULL_REQUEST_TEMPLATE.md)). Trixi.jl's review checklist requires that "a convergence test has been run and the results are posted in the PR" for new methods, and before/after timings for performance work ([review-checklist.md](https://github.com/trixi-framework/Trixi.jl/blob/main/.github/review-checklist.md)). SU2's checklist asks for no new compiler warnings and a test case ([template](https://github.com/su2code/SU2/blob/master/.github/pull_request_template.md)).

Decision: three forms (bug, numerical discrepancy, feature) and `blank_issues_enabled: false`, with a contact link to private vulnerability reporting. The PR template follows the PR rules of `AGENTS.md`: impact, then the change, then a verification section that maps each non-negotiable to a checkbox and a place for the numbers. It ends with an agent disclosure, as in PyTorch and DOLFINx ([CONTRIBUTING.md](https://github.com/FEniCS/dolfinx/blob/main/CONTRIBUTING.md)).

## 4. Ownership and labels

- `CODEOWNERS` takes `@user` or `@org/team` owners with write access; the last matching pattern wins ([About code owners](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)). PETSc maps directories to people ([.gitlab/CODEOWNERS](https://gitlab.com/petsc/petsc/-/blob/main/.gitlab/CODEOWNERS)); ruff maps crates to teams.
- Our lanes are agents, not GitHub accounts, and the lead reviews every PR. A per-path file would only repeat one name and duplicate the ownership table.

Decision: `CODEOWNERS` assigns every path to the lead, so review requests are automatic. The lane table stays in `AGENTS.md`. Split `CODEOWNERS` by path when a second human reviewer joins.

- Labels: Kokkos labels by backend and performance ([labeler.yml](https://github.com/kokkos/kokkos/blob/develop/.github/labeler.yml)); deal.II has `Performance tester` and `LLM Assisted`; Excalibur and PETSc define a triage state set.
- GitHub issue types replace "bug" and "enhancement" labels for the kind of an issue ([issue types](https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/managing-issue-types-in-an-organization)).

Decision: `.github/labels.yml` declares the lane labels, `numerics`, `performance`, `security`, `dependencies` and the triage states. A workflow applies it on push to `main` with [crazy-max/ghaction-github-labeler](https://github.com/crazy-max/ghaction-github-labeler) (MIT), without deleting labels it does not list.

## 5. Format and lint

### C++ and CUDA

- Kokkos and deal.II pin one clang-format version and say so in the file header; their Dependabot skips the clang-format mirror so the formatter never changes by surprise ([Kokkos .clang-format](https://github.com/kokkos/kokkos/blob/develop/.clang-format), [deal.II dependabot.yml](https://github.com/dealii/dealii/blob/master/.github/dependabot.yml)).
- SU2 adopted formatting late and still excludes its core solver from the hook ([.pre-commit-config.yaml](https://github.com/su2code/SU2/blob/master/.pre-commit-config.yaml)). Formatting from the first commit avoids that.
- PETSc declares its project macros in `.clang-format` (`AttributeMacros`, `StatementMacros`) ([.clang-format](https://github.com/petsc/petsc/blob/main/.clang-format)). The same applies to CUDA and HIP qualifiers.
- clang-tidy configurations start from `-*` and enable families: `bugprone`, `modernize`, `performance` (SU2, Kokkos, deal.II); deal.II adds `mpi-*` and `WarningsAsErrors: '*'` and comments the reason for each disabled check ([.clang-tidy](https://github.com/dealii/dealii/blob/master/.clang-tidy)).
- clang-tidy needs a compilation database (`CMAKE_EXPORT_COMPILE_COMMANDS`, Ninja or Makefiles only), so it runs in CI after configure, not in pre-commit ([clang-tidy](https://clang.llvm.org/extra/clang-tidy/), [CMake](https://cmake.org/cmake/help/latest/variable/CMAKE_EXPORT_COMPILE_COMMANDS.html)).
- CMake: DOLFINx checks CMake files with gersemi ([ccpp.yml](https://github.com/FEniCS/dolfinx/blob/main/.github/workflows/ccpp.yml)). cmake-format has had no commit since 2021 and is GPL-3.0; gersemi is maintained and MPL-2.0 ([gersemi](https://github.com/BlankSpruce/gersemi)).

Decision: `.clang-format` based on Google, pinned to clang-format 23.1.3, with a 100-column limit. That is the style the mesh core was already written in: on its first files, this configuration changed only lines longer than 100 columns. CUDA and HIP qualifiers are declared as attribute macros. `.clang-tidy` enables `bugprone`, `clang-analyzer`, `concurrency`, `misc`, `modernize`, `performance`, `portability` and selected `readability` checks, with a reason next to each exclusion, and treats warnings as errors. gersemi formats CMake.

### Python

- Scientific Python guide: ruff with `extend-select` `ARG, B, C4, EM, EXE, FURB, G, ICN, ISC, LOG, NPY, PD, PERF, PGH, PL, PT, PTH, PYI, Q, RET, RSE, RUF, SIM, SLOT, T10, T20, TC, TRY, UP`; ignore `PLR09` and `PLR2004` ([style guide](https://learn.scientific-python.org/development/guides/style/)). Its repo-review rules require bugbear (RF101), a target version (RF002) and the hook id `ruff-check` (PC192).
- DOLFINx bans imports of its nanobind module from user code through ruff `banned-api` ([pyproject.toml](https://github.com/FEniCS/dolfinx/blob/main/python/pyproject.toml)).

Decision: one root `ruff.toml` for `py/` and `vv/`, with the guide's rule set minus `PD` (no pandas), Python 3.12, 100 columns. Packages carry no `[tool.ruff]` table: one in `vv/pyproject.toml` would override the root file for `vv/`, and one in the root `pyproject.toml` would be ignored, since `ruff.toml` takes precedence in the same directory ([ruff configuration](https://docs.astral.sh/ruff/configuration/#config-file-discovery)). The banned import of the compiled module is added once that module exists.

### Text and repository hygiene

- Spelling: deal.II, Trixi.jl, uv and ruff use [typos](https://github.com/crate-ci/typos) (MIT/Apache-2.0), which counts for repo-review PC160; codespell is GPL-2.0.
- pre-commit-hooks: deal.II adds `check-added-large-files` and `no-commit-to-branch` ([.pre-commit-config.yaml](https://github.com/dealii/dealii/blob/master/.pre-commit-config.yaml)).
- JAX and ruff pin hooks by SHA with a `# frozen: vX` comment ([JAX](https://github.com/jax-ml/jax/blob/main/.pre-commit-config.yaml)); Dependabot's `pre-commit` ecosystem updates such pins ([Dependabot options](https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference)).
- check-jsonschema validates workflows, Dependabot config and issue forms ([check-jsonschema](https://github.com/python-jsonschema/check-jsonschema)).

Decision: pre-commit runs pre-commit-hooks, typos, ruff, clang-format, gersemi, check-jsonschema (workflows, Dependabot, issue forms, citation file), actionlint and zizmor, all pinned by SHA. The hook environments resolve their own transitive dependencies; the `uv.lock` rule of `AGENTS.md` covers the project's packages, not these pinned linters. Secrets: GitHub secret scanning with push protection is already on, so we skip gitleaks.

## 6. CI and supply chain

- Pinning and permissions: JAX, uv, ruff and pydantic pin every third-party action by full SHA with a version comment, start workflows with `permissions: {}` or read-only, and set `persist-credentials: false` ([pydantic ci.yml](https://github.com/pydantic/pydantic/blob/main/.github/workflows/ci.yml), [JAX lint.yml](https://github.com/jax-ml/jax/blob/main/.github/workflows/lint.yml)). Scorecard's Pinned-Dependencies and Token-Permissions checks score the same things ([checks.md](https://github.com/ossf/scorecard/blob/main/docs/checks.md)).
- zizmor audits workflows for template injection, excessive permissions, unpinned actions and missing Dependabot cooldown ([audits](https://docs.zizmor.sh/audits/)). Four of the five Python repos run it; the Scientific Python guide requires it (SEC001, [security guide](https://learn.scientific-python.org/development/guides/security/)).
- Scorecard: Kokkos runs the official workflow and shows the OpenSSF Best Practices badge ([scorecard.yml](https://github.com/kokkos/kokkos/blob/develop/.github/workflows/scorecard.yml)). PyTorch has the workflow but disabled. We use the official example as is ([scorecard-action example](https://github.com/ossf/scorecard-action/blob/main/examples/scorecard.yml)): `permissions: {}`, the job gets `security-events: write` and `id-token: write`, results are published and uploaded to code scanning. Its new top-level key `cache-mode` is not known to actionlint 1.7.12, so actionlint ignores that one message.
- Dependabot: JAX and pydantic set `cooldown: default-days: 7` ([JAX](https://github.com/jax-ml/jax/blob/main/.github/dependabot.yml)); repo-review GH212 asks to group action updates ([gha-basic](https://learn.scientific-python.org/development/guides/gha-basic/)). Dependabot supports `github-actions`, `pre-commit`, `uv` and `docker` ([options reference](https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference)).
- GPU CI on a public repository: PyTorch runs hardware jobs only on request through `ciflow/*` labels ([pytorch-probot.yml](https://github.com/pytorch/pytorch/blob/main/.github/pytorch-probot.yml)); JAX and PyTorch declare self-hosted runner labels in `.github/actionlint.yaml`.

Decision: `ci.yml` runs every pre-commit hook on each PR and push to `main`, with top-level `permissions: {}`, SHA-pinned actions, concurrency cancellation and a timeout. It also builds and tests the C++ code with GCC and Clang and warnings as errors, runs clang-tidy on the core and readers (Best Practices `static_analysis`, `warnings_strict`), and type-checks and tests the SDK from `uv.lock`. CodeQL analyses C/C++, Python, TypeScript and the workflows without a build (`build-mode: none`, [CodeQL build modes](https://docs.github.com/en/code-security/code-scanning/creating-an-advanced-setup-for-code-scanning/codeql-code-scanning-for-compiled-languages)), which Scorecard's SAST check counts. Scorecard runs weekly and on push to `main`. Dependabot updates actions, pre-commit hooks, the SDK's `uv.lock` and the site's `bun.lock` weekly, grouped, with a 7-day cooldown, and leaves the clang-format mirror alone. `.github/actionlint.yaml` declares the Spark runner labels from issue #4.

## 7. Versioning, releases and changelog

- SemVer 2.0.0 item 4: "Major version zero (0.y.z) is for initial development. Anything MAY change at any time" ([semver.org](https://semver.org/spec/v2.0.0.html)). uv and ruff bump the minor version for breaking changes and the patch for everything else while unstable ([ruff versioning](https://github.com/astral-sh/ruff/blob/main/docs/versioning.md)). DOLFINx bumps `0.x` only on an ABI break ([RELEASE.md](https://github.com/FEniCS/dolfinx/blob/main/RELEASE.md)).
- Conventional Commits 1.0.0: `feat` is a minor change, `fix` a patch, `!` marks a breaking change ([spec](https://www.conventionalcommits.org/en/v1.0.0/)).
- release-please: "We highly recommend that you use squash-merges". Manifest mode uses `release-please-config.json` and `.release-please-manifest.json`. `bump-minor-pre-major` makes a breaking change bump the minor below 1.0; `bump-patch-for-minor-pre-major` makes `feat` bump the patch. `extra-files` with an `x-release-please-version` marker updates other files ([release-please](https://github.com/googleapis/release-please), [customizing](https://github.com/googleapis/release-please/blob/main/docs/customizing.md)). The action needs `contents`, `issues` and `pull-requests: write`, and pull requests it opens with `GITHUB_TOKEN` trigger no workflow ([release-please-action](https://github.com/googleapis/release-please-action)).
- Pitfall: a static version in `pyproject.toml` is copied into `uv.lock`, so a release PR that bumps only `pyproject.toml` breaks `uv sync --locked`. uv leaves dynamic versions out of the lock ([uv#10622](https://github.com/astral-sh/uv/pull/10622)). This is our inference from the uv change, not a documented release-please limitation.

Decision: release-please in manifest mode, release type `simple`, both pre-major options on, first release `0.1.0` (`initial-version`; the SemVer FAQ suggests starting there, and without it release-please treats the manifest's `0.0.0` as "no release" and proposes `1.0.0`), with the version kept in one file, `version.txt`. CMake and the Python build read it, so the lock never holds a version. That needs one change each from the mesh core and SDK lanes. Its release PR gets no CI run under `GITHUB_TOKEN`; the lead reruns CI on it before merging, or we add a GitHub App token later.

## 8. Agent rules

- AGENTS.md is "a simple, open format for guiding coding agents", read by Codex, Copilot, Cursor, Gemini CLI and others ([agents.md](https://agents.md/)). Claude Code reads `CLAUDE.md`, so ours imports `@AGENTS.md`, as the Scientific Python guide and ruff do ([AI guide](https://learn.scientific-python.org/development/guides/ai/), [ruff CLAUDE.md](https://github.com/astral-sh/ruff/blob/main/CLAUDE.md)).
- JAX, PyTorch, uv and ruff forbid autonomous agents from opening PRs; deal.II forbids autonomous agents; pydantic, DOLFINx and Trixi.jl allow AI help with disclosure ([JAX](https://github.com/jax-ml/jax/blob/main/docs/contributing.md), [PyTorch AI_POLICY.md](https://github.com/pytorch/pytorch/blob/main/AI_POLICY.md), [deal.II](https://github.com/dealii/dealii/blob/master/CONTRIBUTING.md)). Their shared condition is a human who reviews and answers for the change. Here the lead reviews and merges every PR, so we allow agent-authored PRs and require them to say so.
- Rules worth taking:
  - "NEVER assume clippy warnings or test failures are pre-existing" ([uv AGENTS.md](https://github.com/astral-sh/uv/blob/main/AGENTS.md)).
  - Do not put conversation or session history in code, comments or PR text ([uv](https://github.com/astral-sh/uv/blob/main/AGENTS.md), [ruff](https://github.com/astral-sh/ruff/blob/main/AGENTS.md)).
  - Tolerances follow the precision: Trixi.jl uses `atol = 500·eps(RealT)`, `rtol = sqrt(eps(RealT))` ([TrixiTest.jl](https://github.com/trixi-framework/TrixiTest.jl/blob/main/src/macros.jl)); deal.II asks to make a test robust to round-off before adding a second reference output ([testsuite](https://github.com/dealii/dealii/blob/master/doc/developers/testsuite.html)).

Decision: `AGENTS.md` keeps every non-negotiable. It adds the hook gate (never `--no-verify`), the "no failure is pre-existing" rule, the session-history rule, precision-scaled tolerances and the agent disclosure in the PR body.

## Not adopted, and why

- **REUSE.** Spec 3.3 asks for SPDX headers on every file or a `REUSE.toml` ([spec](https://reuse.software/spec-3.3/)). It is not required by GitHub, Scorecard or the Best Practices badge, and a single-licence repository gains little. We keep the root `LICENSE` and put SPDX headers on vendored third-party files only.
- **DCO sign-off.** See [CONTRIBUTING](#contributing).
- **Fuzzing.** Scorecard credits C++ and Python only through OSS-Fuzz or ClusterFuzzLite ([checks.md](https://github.com/ossf/scorecard/blob/main/docs/checks.md#fuzzing)). The mesh readers are the natural target once they land.
- **Markdown formatter, mypy and uv-lock hooks.** Markdown formatting would rewrite the French design documents. mypy and the lock check belong with the SDK's own CI job, added when its package is on `main`.
- **OpenSSF Best Practices badge.** The files above meet the passing criteria we can meet today. Registration on bestpractices.dev needs the lead's GitHub login.

## Repository settings changed

- Private vulnerability reporting: enabled, as `SECURITY.md` requires.
- Squash commit title: "Pull request title", message blank. The default (`COMMIT_OR_PR_TITLE`) uses the commit subject for a single-commit PR ([squash settings](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/configuring-pull-request-merges/configuring-commit-squashing-for-pull-requests)), which breaks the PR-title rule.
- Workflow permission "Allow GitHub Actions to create pull requests": enabled for release-please.

## Unverified

- The formula behind the community score (7 items).
- Whether `type:` in an issue form sets the type on a user-owned repository; the API lists the types, but we have not filed an issue through a form yet.
- The release-please date token for `date-released` in `CITATION.cff`.
