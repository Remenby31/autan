# Contributing

Thank you for helping. The project is pre-alpha: open an issue before you start work, so that it fits the [plan](docs/PLAN.md).

[AGENTS.md](AGENTS.md) holds the rules for every contributor, human or AI agent: what a change must prove, how code and tests are written, who owns which directory. This page covers the mechanics.

## Issues

Pick the form that fits:

- **Bug**: the solver, the SDK or the tooling does something wrong.
- **Numerical discrepancy**: a result disagrees with an analytic solution, a manufactured solution, a conservation law, reference data or another code. Give the case, the mesh, their hashes, the precision and the backend.
- **Feature**: a capability is missing. Say how we would verify it.

Report a vulnerability privately, as [SECURITY.md](SECURITY.md) explains.

## Set up

You need Git, [uv](https://docs.astral.sh/uv/) and, for the C++ core, CMake 3.28 or later, Ninja and a C++20 compiler. Then, from the repository root:

```sh
uv tool install pre-commit
pre-commit install
```

The hook runs the formatters and linters on the files you commit: typos, ruff, clang-format 23.1.3, gersemi, check-jsonschema, actionlint and zizmor. Run them on everything with:

```sh
pre-commit run --all-files
```

CI runs the same command, so a commit that passes locally passes there.

GPU work runs on the DGX Spark, inside our containers; see [AGENTS.md](AGENTS.md#hardware).

## Pull requests

1. Work on a branch, one per issue. Before every push, run `git fetch origin && git merge origin/main`.
2. Open the PR early, as a draft. Fill in [the template](.github/pull_request_template.md): its verification section asks for the evidence that [AGENTS.md](AGENTS.md#non-negotiables) requires.
3. Write the title as a [Conventional Commit](https://www.conventionalcommits.org/en/v1.0.0/), for example `fix(io): reject Fluent zones with a negative face count`. The squash merge keeps only this title on `main`, and the changelog is built from it. Use `feat`, `fix` or `perf` for what users see; `docs`, `test`, `refactor`, `build`, `ci` or `chore` otherwise. Mark a breaking change with `!` before the colon.
4. If an AI agent wrote the change, say so in the PR body and name the tool and the model. The person who opens the PR answers for every line of it.
5. Before you mark the PR ready, get an adversarial review and answer each finding.
6. The lead reviews and merges. Do not merge your own PR.

By contributing, you agree that your contribution is licensed under the [Apache License 2.0](LICENSE), as its section 5 states.

## Releases

Once a `feat` or `fix` PR merges, [release-please](https://github.com/googleapis/release-please) opens a release PR. It bumps [version.txt](version.txt) and writes [CHANGELOG.md](CHANGELOG.md) from the merged PR titles. The lead edits the changelog in that PR and merges it to tag a release. Below 1.0.0, a breaking change bumps the minor version, and a `feat` or `fix` the patch.
