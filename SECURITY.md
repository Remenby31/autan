# Security policy

## Report a vulnerability

Report it privately through GitHub: [open a draft security advisory](https://github.com/Remenby31/cfd-solver/security/advisories/new). Do not open a public issue.

Include the commit you tested, the input that triggers the problem (a mesh or case file, or a script that writes one) and what an attacker gains.

We acknowledge a report within 5 business days and publish a fix and an advisory within 90 days of the report. We credit the reporter unless asked not to.

## Supported versions

The project is pre-alpha and has no release. We fix vulnerabilities on `main` only.

## Scope

The solver reads mesh, case and results files that a user may receive from someone else. In scope:

- memory-safety bugs (out-of-bounds access, use after free, integer overflow in sizes) in the mesh, case and results readers, the Python bindings and the GPU kernels;
- a file or case that makes the solver read or write outside the paths it was given, or run code;
- a vulnerability in our GitHub workflows, such as a pull request that can obtain a secret or write to the repository.

Out of scope:

- a wrong or inaccurate numerical result: use the [numerical discrepancy form](https://github.com/Remenby31/cfd-solver/issues/new?template=2-numerical-discrepancy.yml);
- a crash or a coded error on invalid input that reads no memory it should not: use the [bug form](https://github.com/Remenby31/cfd-solver/issues/new?template=1-bug.yml);
- running out of memory or time on a mesh that is simply large;
- vulnerabilities in a dependency that our code does not reach. Report those upstream.
