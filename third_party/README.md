# Vendored header-only libraries

Every build needs these two libraries, so they live here instead of being downloaded at configure
time: a build works when GitHub's release downloads do not. Their bytes are upstream's, unchanged;
`SHA256SUMS` holds their digests, and a pre-commit hook (`third-party sums`) checks them. No other
hook touches this directory.

| Files | Library | Version | Source | License |
|---|---|---|---|---|
| `nlohmann_json/include/nlohmann/json.hpp` | [nlohmann/json](https://github.com/nlohmann/json) | 3.12.0 | `single_include/nlohmann/json.hpp` of the release archive `json.tar.xz`, SHA-256 `42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa` | MIT (`nlohmann_json/LICENSE.MIT`) |
| `picosha2/picosha2.h` | [PicoSHA2](https://github.com/okdshin/PicoSHA2) | commit `161cb3fc4170fa7a3eca9e582cebd27cc4d1fe29` | `picosha2.h` of that commit's archive, SHA-256 `6cf473a00c98298d3ddee0aed853e3c799791f49dbc01c996ea46cb248e85802` | MIT (`picosha2/LICENSE`) |

To update one: download its archive, check the archive's SHA-256, copy the files above unchanged,
and update this table and `SHA256SUMS`.
