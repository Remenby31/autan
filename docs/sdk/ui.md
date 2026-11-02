# `autan ui`

`autan ui` serves the web UI for the run directories under `DIR` (default `./runs`, where `autan run` writes) on `127.0.0.1:8765`, and opens it in a browser. With `--meshes DIR`, the case editor plans cases against the `*.meshinfo.json` files of that directory; without it, it lists no mesh.

```text
autan ui [--runs DIR] [--port N] [--no-open] [--meshes DIR]
```

It needs the `ui` extra (`uv add 'autan[ui]'`); without it the command exits with `E0805`.
