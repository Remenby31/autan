"""#536: sha256 of NACA 0012 level 32 (points and written file) and the time of each level."""

import hashlib
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from autan_vv.meshes import naca0012

out = Path(sys.argv[1])
lines = [f"platform {platform.machine()} {platform.system()} numpy {np.__version__}"]
with tempfile.TemporaryDirectory() as tmp:
    for nc in (int(v) for v in sys.argv[2:]):
        start = time.perf_counter()
        points, _, _ = naca0012.grid(nc)
        path = Path(tmp) / f"n{nc}.msh"
        naca0012.generate_level(nc, path)
        seconds = time.perf_counter() - start
        sha_points = hashlib.sha256(np.ascontiguousarray(points).tobytes()).hexdigest()
        sha_file = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"nc {nc} points {sha_points} file {sha_file} seconds {seconds:.1f}")
out.write_text("\n".join(lines) + "\n")
sys.stdout.write(out.read_text())
