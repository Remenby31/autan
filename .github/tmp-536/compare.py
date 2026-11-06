"""#536: which arrays of the two dumps differ, and by how many ulps."""

import sys

import numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
for key in a.files:
    x, y = a[key], b[key]
    xs = x.view(np.float64) if x.dtype == np.complex128 else x
    ys = y.view(np.float64) if y.dtype == np.complex128 else y
    differ = int((xs != ys).sum())
    ulp = 0
    if differ:
        xi, yi = xs.astype(np.float64).view(np.int64), ys.astype(np.float64).view(np.int64)
        ulp = int(np.abs(xi - yi).max())
    sys.stdout.write(f"{key:20} {x.shape!s:14} differ {differ:7} of {xs.size:7}  max {ulp} ulp\n")
