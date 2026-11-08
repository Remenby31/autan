"""Declared operation sequences of ADR 0003: each reproduces, operation by operation, a piece of the
build it certifies. Its source and version are pinned in FREEZE.md; a replay that does not match the
stored result bit for bit means the build does not run the declared sequence, and the instance is
invalid (fail-closed)."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any

from oracle import certified as c

if TYPE_CHECKING:
    import autan_spike as s

    from oracle.topology import ReadMesh

# libstdc++ of GCC 13 (the autan-dev image), /usr/include/c++/13/cmath:3550-3563:
#   __x = abs(__x); __y = abs(__y); __z = abs(__z);
#   if (_Tp __a = __x < __y ? __y < __z ? __z : __y : __x < __z ? __z : __x)
#     return __a * sqrt((__x / __a) * (__x / __a) + (__y / __a) * (__y / __a)
#                       + (__z / __a) * (__z / __a));
#   else return {};
HYPOT3_SOURCE = "GCC 13 libstdc++ /usr/include/c++/13/cmath:3550-3563 (std::__hypot3)"


def _pick(x: Fraction, y: Fraction, z: Fraction) -> int:
    """Index of __a among (x, y, z), by the source's comparisons."""
    if x < y:
        return 2 if y < z else 1
    return 2 if x < z else 0


def hypot3_emulated(x: float, y: float, z: float) -> float:
    """std::hypot(x, y, z) of the build, replayed exactly in binary64."""
    f = c.BINARY64
    v = [abs(Fraction(t)) for t in (x, y, z)]
    a = v[_pick(*v)]
    if a == 0:
        return 0.0
    squares = [c.fl_mul(q, q, f) for q in (c.fl_div(t, a, f) for t in v)]
    total = c.fl_add(c.fl_add(squares[0], squares[1], f), squares[2], f)
    return float(c.fl_mul(a, c.round_sqrt(total, f), f))


def hypot3_cert(
    args: tuple[c.Cert, c.Cert, c.Cert], executed: tuple[float, float, float], stored: float
) -> c.Cert:
    """The certified magnitude of a face: the stored value must be the declared sequence of the
    executed arguments bit for bit (else Invalid); its radius around the exact root sqrt(sum X^2)
    comes from the rules on that sequence. The exact identity a sqrt(sum (X/a)^2) = sqrt(sum X^2)
    holds for any a > 0, so the executed branch need only certify a > 0."""
    if hypot3_emulated(*executed) != stored:
        msg = "hypot: the stored magnitude is not the declared sequence of its arguments"
        raise c.Invalid(msg)
    f = c.BINARY64
    for t, cert in zip(executed, args, strict=True):
        if abs(Fraction(t) - cert.value) > cert.radius:
            msg = "hypot: an executed argument lies outside its certificate"
            raise c.Invalid(msg)
    absolute = [c.Cert(abs(cert.value), cert.radius, f) for cert in args]
    k = _pick(*(abs(Fraction(t)) for t in executed))
    a = absolute[k]
    if abs(Fraction(executed[k])) == 0:
        if any(cert.value != 0 or cert.radius != 0 for cert in args):
            msg = "hypot: a zero face whose exact area is not certainly zero"
            raise c.Invalid(msg)
        return c.Cert(Fraction(0), Fraction(0), f)
    if a.value - a.radius <= 0:
        msg = "hypot: the scale a is not certainly positive"
        raise c.Invalid(msg)
    squares = [c.mul(q, q, f) for q in (c.div(t, a, f) for t in absolute)]
    total = c.add(c.add(squares[0], squares[1], f), squares[2], f)
    return c.mul(a, c.sqrt(total, f), f)


# The geometry construction, statement by statement in the C++ evaluation order:
# core/src/geometry.cpp (node_mean, face_geometry, cell_geometry) and codegen/cpp/geometry.cpp
# (exactly_planar, the face moments of pack_geometry, store_lsq_weights, inverse3). It runs once over
# pairs: the executed value, by exact binary64 emulation (which must equal the stored arrays bit for
# bit), and its certificate, by the rules. Branches follow the executed values; the division and
# square-root rules refuse any denominator not separated from zero. Versions pinned in FREEZE.md.
GEOMETRY_SOURCES = (
    "core/src/geometry.cpp: node_mean, face_geometry, cell_geometry",
    "codegen/cpp/geometry.cpp: exactly_planar, pack_geometry (face moments), store_lsq_weights, inverse3",
)

F = Fraction
B = c.BINARY64


@dataclass(frozen=True)
class Pair:
    """An executed binary64 value and its certificate."""

    e: Fraction
    c: c.Cert


Vec = tuple[Pair, Pair, Pair]


def _k(x: Fraction) -> Pair:
    return Pair(x, c.Cert(x, F(0), B))


def _a(x: Pair, y: Pair) -> Pair:
    return Pair(c.fl_add(x.e, y.e, B), c.add(x.c, y.c, B))


def _s(x: Pair, y: Pair) -> Pair:
    return Pair(c.fl_add(x.e, -y.e, B), c.sub(x.c, y.c, B))


def _m(x: Pair, y: Pair) -> Pair:
    return Pair(c.fl_mul(x.e, y.e, B), c.mul(x.c, y.c, B))


def _d(x: Pair, y: Pair) -> Pair:
    return Pair(c.fl_div(x.e, y.e, B), c.div(x.c, y.c, B))


def _sqrt(x: Pair) -> Pair:
    return Pair(c.round_sqrt(x.e, B), c.sqrt(x.c, B))


def _hypot(v: Vec) -> Pair:
    executed = (float(v[0].e), float(v[1].e), float(v[2].e))
    h = hypot3_emulated(*executed)
    return Pair(F(h), hypot3_cert((v[0].c, v[1].c, v[2].c), executed, h))


def _vs(x: Vec, y: Vec) -> Vec:
    return (_s(x[0], y[0]), _s(x[1], y[1]), _s(x[2], y[2]))


def _va(x: Vec, y: Vec) -> Vec:
    return (_a(x[0], y[0]), _a(x[1], y[1]), _a(x[2], y[2]))


def _vm(k: Pair, x: Vec) -> Vec:
    return (_m(k, x[0]), _m(k, x[1]), _m(k, x[2]))


def _cross(x: Vec, y: Vec) -> Vec:
    return (
        _s(_m(x[1], y[2]), _m(x[2], y[1])),
        _s(_m(x[2], y[0]), _m(x[0], y[2])),
        _s(_m(x[0], y[1]), _m(x[1], y[0])),
    )


def _dot(x: Vec, y: Vec) -> Pair:
    return _a(_a(_m(x[0], y[0]), _m(x[1], y[1])), _m(x[2], y[2]))


ZERO, HALF, ONE, THREE, FOUR = _k(F(0)), _k(F(1, 2)), _k(F(1)), _k(F(3)), _k(F(4))
ZV: Vec = (ZERO, ZERO, ZERO)


def _node_mean(pts: list[Vec], nodes: list[int]) -> Vec:
    origin = pts[nodes[0]]
    total = ZV
    for n in nodes:
        total = _va(total, _vs(pts[n], origin))
    return _va(origin, _vm(_d(ONE, _k(F(len(nodes)))), total))


def _exactly_planar(q: list[tuple[Fraction, Fraction, Fraction]]) -> bool:
    nn = len(q)
    if nn == 3:
        return True
    p0 = q[0]
    for a in range(3):
        b, cc = (a + 1) % 3, (a + 2) % 3
        shared, two_lines, second = True, True, 0
        i = 1
        while i < nn and (shared or two_lines):
            r = q[i]
            shared = shared and r[a] == p0[a]
            if r[b] == p0[b] and r[cc] == p0[cc]:
                i += 1
                continue
            if second == 0:
                second = i
            else:
                sv = q[second]
                two_lines = two_lines and r[b] == sv[b] and r[cc] == sv[cc]
            i += 1
        if shared or two_lines:
            return True
    return False


def _faces(read: ReadMesh) -> list[list[int]]:
    return [
        [int(v) for v in read.face_nodes[read.face_offsets[f] : read.face_offsets[f + 1]]]
        for f in range(len(read.owner))
    ]


def _incident(read: ReadMesh) -> list[list[int]]:
    o = [int(v) for v in read.cell_face_offsets]
    return [[int(f) for f in read.cell_faces[o[k] : o[k + 1]]] for k in range(read.n_cells)]


def construct(read: ReadMesh) -> dict[str, list[list[Pair]]]:
    pts: list[Vec] = [
        (_k(F(float(r[0]))), _k(F(float(r[1]))), _k(F(float(r[2])))) for r in read.points
    ]
    faces, incident, ni = _faces(read), _incident(read), len(read.neighbour)
    area: list[Vec] = []
    magnitude: list[Pair] = []
    centroid: list[Vec] = []
    for nodes in faces:
        n = len(nodes)
        mean = _node_mean(pts, nodes)
        acc = ZV
        for i in range(n):
            acc = _va(
                acc, _vm(HALF, _cross(_vs(pts[nodes[i]], mean), _vs(pts[nodes[(i + 1) % n]], mean)))
            )
        mag = _hypot(acc)
        if not mag.e > 0:
            msg = "a face of zero area: its centroid is not defined"
            raise c.Invalid(msg)
        area.append(acc)
        magnitude.append(mag)
        normal = _vm(_d(ONE, mag), acc)
        moment, weight = ZV, ZERO
        for i in range(n):
            a = _vs(pts[nodes[i]], mean)
            b = _vs(pts[nodes[(i + 1) % n]], mean)
            w = _dot(_vm(HALF, _cross(a, b)), normal)
            moment = _va(moment, _vm(_d(w, THREE), _va(a, b)))
            weight = _a(weight, w)
        centroid.append(_va(mean, _vm(_d(ONE, weight), moment)))
    volume: list[Pair] = []
    cell_centroid: list[Vec] = []
    for cell, fs in enumerate(incident):
        apex = ZV
        for f in fs:
            apex = _va(apex, centroid[f])
        apex = _vm(_d(ONE, _k(F(len(fs)))), apex)
        length = ZERO
        for f in fs:
            root = _sqrt(magnitude[f])
            if length.e < root.e:  # std::max(length, root)
                length = root
        inv = _d(ONE, length)
        scaled, moment = ZERO, ZV
        for f in fs:
            sign = ONE if int(read.owner[f]) == cell else _k(F(-1))
            nodes = faces[f]
            n = len(nodes)
            mm = _vm(inv, _vs(_node_mean(pts, nodes), apex))
            for i in range(n):
                a = _vm(inv, _vs(pts[nodes[i]], apex))
                b = _vm(inv, _vs(pts[nodes[(i + 1) % n]], apex))
                tet = _d(_m(sign, _dot(_vm(HALF, _cross(_vs(a, mm), _vs(b, mm))), mm)), THREE)
                scaled = _a(scaled, tet)
                moment = _va(moment, _vm(_d(tet, FOUR), _va(_va(mm, a), b)))
        volume.append(_m(_m(_m(scaled, length), length), length))
        cell_centroid.append(_va(apex, _vm(_d(length, scaled), moment)))
    face_moment: list[list[Pair]] = []
    for f, nodes in enumerate(faces):
        q = [pts[n] for n in nodes]
        if _exactly_planar([(v[0].e, v[1].e, v[2].e) for v in q]):
            face_moment.append([ZERO] * 9)
            continue
        nn = len(q)
        start = 0
        for i in range(1, nn):
            if tuple(v.e for v in q[i]) < tuple(v.e for v in q[start]):
                start = i
        at = [q[(start + i) % nn] for i in range(nn + 1)]
        mean = ZV
        for i in range(nn):
            mean = _va(mean, at[i])
        count = _k(F(nn))
        mean = (_d(mean[0], count), _d(mean[1], count), _d(mean[2], count))
        mom = [ZERO] * 9
        for i in range(nn):
            sv = _cross(_vs(at[i], mean), _vs(at[i + 1], mean))
            cv = (
                _d(_a(_a(mean[0], at[i][0]), at[i + 1][0]), THREE),
                _d(_a(_a(mean[1], at[i][1]), at[i + 1][1]), THREE),
                _d(_a(_a(mean[2], at[i][2]), at[i + 1][2]), THREE),
            )
            rv = _vs(cv, centroid[f])
            for u in range(3):
                for v in range(3):
                    mom[3 * u + v] = _a(mom[3 * u + v], _m(_m(HALF, rv[u]), sv[v]))
        face_moment.append(mom)
    lsq: list[list[Pair]] = []
    for cell, fs in enumerate(incident):
        xc = cell_centroid[cell]
        ds: list[Vec] = []
        for f in fs:
            if f < ni:
                other = int(read.neighbour[f]) if int(read.owner[f]) == cell else int(read.owner[f])
                ds.append(_vs(cell_centroid[other], xc))
            else:
                ds.append(_vs(centroid[f], xc))
        mat = [ZERO] * 9
        for dv in ds:
            w = _d(ONE, _dot(dv, dv))
            for u in range(3):
                for v in range(3):
                    mat[3 * u + v] = _a(mat[3 * u + v], _m(_m(w, dv[u]), dv[v]))
        m0, m1, m2, m3, m4, m5, m6, m7, m8 = mat
        c00 = _s(_m(m4, m8), _m(m5, m7))
        c01 = _s(_m(m5, m6), _m(m3, m8))
        c02 = _s(_m(m3, m7), _m(m4, m6))
        sc = _d(ONE, _a(_a(_m(m0, c00), _m(m1, c01)), _m(m2, c02)))
        inv3 = [
            _m(c00, sc), _m(_s(_m(m2, m7), _m(m1, m8)), sc), _m(_s(_m(m1, m5), _m(m2, m4)), sc),
            _m(c01, sc), _m(_s(_m(m0, m8), _m(m2, m6)), sc), _m(_s(_m(m2, m3), _m(m0, m5)), sc),
            _m(c02, sc), _m(_s(_m(m1, m6), _m(m0, m7)), sc), _m(_s(_m(m0, m4), _m(m1, m3)), sc),
        ]  # fmt: skip
        for dv in ds:
            w = _d(ONE, _dot(dv, dv))
            lsq.append(
                [
                    _m(
                        w,
                        _a(
                            _a(_m(inv3[3 * u], dv[0]), _m(inv3[3 * u + 1], dv[1])),
                            _m(inv3[3 * u + 2], dv[2]),
                        ),
                    )
                    for u in range(3)
                ]
            )
    return {
        "face_area": [list(v) for v in area],
        "face_centroid": [list(v) for v in centroid],
        "face_moment": face_moment,
        "cell_centroid": [list(v) for v in cell_centroid],
        "cell_volume": [[v] for v in volume],
        "lsq_weight": lsq,
    }


def geometry_replay(read: ReadMesh) -> dict[str, list[list[Fraction]]]:
    """Every array the gradient kernels read, as the declared construction executes it."""
    return {k: [[p.e for p in row] for row in rows] for k, rows in construct(read).items()}


def geometry_certified(read: ReadMesh) -> dict[str, list[list[c.Cert]]]:
    """The certificate of every array the gradient kernels read."""
    return {k: [[p.c for p in row] for row in rows] for k, rows in construct(read).items()}


# The intended geometry, evaluated exactly from the nodes, independently of the declared sequence:
# the contract's definitions in rationals, with the fan weights left unnormalised and no length
# scaling (the normal's magnitude and the length scale cancel exactly).

RVec = tuple[Fraction, Fraction, Fraction]


def _rs(x: RVec, y: RVec) -> RVec:
    return (x[0] - y[0], x[1] - y[1], x[2] - y[2])


def _rcross(x: RVec, y: RVec) -> RVec:
    return (x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0])


def _rdot(x: RVec, y: RVec) -> Fraction:
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2]


def geometry_exact(read: ReadMesh) -> dict[str, list[list[Fraction]]]:
    pts: list[RVec] = [(F(float(r[0])), F(float(r[1])), F(float(r[2]))) for r in read.points]
    faces, incident, ni = _faces(read), _incident(read), len(read.neighbour)

    def mean_of(nodes: list[int]) -> RVec:
        return (
            sum((pts[n][0] for n in nodes), F(0)) / len(nodes),
            sum((pts[n][1] for n in nodes), F(0)) / len(nodes),
            sum((pts[n][2] for n in nodes), F(0)) / len(nodes),
        )

    area: list[RVec] = []
    centroid: list[RVec] = []
    for nodes in faces:
        n, m = len(nodes), mean_of(nodes)
        tris = [_rcross(_rs(pts[nodes[i]], m), _rs(pts[nodes[(i + 1) % n]], m)) for i in range(n)]
        s_vec = (
            sum((t[0] for t in tris), F(0)) / 2,
            sum((t[1] for t in tris), F(0)) / 2,
            sum((t[2] for t in tris), F(0)) / 2,
        )
        weights = [_rdot(t, s_vec) / 2 for t in tris]
        total = sum(weights, F(0))
        if total == 0:
            msg = "a face whose fan weights sum to zero"
            raise c.Invalid(msg)
        mom = [F(0), F(0), F(0)]
        for i, w in enumerate(weights):
            a, b = _rs(pts[nodes[i]], m), _rs(pts[nodes[(i + 1) % n]], m)
            for u in range(3):
                mom[u] += w / 3 * (a[u] + b[u])
        area.append(s_vec)
        centroid.append((m[0] + mom[0] / total, m[1] + mom[1] / total, m[2] + mom[2] / total))
    volume: list[Fraction] = []
    cell_centroid: list[RVec] = []
    for cell, fs in enumerate(incident):
        apex = (
            sum((centroid[f][0] for f in fs), F(0)) / len(fs),
            sum((centroid[f][1] for f in fs), F(0)) / len(fs),
            sum((centroid[f][2] for f in fs), F(0)) / len(fs),
        )
        vol, mom = F(0), [F(0), F(0), F(0)]
        for f in fs:
            sign = 1 if int(read.owner[f]) == cell else -1
            nodes = faces[f]
            n = len(nodes)
            mm = _rs(mean_of(nodes), apex)
            for i in range(n):
                a, b = _rs(pts[nodes[i]], apex), _rs(pts[nodes[(i + 1) % n]], apex)
                tet = sign * _rdot(_rcross(_rs(a, mm), _rs(b, mm)), mm) / 6
                vol += tet
                for u in range(3):
                    mom[u] += tet / 4 * (mm[u] + a[u] + b[u])
        volume.append(vol)
        cell_centroid.append(
            (apex[0] + mom[0] / vol, apex[1] + mom[1] / vol, apex[2] + mom[2] / vol)
        )
    face_moment: list[list[Fraction]] = []
    for f, nodes in enumerate(faces):
        q = [pts[n] for n in nodes]
        if _exactly_planar(q):
            face_moment.append([F(0)] * 9)
            continue
        n, m = len(q), mean_of(nodes)
        mom9 = [F(0)] * 9
        for i in range(n):
            a, b = q[i], q[(i + 1) % n]
            sv = _rcross(_rs(a, m), _rs(b, m))
            r = _rs(
                ((m[0] + a[0] + b[0]) / 3, (m[1] + a[1] + b[1]) / 3, (m[2] + a[2] + b[2]) / 3),
                centroid[f],
            )
            for u in range(3):
                for v in range(3):
                    mom9[3 * u + v] += r[u] * sv[v] / 2
        face_moment.append(mom9)
    lsq: list[list[Fraction]] = []
    for cell, fs in enumerate(incident):
        xc = cell_centroid[cell]
        ds = []
        for f in fs:
            if f < ni:
                other = int(read.neighbour[f]) if int(read.owner[f]) == cell else int(read.owner[f])
                ds.append(_rs(cell_centroid[other], xc))
            else:
                ds.append(_rs(centroid[f], xc))
        mat = [
            [sum((d[u] * d[v] / _rdot(d, d) for d in ds), F(0)) for v in range(3)] for u in range(3)
        ]
        det = (
            mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1])
            - mat[0][1] * (mat[1][0] * mat[2][2] - mat[1][2] * mat[2][0])
            + mat[0][2] * (mat[1][0] * mat[2][1] - mat[1][1] * mat[2][0])
        )
        if det == 0:
            msg = f"cell {cell}: a singular least-squares matrix"
            raise c.Invalid(msg)
        adj = [
            [
                (
                    mat[(j + 1) % 3][(i + 1) % 3] * mat[(j + 2) % 3][(i + 2) % 3]
                    - mat[(j + 1) % 3][(i + 2) % 3] * mat[(j + 2) % 3][(i + 1) % 3]
                )
                for j in range(3)
            ]
            for i in range(3)
        ]
        for d in ds:
            dd = _rdot(d, d)
            lsq.append(
                [sum((adj[u][v] * d[v] for v in range(3)), F(0)) / det / dd for u in range(3)]
            )
    return {
        "face_area": [list(v) for v in area],
        "face_centroid": [list(v) for v in centroid],
        "face_moment": face_moment,
        "cell_centroid": [list(v) for v in cell_centroid],
        "cell_volume": [[v] for v in volume],
        "lsq_weight": lsq,
    }


def stored_geometry(m: s.Mesh) -> dict[str, list[list[float]]]:
    """The arrays the gradient kernels read, as stored."""
    return {
        "face_area": [[float(v) for v in row] for row in m.face_area],
        "face_centroid": [[float(v) for v in row] for row in m.face_centroid],
        "face_moment": [[float(v) for v in row] for row in m.face_moment.reshape(m.n_faces, 9)],
        "cell_centroid": [[float(v) for v in row] for row in m.cell_centroid],
        "cell_volume": [[float(v)] for v in m.cell_volume],
        "lsq_weight": [[float(v) for v in row] for row in m.lsq_weight],
    }


def check_geometry(
    read: ReadMesh,
    stored: dict[str, Any],
    certs: dict[str, list[list[c.Cert]]] | None = None,
    exact: dict[str, list[list[Fraction]]] | None = None,
) -> list[str]:
    """Check 2: for every value, the exact (intended) value and the stored value both lie in the
    certified interval [value - R, value + R] of the declared construction. The first is the
    per-instance correspondence; the second bounds the stored value's distance to the intended.
    `certs` and `exact` may be given, as built once for the instance; else they are built here."""
    if certs is None:
        certs = geometry_certified(read)
    if exact is None:
        exact = geometry_exact(read)
    issues: list[str] = []
    # Given or built, the certificates and the exact geometry must cover exactly the stored arrays,
    # with the same shapes: an empty or partial set would check less, silently.
    for label, given in (("certified", certs), ("exact", exact)):
        if set(given) != set(stored):
            issues.append(
                f"invalid: the {label} geometry covers {sorted(given)}, the stored arrays are {sorted(stored)}"
            )
            return issues
        for name, stored_rows in stored.items():
            shape = [len(row) for row in given[name]]
            if shape != [len(row) for row in stored_rows]:
                issues.append(f"invalid: the {label} {name} does not have the stored shape")
                return issues
    for name, rows in certs.items():
        if len(stored[name]) != len(rows):
            issues.append(f"{name}: {len(stored[name])} rows stored, {len(rows)} certified")
            continue
        for i, row in enumerate(rows):
            for j, cert in enumerate(row):
                if abs(exact[name][i][j] - cert.value) > cert.radius:
                    issues.append(
                        f"{name}[{i}][{j}]: the exact value lies outside the certified interval"
                    )
                if abs(F(float(stored[name][i][j])) - cert.value) > cert.radius:
                    issues.append(
                        f"{name}[{i}][{j}]: the stored value lies outside its certified interval"
                    )
    return issues
