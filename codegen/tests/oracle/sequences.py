"""Declared operation sequences of ADR 0003: each reproduces, operation by operation, a piece of the
build it certifies. Its source and version are pinned in FREEZE.md; a replay that does not match the
stored result bit for bit means the build does not run the declared sequence, and the instance is
invalid (fail-closed)."""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

from oracle import certified as c

if TYPE_CHECKING:
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


# The geometry construction, replayed by exact binary64 emulation, statement by statement and in
# the C++ evaluation order: core/src/geometry.cpp (node_mean, face_geometry, cell_geometry) and
# codegen/cpp/geometry.cpp (exactly_planar, the face moments of pack_geometry, store_lsq_weights,
# inverse3). Versions pinned in FREEZE.md.
GEOMETRY_SOURCES = (
    "core/src/geometry.cpp: node_mean, face_geometry, cell_geometry",
    "codegen/cpp/geometry.cpp: exactly_planar, pack_geometry (face moments), store_lsq_weights, inverse3",
)

F = Fraction
Vec = tuple[Fraction, Fraction, Fraction]


def _a(x: F, y: F) -> F:
    return c.fl_add(x, y, c.BINARY64)


def _s(x: F, y: F) -> F:
    return c.fl_add(x, -y, c.BINARY64)


def _m(x: F, y: F) -> F:
    return c.fl_mul(x, y, c.BINARY64)


def _d(x: F, y: F) -> F:
    return c.fl_div(x, y, c.BINARY64)


def _vs(x: Vec, y: Vec) -> Vec:
    return (_s(x[0], y[0]), _s(x[1], y[1]), _s(x[2], y[2]))


def _va(x: Vec, y: Vec) -> Vec:
    return (_a(x[0], y[0]), _a(x[1], y[1]), _a(x[2], y[2]))


def _vm(k: F, x: Vec) -> Vec:
    return (_m(k, x[0]), _m(k, x[1]), _m(k, x[2]))


def _cross(x: Vec, y: Vec) -> Vec:
    return (
        _s(_m(x[1], y[2]), _m(x[2], y[1])),
        _s(_m(x[2], y[0]), _m(x[0], y[2])),
        _s(_m(x[0], y[1]), _m(x[1], y[0])),
    )


def _dot(x: Vec, y: Vec) -> F:
    return _a(_a(_m(x[0], y[0]), _m(x[1], y[1])), _m(x[2], y[2]))


ZERO: Vec = (F(0), F(0), F(0))
HALF, ONE, THREE, FOUR = F(1, 2), F(1), F(3), F(4)


def _node_mean(pts: list[Vec], nodes: list[int]) -> Vec:
    origin = pts[nodes[0]]
    total = ZERO
    for n in nodes:
        total = _va(total, _vs(pts[n], origin))
    return _va(origin, _vm(_d(ONE, F(len(nodes))), total))


def _exactly_planar(q: list[Vec]) -> bool:
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


def geometry_replay(read: ReadMesh) -> dict[str, list[list[Fraction]]]:
    """Every array the gradient kernels read, as the declared construction computes it."""
    pts: list[Vec] = [(F(float(r[0])), F(float(r[1])), F(float(r[2]))) for r in read.points]
    nf, ni, nc = len(read.owner), len(read.neighbour), read.n_cells
    faces = [
        [int(v) for v in read.face_nodes[read.face_offsets[f] : read.face_offsets[f + 1]]]
        for f in range(nf)
    ]
    area: list[Vec] = []
    magnitude: list[F] = []
    centroid: list[Vec] = []
    for nodes in faces:
        n = len(nodes)
        mean = _node_mean(pts, nodes)
        acc = ZERO
        for i in range(n):
            acc = _va(
                acc, _vm(HALF, _cross(_vs(pts[nodes[i]], mean), _vs(pts[nodes[(i + 1) % n]], mean)))
            )
        mag = F(hypot3_emulated(float(acc[0]), float(acc[1]), float(acc[2])))
        area.append(acc)
        magnitude.append(mag)
        if not mag > 0:
            centroid.append(mean)
            continue
        normal = _vm(_d(ONE, mag), acc)
        moment, weight = ZERO, F(0)
        for i in range(n):
            a = _vs(pts[nodes[i]], mean)
            b = _vs(pts[nodes[(i + 1) % n]], mean)
            w = _dot(_vm(HALF, _cross(a, b)), normal)
            moment = _va(moment, _vm(_d(w, THREE), _va(a, b)))
            weight = _a(weight, w)
        centroid.append(_va(mean, _vm(_d(ONE, weight), moment)))
    offsets = [int(v) for v in read.cell_face_offsets]
    incident = [[int(f) for f in read.cell_faces[offsets[k] : offsets[k + 1]]] for k in range(nc)]
    volume: list[F] = []
    cell_centroid: list[Vec] = []
    for cell, fs in enumerate(incident):
        apex = ZERO
        for f in fs:
            apex = _va(apex, centroid[f])
        apex = _vm(_d(ONE, F(len(fs))), apex)
        length = F(0)
        for f in fs:
            length = max(length, c.round_sqrt(magnitude[f], c.BINARY64))
        inv = _d(ONE, length)
        scaled, moment = F(0), ZERO
        for f in fs:
            sign = ONE if int(read.owner[f]) == cell else -ONE
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
    face_moment: list[list[F]] = []
    for f, nodes in enumerate(faces):
        q = [pts[n] for n in nodes]
        if _exactly_planar(q):
            face_moment.append([F(0)] * 9)
            continue
        nn = len(q)
        start = 0
        for i in range(1, nn):
            if q[i] < q[start]:
                start = i
        at = [q[(start + i) % nn] for i in range(nn + 1)]
        mean = ZERO
        for i in range(nn):
            mean = _va(mean, at[i])
        mean = (_d(mean[0], F(nn)), _d(mean[1], F(nn)), _d(mean[2], F(nn)))
        mom = [F(0)] * 9
        for i in range(nn):
            sv = _cross(_vs(at[i], mean), _vs(at[i + 1], mean))
            cv = tuple(_d(_a(_a(mean[u], at[i][u]), at[i + 1][u]), THREE) for u in range(3))
            rv = _vs(cv, centroid[f])  # type: ignore[arg-type]
            for u in range(3):
                for v in range(3):
                    mom[3 * u + v] = _a(mom[3 * u + v], _m(_m(HALF, rv[u]), sv[v]))
        face_moment.append(mom)
    lsq: list[list[F]] = []
    for cell, fs in enumerate(incident):
        xc = cell_centroid[cell]
        ds: list[Vec] = []
        for f in fs:
            if f < ni:
                other = int(read.neighbour[f]) if int(read.owner[f]) == cell else int(read.owner[f])
                ds.append(_vs(cell_centroid[other], xc))
            else:
                ds.append(_vs(centroid[f], xc))
        mat = [F(0)] * 9
        for d in ds:
            w = _d(ONE, _a(_a(_m(d[0], d[0]), _m(d[1], d[1])), _m(d[2], d[2])))
            for u in range(3):
                for v in range(3):
                    mat[3 * u + v] = _a(mat[3 * u + v], _m(_m(w, d[u]), d[v]))
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
        for d in ds:
            w = _d(ONE, _a(_a(_m(d[0], d[0]), _m(d[1], d[1])), _m(d[2], d[2])))
            lsq.append(
                [
                    _m(
                        w,
                        _a(
                            _a(_m(inv3[3 * u], d[0]), _m(inv3[3 * u + 1], d[1])),
                            _m(inv3[3 * u + 2], d[2]),
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
