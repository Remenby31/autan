# Reference cases

Source: `vv/src/autan_vv/cases/` and its catalogue `catalog.json` (schema `autan.case-library/1`). The wider survey of datasets and their licences is [docs/research/validation-datasets.md](../research/validation-datasets.md); this page covers what the code fetches.

```sh
uv run autan-vv cases list
uv run autan-vv cases fetch tmr-2dzp-flatplate      # into ~/.cache/autan-vv/<case>/
uv run pytest -m network                            # fetch, verify and parse every case
```

## How files are pinned

Every file has a URL, a size and a SHA-256. TMR data files use `raw.githubusercontent.com/TMBWG/turbmodels/<commit>/…` at commit `20a39f5`, which is immutable; the TMR site moved from turbmodels.larc.nasa.gov to [tmbwg.github.io/turbmodels](https://tmbwg.github.io/turbmodels/) and its old links now redirect to a NASA notice. TMR grids exist only inside the zip archives on nasa.gov (`wp-content/uploads/2026/02/…`); the catalogue pins each archive and each member's own SHA-256. Archives are downloaded once into `blobs/` by checksum. A mismatch raises `V0601` (download) or `V0602` (member) and leaves nothing behind; there is no mirror fallback.

## Cases

| Case | Content | Parsers |
|---|---|---|
| `tmr-2dzp-flatplate` | 5 nested PLOT3D grids (35×25 … 545×385, 16 MB zip); CFL3D and FUN3D SA: cf along the plate, cf(0.97) and CD per grid, profiles at x = 0.97 | PLOT3D 2D, Tecplot |
| `tmr-2dbc-bump` | 5 grids (89×41 … 1409×641, 166 MB zip); CFL3D and FUN3D SA: cp, cf, forces per grid, profiles at x = 0.75 | PLOT3D 2D, Tecplot |
| `tmr-2dn00-naca0012` | 5 grids (113×33 … 1793×513, 110 MB zip) and the exact airfoil points; CFL3D SA cp, cf, CL, CD at α = 0, 10, 15°; experiments of Ladson, Gregory and O'Reilly, Abbott and von Doenhoff, McCroskey | PLOT3D 2D, Tecplot |
| `tmr-2dn00-naca0012-numerics` | CFL3D and FUN3D SA forces on grid families I–III at α = 10° (Diskin et al. 2015), cp and cf | Tecplot |
| `onera-m6` | Schmitt and Charpin pressure at 7 span stations, runs 2308 (M 0.8395, α 3.06°) and 2565 (M 0.8372, α 6.06°), from TMR and NPARC (NPARC `.ex` files store −Cp); section profiles | Tecplot, columns |
| `rae2822` | Geometry; case 6 Cp (NPARC); cases 1, 6, 7, 9, 12 conditions, forces, Cp, cf from the 1981 AFOSR-HTTM-Stanford conference tape | columns, 1981 tape |

Pointers only (too large to fetch): HLPW-5 (CRM-HL, finest grids 2×10⁸–10⁹ cells, archive 1.75 GB at aiaa-hlpw.org) and DPW-7/DPW-8 (dpw.larc.nasa.gov; DPW-7 Vassberg grids 5.3–221 M points), in `pointers()`.

**RAE 2822 case 10** has no reachable digital source (the AGARD AR-138 PDF times out at sto.nato.int and DTIC refuses); it is not in the catalogue. The NWTF/Airbus RAE 2822 dataset is CC BY-NC 4.0, refused by our licence policy.

## Verification

Offline tests check the checksum and member errors through `file://` URLs and each parser on snippets written from its format description. Network tests (`-m network`) fetch every file of every case, verify it, parse it, check grid dimensions against the file names, compare the values transcribed in the catalogue from the TMR pages with the downloaded data files (flat plate cf and CD for CFL3D and FUN3D, NACA 0012 CL and CD), and decode RAE 2822 case 9 from the 1981 tape to its published conditions (M = 0.73, α = 3.19°, corrected 2.79°, Re = 6.5×10⁶).

## Licences

TMR repository: CC0-1.0. nasa.gov archives and NPARC (grc.nasa.gov) pages: US Government works with no stated restriction. Nothing is vendored: files are fetched by hash at use time.

## References

- C. L. Rumsey, B. R. Smith and G. P. Huang, Description of a website resource for turbulence modeling verification and validation, AIAA 2010-4742. doi:10.2514/6.2010-4742
- B. Diskin, J. L. Thomas, C. L. Rumsey and A. Schwöppe, Grid convergence for turbulent flows, AIAA 2015-1746; *AIAA J.* 54(9):2563–2588, 2016. doi:10.2514/1.J054555
- V. Schmitt and F. Charpin, Pressure distributions on the ONERA-M6-wing at transonic Mach numbers, AGARD AR-138, 1979.
- P. H. Cook, M. A. McDonald and M. C. P. Firmin, Aerofoil RAE 2822 — pressure distributions, and boundary layer and wake measurements, AGARD AR-138, 1979.
- C. L. Ladson, NASA TM-4074, 1988; N. Gregory and C. L. O'Reilly, ARC R&M 3726, 1970; W. J. McCroskey, NASA TM-100019, 1987; I. H. Abbott and A. E. von Doenhoff, *Theory of Wing Sections*, Dover, 1959.
