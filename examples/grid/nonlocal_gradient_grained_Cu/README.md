# Nonlocal crystal plasticity of homogeneously and gradient grained copper

Reproduction of

> X. Zhang, J. Zhao, G. Kang, M. Zaiser,
> *Geometrically necessary dislocations and related kinematic hardening in gradient grained materials: A nonlocal crystal plasticity study*,
> Int. J. Plast. 163 (2023) 103553, https://doi.org/10.1016/j.ijplas.2023.103553

with `DAMASK_grid` and the `nonlocal` plasticity model.

## Model

The dislocation density based nonlocal model of DAMASK (Reuber et al. 2014, Kords 2013) is the basis of
the model in the paper. It was extended by the paper's additions (all optional, defaults keep the
previous behavior):

| key | paper | implementation |
|---|---|---|
| `GB_transmission: probabilistic` | sec. 3.3, eqs. 43–45 | Dislocations arriving at a grain boundary are transmitted with probability P_trans (eq. 44, C = `C_GB`) evaluated with the emitting point's total density and resolved shear stress. The critical stress (eq. 45) uses a GB energy from Read–Shockley below `theta_GB` and `E_GB` above, η = `eta_GB`, and the smallest residual Burgers vector. The outgoing system maximizes slip-trace alignment × Burgers-vector alignment among the systems of the receiving grain that operate away from the boundary (τ_eff > 0). Blocked dislocations stay at the boundary (zero flux). |
| `rate_equations: Zhang2023` | eqs. 17–20 | Dipole knock-out with (d̂ − ď) and jog length ď_s/2. |
| `short_range_stress_correction: true` | eq. 26 | Back stress from gradients of the excess (GND) densities (already in DAMASK). |
| `flux: false` | Fig. 9 | Local model (no dislocation transport). |
| output `tau_back` | Fig. 17 | Back stress per slip system. |

Numerical additions:

* `positivity_limit: true` limits the loss of a dislocation species to its available density instead of
  forcing a cutback. Without it, nearly depleted species make the time integration fail at any time increment.
* `flux_velocity_limit` (not used for the results) caps the transport velocity at the CFL limit. This is
  much faster, but the results then depend on the time increment: the flow stress at 1% strain rose by
  about 13% per halving of the increment for d = 20 µm. The cause is a few *starved* slip systems
  (mobile density ≲ 10⁹ m⁻², about 0.6% of the density) that move very fast and carry about 20–30% of the flux.
* The back-stress gradient also works for grids with a single cell along one direction.

## Parameters

Table 4 of the paper (`material_Cu.yaml`). Values not given in the paper:

| parameter | value | reason |
|---|---|---|
| Poisson ratio, shear modulus | isostrain bound of the cubic C_ij | DAMASK default |
| Peierls stress | 0.5 MPa (text, Grilli 2016) | thermally activated part negligible |
| solid solution | none (`Q_sol` → 0) | pure Cu |
| climb | none (`D_0 = 0`) | room temperature |
| `f_ed_mult` | 1 | edges and screws contribute equally (eq. 13) |
| `E_GB` | 1.0 J/m² | high-angle GB energy of Cu (Bulatov et al. 2014, simplified to Read–Shockley + constant) |
| `C_GB`, `eta_GB`, `theta_GB` | 1, 0.5, 15° | text |

## Simulations

* **Homogeneous:** 50-grain 2D Voronoi RVE, 40×40×1 cells (Fig. 6). The same seeds and orientations are used for all grain sizes (d = √(area per grain)), with a periodic cell. Plane strain is enforced through F_zz = 1, and uniaxial tension in x at 6×10⁻⁴ s⁻¹ up to 5% true strain uses P_yy = 0.
* **Gradient:** layers I/II/III with grain sizes 5/10/20 µm (3 rows each, 80 µm × 105 µm, 84 grains, Fig. 13). The model is mirrored at the core to obtain a periodic cell (the two surface layers meet across the periodic boundary), with 3 cells per 5 µm grain.

Because of the cost of the paper-exact numerics (sub-stepping to satisfy the CFL condition), a reduced
set was computed: see the result section below for what is included.

## Usage

```bash
python generate.py --out runs --cells-per-d 3      # inputs for all cases
N_JOBS=4 ./run_all.sh runs                         # run (DAMASK_GRID=/path/to/damask_grid)
python postprocess.py --runs runs --out data       # reduce to data/*.npz
python plot_figures.py --data data --figs figures  # figures
```
