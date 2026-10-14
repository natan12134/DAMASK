#!/usr/bin/env python3
"""
Generate DAMASK_grid input files for the simulations of

  X. Zhang, J. Zhao, G. Kang, M. Zaiser,
  Geometrically necessary dislocations and related kinematic hardening in gradient grained
  materials: A nonlocal crystal plasticity study,
  Int. J. Plast. 163 (2023) 103553, https://doi.org/10.1016/j.ijplas.2023.103553

Cases
-----
homogeneous : 2D RVE with 50 Voronoi grains on 40x40x1 cells (Fig. 6), grain sizes 5-80 µm,
              initial densities 1.2/2.4/3.6e12 /m² (Figs. 7-12)
gradient    : sandwich of layers with grain sizes d, 2d, 4d (d = 5 µm, Fig. 13), three
              orientation sets (Figs. 14-18)

Variants: 'full' (flux + back stress), 'noBS' (flux, no back stress), 'noFlux' (local model).
"""
import argparse
from pathlib import Path

import numpy as np
import damask

ROOT = Path(__file__).resolve().parent
STRAIN_RATE = 6.0e-4                                                                                 # 1/s
TRUE_STRAIN = 0.05
GRAIN_SIZES = [5, 10, 20, 40, 80]                                                                    # µm
RHO_0 = {'1.2': 2.5e10, '2.4': 5.0e10, '3.6': 7.5e10}                                                # per type and system
VARIANTS = ['full', 'noBS', 'noFlux']


def material(O, variant, rho0=2.5e10):
    m = damask.ConfigMaterial.load(ROOT/'material_Cu.yaml')
    pl = m['phase']['Cu']['mechanical']['plastic']
    for t in ['ed_pos', 'ed_neg', 'sc_pos', 'sc_neg']:
        pl[f'rho_u_{t}_0'] = [rho0]
    pl['flux'] = variant != 'noFlux'
    pl['short_range_stress_correction'] = variant == 'full'
    return m.material_add(O=O, phase='Cu', homogenization='SX')


def load(dt_yield, dt, n_out=60):
    """Two load steps with the same boundary conditions: fine time steps through yield, then coarser ones."""
    t_end = (np.exp(TRUE_STRAIN)-1.0)/STRAIN_RATE
    t_yield = 0.5
    bc = {'mechanical': {'F_dot': [[STRAIN_RATE, 0, 0],
                                   [0, 'x', 0],
                                   [0, 0, 0]],
                         'P':     [['x', 'x', 'x'],
                                   ['x', 0, 'x'],
                                   ['x', 'x', 'x']]}}
    N_1 = int(np.ceil(t_yield/dt_yield))
    N_2 = int(np.ceil((t_end-t_yield)/dt))
    return damask.LoadcaseGrid({'solver': {'mechanical': 'spectral_basic'},
                                'loadstep': [{'boundary_condition': bc,
                                              'discretization': {'t': t_yield, 'N': N_1},
                                              'f_out': max(1, N_1//5)},
                                             {'boundary_condition': bc,
                                              'discretization': {'t': float(t_end-t_yield), 'N': N_2},
                                              'f_out': max(1, N_2//n_out)}]})


def numerics():
    return damask.YAML({'solver': {'grid': {'N_cutback_max': 10,
                                            'mechanical': {'N_iter_max': 25,
                                                           'eps_abs_div(P)': 1.0e-4,
                                                           'eps_rel_div(P)': 5.0e-4,
                                                           'eps_abs_P': 1.0e3,
                                                           'eps_rel_P': 1.0e-3}}},
                        'phase': {'mechanical': {'plastic': {'integrator_state': 'Euler'}}}})


def homogeneous(rng_seed=20230205):
    """50-grain 2D Voronoi RVE (same topology and orientations for all grain sizes)."""
    rng = np.random.default_rng(rng_seed)
    cells = np.array([40, 40, 1])
    seeds = damask.seeds.from_random(np.ones(3), 50, cells, rng=rng)
    seeds[:, 2] = 0.5/cells[2]
    O = damask.Rotation.from_random(50, rng=rng)
    out = []
    for d in GRAIN_SIZES:
        L = d*1e-6*np.sqrt(50.0)                                                                    # equivalent square grain size d
        size = L*np.array([1.0, 1.0, 1.0/cells[0]])
        g = damask.GeomGrid.from_Voronoi_tessellation(cells, size, seeds*size)
        for rho_label, rho0 in RHO_0.items():
            for variant in VARIANTS:
                if rho_label != '1.2' and variant != 'full': continue
                if variant == 'noFlux' and d != GRAIN_SIZES[0]: continue                            # local model is size independent
                out.append((f'homog_d{d}_rho{rho_label}_{variant}', g, material(O, variant, rho0), d))
    return out


def gradient_geometry(cells_per_d):
    """
    Half sandwich of Fig. 13 (80 µm x 105 µm): layer I (3 rows of 5 µm grains), layer II (3 rows of
    10 µm grains), layer III (3 rows of 20 µm grains), mirrored at the core to obtain a periodic cell
    (I-II-III-III-II-I), i.e. a full sandwich with the two surfaces joined across the periodic boundary.
    """
    d = 5.0e-6
    nx = 16*cells_per_d
    layers = [(1, 3), (2, 3), (4, 3)]                                                               # (grain size/d, rows), from surface to core
    rows = []
    gid = 0
    for f, n_rows in layers:
        for _ in range(n_rows):
            row = np.repeat(np.arange(gid, gid+16//f), f*cells_per_d)
            rows += [row]*(f*cells_per_d)
            gid += 16//f
    lower = np.array(rows)                                                                          # (y, x), surface (layer I) at y=0
    N_half = gid
    full = np.vstack([lower, lower[::-1] + N_half])                                                 # mirrored upper half gets own grains
    material_ID = full.T[:, :, None]                                                                # (x, y, z)
    ny = material_ID.shape[1]
    size = np.array([nx, ny, 1])*d/cells_per_d
    layer_of_grain = np.concatenate([np.repeat([1, 2, 3], [48, 24, 12])]*2)
    return damask.GeomGrid(material_ID, size), 2*N_half, layer_of_grain


def gradient(cells_per_d, rng_seed=20230116):
    g, N, layer_of_grain = gradient_geometry(cells_per_d)
    rng = np.random.default_rng(rng_seed)
    out = []
    for o in range(1, 4):
        O_half = damask.Rotation.from_random(N//2, rng=rng)
        O = damask.Rotation(np.concatenate([O_half.quaternion, O_half.quaternion]))                  # mirror symmetric orientations
        for variant in VARIANTS:
            out.append((f'grad_o{o}_{variant}', g, material(O, variant), 5))
    np.savetxt(ROOT/'layer_of_grain.txt', layer_of_grain, fmt='%d')
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default=str(ROOT/'runs'), help='directory for simulation inputs')
    parser.add_argument('--cells-per-d', type=int, default=4, help='gradient model: cells per smallest grain size')
    parser.add_argument('--dt', type=float, nargs=2, default=[0.002, 0.02],
                        help='time increment (yield, main) in s for a cell size of 1 µm, scaled linearly with cell size')
    parser.add_argument('--only', default='', help='substring filter for case names')
    args = parser.parse_args()

    out = Path(args.out)
    for name, g, m, d in homogeneous() + gradient(args.cells_per_d):
        if args.only not in name: continue
        w = out/name
        w.mkdir(parents=True, exist_ok=True)
        g.save(w/'geom.vti')
        m.save(w/'material.yaml')
        numerics().save(w/'numerics.yaml')
        dx = g.size[0]/g.cells[0]/1e-6
        dt_yield, dt = [min(f*dx, 0.25) for f in args.dt]
        if name.endswith('noFlux'): dt = max(dt, 0.05)
        l = load(dt_yield, dt)
        l.save(w/'load.yaml')
        N = sum(s['discretization']['N'] for s in l['loadstep'])
        print(f'{name:30s} cells={g.cells} size={g.size*1e6} µm N={N}')
