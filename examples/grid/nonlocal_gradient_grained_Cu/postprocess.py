#!/usr/bin/env python3
"""
Reduce DAMASK result files of the Zhang et al. (2023) reproduction to compact NumPy archives.

For every case directory containing 'job.hdf5' a file 'data/<case>.npz' is written with
 - curves: true strain, true stress (xx), and volume averages of total/GND density
 - final fields (last increment): total density, GND density, von Mises stress, back stress
 - for gradient cases additionally the layer-wise evolution of densities and von Mises stress
"""
import argparse
from pathlib import Path

import numpy as np
import damask

ROOT = Path(__file__).resolve().parent
TYPES = ['ed_pos', 'ed_neg', 'sc_pos', 'sc_neg']


def densities(r):
    """Total and GND density per cell (summed over slip systems) for all increments."""
    u = {t: r.get(f'rho_u_{t}', flatten=False) for t in TYPES}
    b = {t: r.get(f'rho_b_{t}', flatten=False) for t in TYPES}
    d_ed = r.get('rho_d_ed', flatten=False)
    d_sc = r.get('rho_d_sc', flatten=False)
    rho_tot, rho_gnd = {}, {}
    for inc in u[TYPES[0]]:
        k = lambda t: u[t][inc]['phase']['Cu'] + np.abs(b[t][inc]['phase']['Cu'])                  # noqa
        tot = sum(k(t) for t in TYPES) + d_ed[inc]['phase']['Cu'] + d_sc[inc]['phase']['Cu']
        gnd = np.abs(k('ed_pos') - k('ed_neg')) + np.abs(k('sc_pos') - k('sc_neg'))
        rho_tot[inc] = tot.sum(axis=1)
        rho_gnd[inc] = gnd.sum(axis=1)
    return rho_tot, rho_gnd


def process(case, out):
    r = damask.Result(case/'job.hdf5')
    g = damask.GeomGrid.load(case/'geom.vti')
    material_ID = g.material.flatten(order='F')

    P = r.get('P', flatten=False)
    F = r.get('F', flatten=False)
    rho_tot, rho_gnd = densities(r)
    tau_b = r.get('tau_back', flatten=False) if 'full' in case.name else None

    incs = list(P.keys())
    eps, sig, rt, rg = [], [], [], []
    sigma_vM = {}
    for inc in incs:
        Pi, Fi = P[inc]['phase']['Cu'], F[inc]['phase']['Cu']
        sigma = np.einsum('nij,nkj->nik', Pi, Fi)/np.linalg.det(Fi)[:, None, None]                  # Cauchy stress per cell
        sigma_vM[inc] = damask.mechanics.equivalent_stress_Mises(sigma)
        F_av = Fi.mean(axis=0)
        eps.append(np.log(F_av[0, 0]))
        sig.append(sigma[:, 0, 0].mean())
        rt.append(rho_tot[inc].mean())
        rg.append(rho_gnd[inc].mean())

    last = incs[-1]
    data = dict(cells=g.cells, size=g.size, material=material_ID,
                strain=np.array(eps), stress=np.array(sig),
                rho_tot=np.array(rt), rho_gnd=np.array(rg),
                field_rho_tot=rho_tot[last], field_rho_gnd=rho_gnd[last],
                field_sigma_vM=sigma_vM[last])
    if tau_b is not None:
        data['field_tau_back'] = tau_b[last]['phase']['Cu']

    layer_file = ROOT/'layer_of_grain.txt'
    if case.name.startswith('grad') and layer_file.exists():
        layer = np.loadtxt(layer_file, dtype=int)[material_ID]
        data['layer'] = layer
        for name, field in [('rho_tot', rho_tot), ('rho_gnd', rho_gnd), ('sigma_vM', sigma_vM)]:
            data[f'layer_{name}'] = np.array([[field[inc][layer == l].mean() for l in (1, 2, 3)] for inc in incs])

    np.savez_compressed(out/f'{case.name}.npz', **data)
    print(f'{case.name}: {len(incs)} increments, final strain {eps[-1]:.4f}, stress {sig[-1]/1e6:.1f} MPa')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', default=str(ROOT/'runs'))
    parser.add_argument('--out', default=str(ROOT/'data'))
    parser.add_argument('--only', default='')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    for case in sorted(Path(args.runs).iterdir()):
        if args.only in case.name and (case/'job.hdf5').exists():
            process(case, out)
