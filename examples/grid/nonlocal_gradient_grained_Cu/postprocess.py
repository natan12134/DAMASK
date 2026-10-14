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


def densities(rv):
    """Total and GND density per cell (summed over slip systems) of a single-increment view."""
    k = {t: rv.get(f'rho_u_{t}') + np.abs(rv.get(f'rho_b_{t}')) for t in TYPES}
    tot = sum(k.values()) + rv.get('rho_d_ed') + rv.get('rho_d_sc')
    gnd = np.abs(k['ed_pos'] - k['ed_neg']) + np.abs(k['sc_pos'] - k['sc_neg'])
    return tot.sum(axis=1), gnd.sum(axis=1)


def process(case, out):
    r = damask.Result(case/'job.hdf5')
    g = damask.GeomGrid.load(case/'geom.vti')
    material_ID = g.material.flatten(order='F')
    layer_file = ROOT/'layer_of_grain.txt'
    layer = np.loadtxt(layer_file, dtype=int)[material_ID] \
            if case.name.startswith('grad') and layer_file.exists() else None

    eps, sig, rt, rg, layers = [], [], [], [], []
    for inc in r.increments:
        rv = r.view(increments=inc)
        P, F = rv.get('P'), rv.get('F')
        if P is None: continue
        sigma = np.einsum('nij,nkj->nik', P, F)/np.linalg.det(F)[:, None, None]                     # Cauchy stress per cell
        vM = damask.mechanics.equivalent_stress_Mises(sigma)
        tot, gnd = densities(rv)
        eps.append(np.log(F.mean(axis=0)[0, 0]))
        sig.append(sigma[:, 0, 0].mean())
        rt.append(tot.mean())
        rg.append(gnd.mean())
        if layer is not None:
            layers.append([[f[layer == l].mean() for l in (1, 2, 3)] for f in (tot, gnd, vM)])

    data = dict(cells=g.cells, size=g.size, material=material_ID,
                strain=np.array(eps), stress=np.array(sig),
                rho_tot=np.array(rt), rho_gnd=np.array(rg),
                field_rho_tot=tot, field_rho_gnd=gnd, field_sigma_vM=vM)
    if 'full' in case.name:
        data['field_tau_back'] = rv.get('tau_back')
    if layer is not None:
        data['layer'] = layer
        layers = np.array(layers)
        for i, name in enumerate(['rho_tot', 'rho_gnd', 'sigma_vM']):
            data[f'layer_{name}'] = layers[:, i, :]

    np.savez_compressed(out/f'{case.name}.npz', **data)
    print(f'{case.name}: {len(eps)} increments, final strain {eps[-1]:.4f}, stress {sig[-1]/1e6:.1f} MPa')


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
