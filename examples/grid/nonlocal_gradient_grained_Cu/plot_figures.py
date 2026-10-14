#!/usr/bin/env python3
"""
Recreate the result figures (Figs. 7-18) of Zhang et al., Int. J. Plast. 163 (2023) 103553 from the
reduced simulation data in 'data/*.npz' (see postprocess.py), plus a direct comparison of the
gradient grained and homogeneously grained copper.
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent
SIZES = [5, 10, 20, 40, 80]
COLOR = dict(zip(SIZES, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']))
MARKER = dict(zip(SIZES, ['^', 'o', 's', 'D', 'v']))
LAYER_COLOR = {1: '#2a78d6', 2: '#eb6834', 3: '#1baf7a'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SEQ = LinearSegmentedColormap.from_list('seq_blue', ['#f3f7fd', '#b9d3f2', '#6aa1e6', '#2a78d6', '#174a8c', '#0b2547'])
DIV = LinearSegmentedColormap.from_list('div', ['#174a8c', '#6aa1e6', '#e4e3df', '#ef8a62', '#a3261f'])

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.6,
                     'legend.frameon': False, 'savefig.dpi': 200, 'figure.dpi': 100})


def load(name):
    f = DATA/f'{name}.npz'
    return dict(np.load(f)) if f.exists() else None


def curve(ax, d, **kw):
    n = len(d['strain'])
    ax.plot(d['strain'], d['stress']/1e6, markevery=max(1, n//6), markersize=4, **kw)


def strain_axis(ax, ylabel='True stress (MPa)'):
    ax.set_xlim(0, 0.05)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('True strain')
    ax.set_ylabel(ylabel)


def interp(d, eps):
    return np.interp(eps, d['strain'], d['stress'])


def field(d, key):
    nx, ny = d['cells'][:2]
    return d[key].reshape((nx, ny), order='F').T                                                    # (y, x)


def save(fig, name):
    fig.savefig(FIGS/name, bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


def fig7():
    fig, axes = plt.subplots(2, 2, figsize=(8, 6.4))
    for ax, rho in zip([axes[0, 0], axes[1, 0], axes[1, 1]], ['1.2', '2.4', '3.6']):
        for s in SIZES:
            d = load(f'homog_d{s}_rho{rho}_full')
            if d: curve(ax, d, color=COLOR[s], marker=MARKER[s], mfc='white', label=f'{s} µm')
        strain_axis(ax)
        ax.set_title(rf'$\rho_\mathrm{{initial}}$ = {rho}$\times10^{{12}}$ m$^{{-2}}$', fontsize=9, loc='left')
        ax.legend(ncol=2, fontsize=8)
    ax = axes[0, 1]
    x = np.linspace(0, 20, 50)
    for eps, c, lab in [(0.005, INK2, r'$\varepsilon$ = 0.5%'), (0.05, '#e34948', r'$\varepsilon$ = 5%')]:
        pts = [(s**-0.5*np.sqrt(1000), interp(d, eps)/1e6) for s in SIZES
               if (d := load(f'homog_d{s}_rho1.2_full')) is not None and d['strain'][-1] >= eps*0.98]
        if len(pts) > 1:
            pts = np.array(pts)
            k = np.polyfit(pts[:, 0], pts[:, 1], 1)
            ax.plot(pts[:, 0], pts[:, 1], '^', mfc='white', color=c, label=f'simulation, {lab}')
            ax.plot(x, np.polyval(k, x), '--', color=c, lw=1.2,
                    label=rf'fit: {k[1]:.0f} + {k[0]:.1f} $d^{{-1/2}}$')
    # Hall-Petch lines of Zhang et al. 2023, Fig. 7(b) (fits to Armstrong et al. 1962 / Hansen and Ralph 1982; digitized)
    for a, k in [(31.0, 2.4), (97.0, 5.4)]:
        ax.plot(x, a + k*x, ':', color='#6250d6', lw=1.2)
    ax.plot([], [], ':', color='#6250d6', label='Zhang et al. (2023) fit to exp.')
    ax.set_xlim(0, 20); ax.set_ylim(0, 300)
    ax.set_xlabel(r'$d^{-1/2}$ (mm$^{-1/2}$)'); ax.set_ylabel('Flow stress (MPa)')
    ax.legend(fontsize=7)
    for ax, l in zip(axes.flat, 'abcd'):
        ax.text(-0.14, 1.04, f'({l})', transform=ax.transAxes, fontsize=10)
    fig.tight_layout()
    save(fig, 'fig07_grain_size_effect.png')


def fig8_9():
    for ref, label, name in [('noBS', 'Dashed: without back stress', 'fig08_back_stress.png'),
                             ('noFlux', 'Dashed: without dislocation flux', 'fig09_flux.png')]:
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        for s in SIZES:
            d = load(f'homog_d{s}_rho1.2_full')
            if d: curve(ax, d, color=COLOR[s], marker=MARKER[s], mfc='white', label=f'{s} µm')
            if ref == 'noBS' and (d := load(f'homog_d{s}_rho1.2_noBS')):
                ax.plot(d['strain'], d['stress']/1e6, '--', color=COLOR[s], lw=1.2)
        if ref == 'noFlux' and (d := load('homog_d5_rho1.2_noFlux')):
            ax.plot(d['strain'], d['stress']/1e6, '--', color=INK, lw=1.4, label='no dislocation flux')
        strain_axis(ax)
        ax.set_title(label, fontsize=8, loc='left', color=INK2)
        ax.legend(ncol=2, fontsize=8)
        save(fig, name)


def fig10():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True)
    loc = load('homog_d5_rho1.2_noFlux')
    for s in SIZES:
        full, nobs = load(f'homog_d{s}_rho1.2_full'), load(f'homog_d{s}_rho1.2_noBS')
        if not (full and nobs and loc): continue
        e = np.linspace(0, min(full['strain'][-1], nobs['strain'][-1], loc['strain'][-1]), 60)[1:]
        kin = (interp(full, e) - interp(nobs, e))/1e6
        iso = (interp(nobs, e) - interp(loc, e))/1e6
        a1.plot(e, kin, '--', color=COLOR[s], label=f'{s} µm')
        a2.plot(e, iso, color=COLOR[s], marker=MARKER[s], markevery=10, ms=4, mfc='white', label=f'{s} µm')
    a1.set_ylabel('GND-induced hardening (MPa)')
    a1.set_title('Kinematic (back stress)', fontsize=9, loc='left')
    a2.set_title('Isotropic (forest)', fontsize=9, loc='left')
    for a in (a1, a2):
        a.set_xlim(0, 0.05); a.set_xlabel('True strain'); a.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    save(fig, 'fig10_kinematic_isotropic_hardening.png')


def fig11():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.4), sharey=True)
    for s in SIZES:
        d = load(f'homog_d{s}_rho1.2_full')
        if not d: continue
        a1.plot(d['strain'], d['rho_tot']/1e13, color=COLOR[s], marker=MARKER[s], markevery=max(1, len(d['strain'])//6),
                ms=4, mfc='white', label=f'{s} µm')
        a2.plot(d['strain'], d['rho_gnd']/1e13, '--', color=COLOR[s], label=f'{s} µm')
    a1.set_ylabel(r'Dislocation density ($10^{13}$ m$^{-2}$)')
    a1.set_title('Total', fontsize=9, loc='left'); a2.set_title('GND', fontsize=9, loc='left')
    for a in (a1, a2):
        a.set_xlim(0, 0.05); a.set_ylim(bottom=0); a.set_xlabel('True strain'); a.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    save(fig, 'fig11_density_evolution.png')


def maps(cases, key, title, scale, label, name, vmax=None, cmap=SEQ, vmin=0.0, labels=None):
    ds = [load(c) for c in cases]
    ds = [(d, l) for d, l in zip(ds, labels or cases) if d is not None]
    if not ds: return
    vmax = vmax or max(np.percentile(d[key], 99.5) for d, _ in ds)/scale
    fig, axes = plt.subplots(1, len(ds), figsize=(1.9*len(ds)+0.8, 2.3), squeeze=False)
    for ax, (d, l) in zip(axes[0], ds):
        im = ax.imshow(field(d, key)/scale, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(l, fontsize=8)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.85, label=label)
    fig.suptitle(title, fontsize=9, x=0.02, y=1.06, ha='left')
    save(fig, name)


def fig12():
    cases = [f'homog_d{s}_rho1.2_full' for s in SIZES]
    labels = [f'd = {s} µm' for s in SIZES]
    maps(cases, 'field_rho_tot', '(a) Total dislocation density at ~5% strain', 1e14, r'$10^{14}$ m$^{-2}$',
         'fig12a_total_density_maps.png', labels=labels)
    maps(cases, 'field_rho_gnd', '(b) GND density at ~5% strain', 1e14, r'$10^{14}$ m$^{-2}$',
         'fig12b_gnd_density_maps.png', labels=labels)


def avg(cases, key):
    ds = [d for c in cases if (d := load(c)) is not None]
    if not ds: return None, None
    e = np.linspace(0, min(d['strain'][-1] for d in ds), 80)
    if key == 'stress':
        return e, np.mean([interp(d, e) for d in ds], axis=0)
    return e, np.mean([np.array([np.interp(e, d['strain'], d[key][:, l]) for l in range(3)]).T for d in ds], axis=0)


def fig13():
    d = load('grad_o1_full')
    if not d: return
    m = field(d, 'material').astype(float)
    ny = m.shape[0]
    rng = np.random.default_rng(0)
    lut = rng.permutation(int(m.max())+1)
    fig, ax = plt.subplots(figsize=(3.2, 4.2))
    ax.imshow(lut[m[:ny//2].astype(int)], origin='lower', cmap='tab20', interpolation='nearest',
              extent=[0, d['size'][0]*1e6, 0, d['size'][1]*1e6/2])
    for y, l in [(7.5, 'I'), (30, 'II'), (75, 'III')]:
        ax.text(-6, y, l, ha='right', va='center', fontsize=10)
    ax.set_xlabel('x (µm)'); ax.set_ylabel('y (µm, from surface)'); ax.grid(False)
    ax.set_title('Gradient grained model (half of the periodic cell)', fontsize=8, loc='left')
    save(fig, 'fig13_gradient_model.png')


def grad_cases(variant):
    return [f'grad_o{o}_{variant}' for o in (1, 2, 3)]


def fig14_15_16():
    full, nof = load('grad_o1_full'), load('grad_o1_noFlux')
    for key, scale, lab, title, name in [
            ('field_sigma_vM', 1e6, 'MPa', '(a) von Mises stress at ~5% strain', 'fig14a_vonMises_maps.png'),
            ('field_rho_tot', 1e14, r'$10^{14}$ m$^{-2}$', '(a) Total dislocation density at ~5% strain', 'fig15a_total_density_maps.png'),
            ('field_rho_gnd', 1e14, r'$10^{14}$ m$^{-2}$', '(b) GND density at ~5% strain', 'fig15b_gnd_density_maps.png')]:
        if full and nof:
            maps(['grad_o1_full', 'grad_o1_noFlux'], key, title, scale, lab, name,
                 labels=['with dislocation flux', 'no dislocation flux'])

    # Fig. 14(b): layer averages of von Mises stress at the final strain
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    for v, sty, lab in [('full', '-o', 'with dislocation flux'), ('noFlux', '--s', 'no dislocation flux')]:
        e, L = avg(grad_cases(v), 'layer_sigma_vM')
        if e is not None:
            ax.plot([1, 2, 3], L[-1]/1e6, sty, color=INK if v == 'noFlux' else '#e34948', mfc='white', label=lab)
    ax.set_xticks([1, 2, 3], ['I', 'II', 'III']); ax.set_xlabel('Layer'); ax.set_ylabel('von Mises stress (MPa)')
    ax.legend(fontsize=8)
    save(fig, 'fig14b_vonMises_layers.png')

    # Fig. 15(c,d): layer-wise density evolution
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3), sharey=True)
    for ax, key, t in zip(axes, ['layer_rho_tot', 'layer_rho_gnd'], ['(c) Total dislocation density', '(d) GND density']):
        for v, ls in [('full', '-'), ('noFlux', ':')]:
            e, L = avg(grad_cases(v), key)
            if e is None: continue
            for l in range(3):
                ax.plot(e, L[:, l]/1e13, ls, color=LAYER_COLOR[l+1],
                        label=f'Layer {"I"*(l+1) if l < 2 else "III"}' + ('' if v == 'full' else ', no flux'))
        ax.set_title(t, fontsize=9, loc='left'); ax.set_xlim(0, 0.05); ax.set_ylim(bottom=0); ax.set_xlabel('True strain')
        ax.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel(r'Dislocation density ($10^{13}$ m$^{-2}$)')
    fig.tight_layout()
    save(fig, 'fig15cd_layer_density_evolution.png')

    # Fig. 16: layer averages at final strain
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    for key, c, lab in [('layer_rho_tot', INK, 'Total dislocation density'), ('layer_rho_gnd', '#e34948', 'GND density')]:
        for v, sty in [('full', '-o'), ('noFlux', '--s')]:
            e, L = avg(grad_cases(v), key)
            if e is not None:
                ax.plot([1, 2, 3], L[-1]/1e13, sty, color=c, mfc='white', label=lab if v == 'full' else None)
    ax.plot([], [], '--', color=INK2, label='dashed: without dislocation flux')
    ax.set_xticks([1, 2, 3], ['I', 'II', 'III']); ax.set_xlabel('Layer')
    ax.set_ylabel(r'Dislocation density ($10^{13}$ m$^{-2}$)'); ax.set_ylim(bottom=0); ax.legend(fontsize=7)
    save(fig, 'fig16_layer_densities.png')


def fig17():
    d = load('grad_o1_full')
    if d is None or 'field_tau_back' not in d: return
    nx, ny = d['cells'][:2]
    fig, axes = plt.subplots(2, 6, figsize=(11, 4.6))
    for s, ax in enumerate(axes.flat):
        f = d['field_tau_back'][:, s].reshape((nx, ny), order='F').T[:ny//2]/1e6
        im = ax.imshow(f, origin='lower', cmap=DIV, vmin=-9, vmax=9, interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.text(0.04, 0.06, f'S{s+1}', transform=ax.transAxes, fontsize=9, color=INK)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label=r'Back stress $\tau_\mathrm{b}$ (MPa)')
    fig.suptitle('Back stress on the 12 slip systems at ~5% strain (surface at bottom, core at top)', fontsize=9, x=0.02, ha='left')
    save(fig, 'fig17_back_stress_maps.png')


def fig18():
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for v, ls, c, lab in [('full', '-', INK, 'Gradient grained copper'),
                          ('noBS', ':', '#e34948', 'no back stress'),
                          ('noFlux', '--', '#2a78d6', 'no dislocation flux')]:
        e, s = avg(grad_cases(v), 'stress')
        if e is not None: ax.plot(e, s/1e6, ls, color=c, label=lab)
    strain_axis(ax); ax.legend(fontsize=8)
    save(fig, 'fig18_gradient_stress_strain.png')


def comparison():
    """Gradient grained vs homogeneously grained copper with the grain sizes of the three layers."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.5))
    e, s = avg(grad_cases('full'), 'stress')
    if e is not None:
        a1.plot(e, s/1e6, color=INK, lw=2.2, label='gradient (d, 2d, 4d)')
    fs = {1: 15/105, 2: 30/105, 3: 60/105}                                                          # area fractions of layers I, II, III
    homs = {l: load(f'homog_d{5*2**(l-1)}_rho1.2_full') for l in (1, 2, 3)}
    for l, d in homs.items():
        if d: curve(a1, d, color=COLOR[5*2**(l-1)], marker=MARKER[5*2**(l-1)], mfc='white', ls='--', lw=1.2,
                    label=f'homogeneous {5*2**(l-1)} µm')
    if all(homs.values()) and e is not None:
        rom = sum(fs[l]*interp(homs[l], e) for l in (1, 2, 3))
        a1.plot(e, rom/1e6, ':', color='#6250d6', lw=1.6, label='rule of mixtures')
    strain_axis(a1); a1.legend(fontsize=7)
    a1.set_title('(a) Stress-strain response', fontsize=9, loc='left')

    full = [d for c in grad_cases('full') if (d := load(c))]
    if full and all(homs.values()):
        x = np.arange(3)
        g = np.mean([[d['field_rho_gnd'][d['layer'] == l].mean() for l in (1, 2, 3)] for d in full], axis=0)
        h = [homs[l]['rho_gnd'][-1] for l in (1, 2, 3)]
        a2.bar(x-0.18, np.array(g)/1e13, 0.34, color='#2a78d6', label='layer in gradient sample')
        a2.bar(x+0.18, np.array(h)/1e13, 0.34, color='#eb6834', label='homogeneous counterpart')
        a2.set_xticks(x, ['I (5 µm)', 'II (10 µm)', 'III (20 µm)'])
        a2.set_ylabel(r'GND density at 5% ($10^{13}$ m$^{-2}$)'); a2.legend(fontsize=7)
    a2.set_title('(b) GND density per layer', fontsize=9, loc='left')
    fig.tight_layout()
    save(fig, 'fig19_comparison_gradient_homogeneous.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=str(ROOT/'data'))
    parser.add_argument('--figs', default=str(ROOT/'figures'))
    args = parser.parse_args()
    DATA, FIGS = Path(args.data), Path(args.figs)
    FIGS.mkdir(exist_ok=True)
    for f in [fig7, fig8_9, fig10, fig11, fig12, fig13, fig14_15_16, fig17, fig18, comparison]:
        f()
