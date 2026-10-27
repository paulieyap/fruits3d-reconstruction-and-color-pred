"""Evaluate a pcdiff checkpoint on harvester tomatoes (run inside the pcdiff container).

  python3 harvester/eval_harvester.py --diff <ckpt> --data <dataset_root> --out <dir> [--n-syn 200] [-T 50 -s 1.0]

syn/val : full procedural GT  -> Chamfer (mm), F-score @ 2.5/5 mm, volume (convex hull) error
real/*  : wrist-view input, stacker view held out. Held-out points further than 3 mm from every
          input point are the surface the wrist camera never saw; we report how close the
          completion comes to them (median mm, % within 5 mm). A sphere fitted to the input is the
          baseline on the same points.
Writes metrics.csv (one row per sample), summary.json and preds/<id>.npz (input, pred, gt/heldout).
With --traj-every k the preds also hold the denoising trajectory: the initial noise and every k-th solver
step (traj (F, N, 6) float16, traj_step, traj_t).
"""
import glob, json, os, sys

import click
import numpy as np
import torch
from scipy.spatial import ConvexHull, cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from pcdiff.tools.diff_completion_pipeline import DiffCompletion  # noqa: E402
from pcdiff.utils.collations import point_set_to_sparse  # noqa: E402


def fit_sphere(p):
    A = np.c_[2 * p, np.ones(len(p))]
    sol, *_ = np.linalg.lstsq(A, (p ** 2).sum(1), rcond=None)
    c = sol[:3]
    return c, float(np.sqrt(max(sol[3] + c @ c, 1e-12)))


def sphere_points(c, r, n=8192):
    v = np.random.default_rng(0).normal(size=(n, 3))
    return c + r * v / np.linalg.norm(v, axis=1, keepdims=True)


def clean_pred(pred, k=8, z=2.5):
    """Drop stray generated points (mean kNN distance outliers)."""
    d, _ = cKDTree(pred).query(pred, k + 1)
    m = d[:, 1:].mean(1)
    return pred[m < np.median(m) + z * 1.4826 * np.median(np.abs(m - np.median(m)))]


def chamfer_f(gt, pred, ths=(0.0025, 0.005)):
    d_gp, _ = cKDTree(pred).query(gt)
    d_pg, _ = cKDTree(gt).query(pred)
    out = {'cd_mm': 1000 * (d_gp.mean() + d_pg.mean()) / 2}
    for t in ths:
        p, r = (d_pg < t).mean(), (d_gp < t).mean()
        out[f'f{t * 1000:g}'] = 100 * 2 * p * r / max(p + r, 1e-9)
    return out


def hull_vol(p):
    try:
        return ConvexHull(p).volume * 1e6
    except Exception:
        return float('nan')


def batches(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


@click.command()
@click.option('--diff', required=True)
@click.option('--data', required=True)
@click.option('--out', required=True)
@click.option('--n-syn', default=200)
@click.option('--n-real', default=0, help='0 = all real val pairs')
@click.option('--real-split', default='val')
@click.option('--syn-split', default='val', help='train = check for overfitting against the val numbers')
@click.option('-T', 'steps', default=50)
@click.option('-s', 'cond_w', default=1.0)
@click.option('--bs', default=8)
@click.option('--x0-fix-tc', type=int, default=None, help='epsilon models: re-centre/rescale the implied x0 above this t')
@click.option('--x0-prior-std', default=None, help='comma-separated normalised XYZ std of a GT fruit, e.g. 0.83,0.79,0.86')
@click.option('--traj-every', default=0, help='also save the sample at every k-th denoising step (0 = off)')
def main(diff, data, out, n_syn, n_real, real_split, syn_split, steps, cond_w, bs, x0_fix_tc, x0_prior_std, traj_every):
    os.makedirs(os.path.join(out, 'preds'), exist_ok=True)
    ov = {}
    if x0_fix_tc is not None:
        ov['x0_fix_tc'] = x0_fix_tc
        ov['x0_prior_std'] = [float(v) for v in x0_prior_std.split(',')] if x0_prior_std else None
    dc = DiffCompletion(diff, steps, cond_w, ov)
    dcfg = dc.hparams['data']
    norm = (dcfg.get('norm_mean'), dcfg.get('norm_std'))
    recentre = dcfg.get('recentre') == 'partial'
    rng = np.random.default_rng(0)
    syn = sorted(glob.glob(os.path.join(data, 'syn', syn_split, '*.npz')))
    syn = list(rng.permutation(syn)[:n_syn]) if n_syn else []
    real = sorted(glob.glob(os.path.join(data, 'real', real_split, '*.npz')))
    real = list(rng.permutation(real)[:n_real]) if n_real else real
    save_steps = ((-1,) + tuple(range(traj_every - 1, steps, traj_every))) if traj_every else ()
    ts = dc.model.dpm_scheduler.timesteps.cpu().numpy()
    # noise level each saved sample sits at: the initial noise at ts[0], after step i at ts[i + 1] (0 after the last)
    traj_t = np.array([int(ts[i + 1]) if i + 1 < len(ts) else 0 for i in save_steps])
    rows = []
    for kind, files in (('syn', syn), ('real', real)):
        for chunk in batches(files, bs):
            samples = [np.load(f) for f in chunk]
            # the DPM-solver scheduler keeps per-batch history, so every batch must have the same size:
            # pad the last one by repeating its first sample (zip below drops the padding again)
            samples += [samples[0]] * (bs - len(samples))
            full, part, mean, std, offs = [], [], [], [], []
            for s in samples:
                p = s['rgbd_pcd'].astype(np.float64)
                # no GT for real samples: the partial (with its colours) stands in, it only sets the point count
                # and the RGB statistics (all -1 colours would make them NaN)
                g = s['aligned_data'].astype(np.float64) if kind == 'syn' else p
                off = p[:, :3].mean(0) if recentre else np.zeros(3)
                p = p.copy(); g = g.copy()
                p[:, :3] -= off; g[:, :3] -= off
                sp = point_set_to_sparse(g, p, 30000, 15000, 0.01, *norm)
                full.append(sp[0]); mean.append(sp[1]); std.append(sp[2]); part.append(sp[3]); offs.append(off)
            batch = {'pcd_full': torch.stack(full).float(), 'pcd_part': torch.stack(part).float(),
                     'mean': torch.stack(mean).float(), 'std': torch.stack(std).float()}
            if save_steps:
                preds, traj = dc.complete(batch, save_steps)
                traj = np.stack([traj[k] for k in save_steps], 1)  # (B, F, N, 6)
            else:
                preds, traj = dc.complete(batch), [None] * len(samples)
            for f, s, pr, off, tr in zip(chunk, samples, preds, offs, traj):
                extra = {} if tr is None else {'traj': (tr + np.r_[off, 0, 0, 0]).astype(np.float16),
                                               'traj_step': np.array(save_steps), 'traj_t': traj_t}
                fid = os.path.basename(f)[:-4]
                inp = s['rgbd_pcd'][:, :3].astype(np.float64)
                pred = clean_pred(pr[:, :3] + off)
                c, r = fit_sphere(inp)
                sph = sphere_points(c, r)
                row = {'kind': kind, 'id': fid, 'n_in': len(inp), 'r_sphere_mm': 1000 * r}
                if kind == 'syn':
                    gt = s['aligned_data'][:, :3].astype(np.float64)
                    for name, P in (('ddpm', pred), ('sphere', sph)):
                        for k, v in chamfer_f(gt, P).items():
                            row[f'{name}_{k}'] = v
                        row[f'{name}_vol_err_pct'] = 100 * (hull_vol(P) - hull_vol(gt)) / hull_vol(gt)
                    np.savez_compressed(os.path.join(out, 'preds', fid + '.npz'), inp=s['rgbd_pcd'], pred=pr + np.r_[off, 0, 0, 0],
                                        gt=s['aligned_data'], **extra)
                else:
                    held = s['heldout'][:, :3].astype(np.float64)
                    d_in, _ = cKDTree(inp).query(held)
                    unseen = held[d_in > 0.003]
                    row['n_unseen'] = len(unseen)
                    for name, P in (('ddpm', pred), ('sphere', sph)):
                        if len(unseen) >= 30:
                            d, _ = cKDTree(P).query(unseen)
                            row[f'{name}_unseen_med_mm'] = 1000 * np.median(d)
                            row[f'{name}_unseen_in5mm_pct'] = 100 * (d < 0.005).mean()
                        d, _ = cKDTree(P).query(held)
                        row[f'{name}_held_med_mm'] = 1000 * np.median(d)
                    row['ddpm_vol_ml'] = hull_vol(pred)
                    row['sphere_vol_ml'] = 4 / 3 * np.pi * r ** 3 * 1e6
                    np.savez_compressed(os.path.join(out, 'preds', fid + '.npz'), inp=s['rgbd_pcd'], pred=pr + np.r_[off, 0, 0, 0],
                                        heldout=s['heldout'], crop_jpg=s['crop_jpg'], meta=s['meta'], **extra)
                rows.append(row)
            print(f'{kind} {len(rows)}', flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ('kind', 'id'), k))
    with open(os.path.join(out, 'metrics.csv'), 'w') as fh:
        fh.write(','.join(keys) + '\n')
        for r in rows:
            fh.write(','.join(str(r.get(k, '')) for k in keys) + '\n')
    summ = {'ckpt': diff, 'T': steps, 's': cond_w, 'x0_fix_tc': x0_fix_tc, 'x0_prior_std': x0_prior_std}
    for kind in ('syn', 'real'):
        rs = [r for r in rows if r['kind'] == kind]
        summ[kind] = {'n': len(rs)}
        for k in keys:
            vals = [r[k] for r in rs if k in r and isinstance(r[k], (int, float, np.floating)) and k not in ('kind', 'id')]
            if vals:
                summ[kind][k] = float(np.nanmedian(vals))
    json.dump(summ, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
