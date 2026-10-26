"""Collect everything the results page shows into one JSON (run on the GPU box, prep env).

  python export_page_data.py --run v1=runs/harvester_v1 --run v2=runs/harvester_v2 --ds data/harvester_ds_v1 --views data/views --out page_data.json

- curves: TensorBoard scalars (train loss per ~100 steps, val CD / F per epoch), read with tensorboard's
  event reader if available, else from <run>/scalars.csv (tools/export_scalars.py output)
- eval: summary.json + metrics.csv of every evaluated checkpoint
- samples: a few syn / real predictions of the chosen checkpoint, point clouds quantised to 0.1 mm int16
  around the input centroid, base64; real samples carry the wrist-image crop
- dataset: counts and radius / depth histograms from index.csv, one SAM 3 overlay per camera
"""
import argparse, base64, csv, glob, io, json, os

import cv2
import numpy as np


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def q(p, c, n, rng):
    p = p[np.isfinite(p).all(1)]
    if len(p) > n:
        p = p[rng.choice(len(p), n, replace=False)]
    return b64(np.clip(np.round((p - c) * 1e4), -32767, 32767).astype(np.int16))


def rgb8(c, n_idx):
    return b64((np.clip(c[n_idx], 0, 1) * 255).astype(np.uint8))


def read_scalars(run):
    rows = []
    csvp = os.path.join(run, 'scalars.csv')
    if os.path.isfile(csvp):
        rows = list(csv.DictReader(open(csvp)))
    out = {}
    for r in rows:
        out.setdefault(r['tag'], []).append((int(r['step']), float(r['epoch']) if r['epoch'] else None, float(r['value'])))
    for k in out:
        out[k].sort()
    curves = {}
    for tag, vals in out.items():
        if tag.startswith('train/') and len(vals) > 400:  # thin dense step curves
            vals = vals[::max(1, len(vals) // 400)]
        curves[tag] = [[s, e, round(v, 6)] for s, e, v in vals]
    return curves


def overlay(view_dir):
    img = cv2.imread(os.path.join(view_dir, 'color.jpg'))
    lab = cv2.imread(os.path.join(view_dir, 'inst.png'), cv2.IMREAD_UNCHANGED)
    rng = np.random.default_rng(3)
    vis = img.copy()
    for k in np.unique(lab)[1:]:
        m = lab == k
        col = np.array([40, 60, 235]) * rng.uniform(0.6, 1.0) + rng.uniform(0, 60, 3)
        vis[m] = (0.45 * vis[m] + 0.55 * col).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, (255, 255, 255), 2)
    vis = cv2.resize(np.hstack([img, vis]), (1280, 400))
    ok, jpg = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(jpg.tobytes()).decode(), int(len(np.unique(lab)) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='append', required=True, help='name=run_dir (repeatable); evals are read from every run_dir/eval')
    ap.add_argument('--ds', required=True)
    ap.add_argument('--views', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ckpt-eval', default=None, help='eval/<name> folder for the 3D samples (default: last)')
    ap.add_argument('--n-syn', type=int, default=12)
    ap.add_argument('--n-real', type=int, default=16)
    ap.add_argument('--overlays', nargs='*', default=[])
    ap.add_argument('--scan', default=None, help='folder of quick evals (<name>/summary.json), e.g. checkpoint/sampler scans')
    ap.add_argument('--hide-eval', nargs='*', default=[], help='eval folder names to leave out of the table')
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    runs = dict(r.split('=', 1) for r in a.run)
    data = {'curves': {n: read_scalars(d) for n, d in runs.items()}, 'evals': {}, 'samples': [], 'dataset': {}}
    eval_dirs = {}
    for d in sorted(p for rd in runs.values() for p in glob.glob(os.path.join(rd, 'eval', '*'))):
        if os.path.isfile(os.path.join(d, 'summary.json')):
            eval_dirs[os.path.basename(d)] = d
            data['evals'][os.path.basename(d)] = {
                'summary': json.load(open(os.path.join(d, 'summary.json'))),
                'rows': list(csv.DictReader(open(os.path.join(d, 'metrics.csv'))))}

    for n in a.hide_eval:
        data['evals'].pop(n, None)
    if a.scan:
        data['scan'] = {os.path.basename(os.path.dirname(f)): {k: json.load(open(f))[k] for k in ('syn', 'real')}
                        for f in glob.glob(os.path.join(a.scan, '*', 'summary.json'))}
    ev = a.ckpt_eval or (sorted(data['evals'])[-1] if data['evals'] else None)
    data['sample_ckpt'] = ev
    if ev:
        rows = {r['id']: r for r in data['evals'][ev]['rows']}
        preds = sorted(glob.glob(os.path.join(eval_dirs[ev], 'preds', '*.npz')))
        syn = [p for p in preds if rows.get(os.path.basename(p)[:-4], {}).get('kind') == 'syn']
        real = [p for p in preds if rows.get(os.path.basename(p)[:-4], {}).get('kind') == 'real']
        for kind, files, n in (('syn', syn, a.n_syn), ('real', real, a.n_real)):
            files = [files[i] for i in np.linspace(0, len(files) - 1, min(n, len(files))).astype(int)] if files else []
            for f in files:
                z = np.load(f)
                fid = os.path.basename(f)[:-4]
                inp = z['inp'].astype(np.float64)
                c = inp[:, :3].mean(0)
                ii = rng.choice(len(inp), min(2500, len(inp)), replace=False)
                pred = z['pred'].astype(np.float64)
                pi = rng.choice(len(pred), min(4000, len(pred)), replace=False)
                s = {'kind': kind, 'id': fid, 'metrics': {k: v for k, v in rows[fid].items() if v not in ('', None)},
                     'inp': b64(np.clip(np.round((inp[ii, :3] - c) * 1e4), -32767, 32767).astype(np.int16)),
                     'inp_rgb': rgb8(inp[:, 3:], ii),
                     'pred': b64(np.clip(np.round((pred[pi, :3] - c) * 1e4), -32767, 32767).astype(np.int16))}
                if 'traj' in z.files:  # denoising trajectory on 1500 of the shown pred points, so frames morph (and the page stays small)
                    s['traj'] = [b64(np.clip(np.round((fr[:, :3].astype(np.float64) - c) * 1e4), -32767, 32767).astype(np.int16))
                                 for fr in z["traj"][:, pi[:1500]]]
                    s['traj_step'] = [int(v) + 1 for v in z['traj_step']]  # solver steps done: 0 = pure noise
                    s['traj_t'] = [int(v) for v in z['traj_t']]
                other = z['gt'] if kind == 'syn' else z['heldout']
                s['other'] = q(other[:, :3].astype(np.float64), c, 3000, rng)
                if kind == 'real':
                    s['crop'] = base64.b64encode(z['crop_jpg'].tobytes()).decode()
                data['samples'].append(s)

    idx = list(csv.DictReader(open(os.path.join(a.ds, 'index.csv'))))
    syn = [r for r in idx if r['kind'] == 'syn']
    data['dataset'] = {
        'syn_train': sum(r['split'] == 'train' for r in syn), 'syn_val': sum(r['split'] == 'val' for r in syn),
        'real_train': sum(r['kind'] == 'real' and r['split'] == 'train' for r in idx),
        'real_val': sum(r['kind'] == 'real' and r['split'] == 'val' for r in idx),
        'snapshots': len({'_'.join(r['id'].split('_')[:3]) for r in syn}),
        'days': sorted({r['id'][:8] for r in syn}),
        'cams': {c: sum(r['cam'] == c for r in syn) for c in sorted({r['cam'] for r in syn})},
        'r_hist': np.histogram([float(r['r_fit_mm']) for r in syn], bins=np.arange(12, 46, 1))[0].tolist(),
        'depth_hist': np.histogram([float(r['depth_m']) for r in syn], bins=np.arange(0.2, 1.35, 0.05))[0].tolist(),
        'n_views': len(glob.glob(os.path.join(a.views, '*', '*', 'inst.json'))),
    }
    data['overlays'] = []
    for v in a.overlays:
        jpg, n = overlay(v)
        data['overlays'].append({'view': '/'.join(v.rstrip('/').split('/')[-2:]), 'jpg': jpg, 'n': n})
    json.dump(data, open(a.out, 'w'))
    print('wrote', a.out, os.path.getsize(a.out) // 1024, 'KB', len(data['samples']), 'samples')


if __name__ == '__main__':
    main()
