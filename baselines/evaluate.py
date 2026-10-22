"""Evaluate shape completion methods per fruit: official challenge metrics, volume error, occlusion.

Inputs come from the sample cache (tools/build_cache.py), so every method sees the same fused
partial observation the DDPM is trained on. Run in pcdiff:cpu (or any pcdiff image):

  python3 -m baselines.evaluate --cache CACHE_DIR --split val --out OUT_DIR \
      [--methods sphere,ellipsoid_aa,...] [--pred ddpm=DIR_WITH_<fid>.ply ...] [--workers 8]

Writes OUT_DIR/results.csv (one row per fruit and method) and OUT_DIR/completions/<method>/<fid>.ply.

Metrics follow github.com/PRBonn/shape_completion_toolkit (vendored in pcdiff/competition_tools):
Chamfer distance = mean of the two mean nearest-neighbour distances (reported in mm), precision /
recall / F-score = normalised area under the curve over thresholds 1-10 mm (in %), plus all three
at a single 5 mm threshold.

Volume ground truth is PROVISIONAL: the GT is a point cloud, so its volume is taken from a
Poisson surface (normals re-estimated, oriented away from the centroid), with the convex hull of
the GT points reported as a cross-check. Predicted volumes use each method's own surface
(analytic sphere/ellipsoid, hull for symmetry/hull methods, Poisson for external point clouds)
and, for comparability, the convex hull of the predicted points.
"""
import csv
import multiprocessing
import os
import sys
import time

import click
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull, cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pcdiff', 'competition_tools'))
from chamfer_distance import ChamferDistance  # noqa: E402  (toolkit uses flat imports)
from precision_recall import PrecisionRecall  # noqa: E402

from baselines.geometric import METHODS  # noqa: E402

VISIBLE_THRESH = 0.005  # GT point counts as observed if a partial point lies within 5 mm
F_THRESH = 0.005

FIELDS = ['fruit_id', 'subset', 'method', 'cd_mm', 'precision', 'recall', 'fscore',
          'precision_5mm', 'recall_5mm', 'fscore_5mm',
          'vol_gt_ml', 'vol_gt_hull_ml', 'vol_pred_ml', 'vol_pred_hull_ml',
          'vol_err_ml', 'vol_err_pct', 'vol_hull_err_pct', 'visible_frac', 'n_partial', 'n_used', 'runtime_s']


def to_pcd(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))
    return pcd


def mesh_volume(mesh):
    """Signed volume by the divergence theorem; also usable when Poisson leaves small defects."""
    v = np.asarray(mesh.vertices)
    t = np.asarray(mesh.triangles)
    return abs(np.einsum('ij,ij->i', v[t[:, 0]], np.cross(v[t[:, 1]], v[t[:, 2]])).sum() / 6)


def poisson_volume(xyz, depth=8):
    pcd = to_pcd(xyz).voxel_down_sample(0.001)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30))
    n = np.asarray(pcd.normals)
    p = np.asarray(pcd.points)
    flip = np.einsum('ij,ij->i', n, p - p.mean(0)) < 0
    n[flip] *= -1
    pcd.normals = o3d.utility.Vector3dVector(n)
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    return mesh_volume(mesh)


def hull_volume(xyz):
    return float(ConvexHull(xyz).volume)


def score(gt, pred):
    cd = ChamferDistance()
    pr = PrecisionRecall(0.001, 0.01, 100)
    gt_pcd, pred_pcd = to_pcd(gt), to_pcd(pred)
    cd.update(gt_pcd, pred_pcd)
    pr.update(gt_pcd, pred_pcd)
    p, r, f = pr.compute_auc()
    p5, r5, f5, _ = pr.compute_at_threshold(F_THRESH)
    # toolkit precision/recall are already in percent
    return {'cd_mm': cd.compute() * 1000, 'precision': p, 'recall': r, 'fscore': f,
            'precision_5mm': p5, 'recall_5mm': r5, 'fscore_5mm': f5}


def evaluate_fruit(args):
    path, methods, preds, out = args
    fid = os.path.basename(path)[:-4]
    c = np.load(path)
    gt = c['gt_points'].astype(np.float64)
    partial = c['rgbd_pcd'][:, :3].astype(np.float64)

    vol_gt = poisson_volume(gt)
    common = {
        'fruit_id': fid,
        'subset': 'lab' if fid.startswith('lab') else 'greenhouse',
        'vol_gt_ml': vol_gt * 1e6,
        'vol_gt_hull_ml': hull_volume(gt) * 1e6,
        'visible_frac': float((cKDTree(partial).query(gt, k=1)[0] < VISIBLE_THRESH).mean()),
        'n_partial': len(partial),
    }

    rows = []
    todo = [(m, None) for m in methods] + list(preds.items())
    for name, pred_dir in todo:
        t = time.time()
        if pred_dir is None:
            comp = METHODS[name](partial)
            pts, vol = comp.points, comp.volume
            n_used = comp.params.get('n_used', len(partial))
        else:
            ply = os.path.join(pred_dir, fid + '.ply')
            if not os.path.isfile(ply):
                continue
            pts = np.asarray(o3d.io.read_point_cloud(ply).points)
            vol = poisson_volume(pts)
            n_used = len(partial)
        runtime = time.time() - t

        if pred_dir is None:
            d = os.path.join(out, 'completions', name)
            os.makedirs(d, exist_ok=True)
            o3d.io.write_point_cloud(os.path.join(d, fid + '.ply'), to_pcd(pts))

        vol_hull = hull_volume(pts)
        row = dict(common, method=name, **score(gt, pts),
                   vol_pred_ml=vol * 1e6, vol_pred_hull_ml=vol_hull * 1e6,
                   vol_err_ml=(vol - vol_gt) * 1e6, vol_err_pct=(vol / vol_gt - 1) * 100,
                   vol_hull_err_pct=(vol_hull / vol_gt - 1) * 100,
                   n_used=n_used, runtime_s=runtime if pred_dir is None else float('nan'))
        rows.append(row)
    return rows


@click.command()
@click.option('--cache', required=True, help='cache dir with <split>/<fid>.npz')
@click.option('--split', default='val')
@click.option('--out', required=True)
@click.option('--methods', default=','.join(METHODS), help='comma-separated geometric methods ("" for none)')
@click.option('--pred', multiple=True, help='NAME=DIR of <fid>.ply completions from another method (e.g. the DDPM)')
@click.option('--workers', type=int, default=8)
def main(cache, split, out, methods, pred, workers):
    methods = [m for m in methods.split(',') if m]
    preds = dict(p.split('=', 1) for p in pred)
    os.makedirs(out, exist_ok=True)
    split_dir = os.path.join(cache, split)
    paths = sorted(os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith('.npz') and '.tmp' not in f)

    jobs = [(p, methods, preds, out) for p in paths]
    rows = []
    with multiprocessing.get_context('spawn').Pool(workers) as pool:
        for k, fruit_rows in enumerate(pool.imap_unordered(evaluate_fruit, jobs), 1):
            rows += fruit_rows
            print(f'[{k}/{len(jobs)}] {fruit_rows[0]["fruit_id"] if fruit_rows else "?"}', flush=True)

    rows.sort(key=lambda r: (r['method'], r['fruit_id']))
    with open(os.path.join(out, 'results.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {len(rows)} rows to {os.path.join(out, "results.csv")}')


if __name__ == '__main__':
    main()
