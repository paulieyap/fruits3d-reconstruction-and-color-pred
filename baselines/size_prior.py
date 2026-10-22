"""Fruit size prior for the *_prior baselines, estimated from a labelled split (never from the one
being evaluated): equivalent-sphere radius r = (3V / 4pi)^(1/3) of each GT (Poisson volume),
range = [min, max] widened by `margin`.

  python3 -m baselines.size_prior --cache CACHE_DIR --split train --out size_prior.json
"""
import json
import multiprocessing
import os

import click
import numpy as np

from baselines.evaluate import poisson_volume


def radius(path):
    v = poisson_volume(np.load(path)['gt_points'].astype(np.float64))
    return os.path.basename(path)[:-4], float((3 * v / (4 * np.pi)) ** (1 / 3))


@click.command()
@click.option('--cache', required=True)
@click.option('--split', default='train')
@click.option('--out', required=True)
@click.option('--margin', default=0.1, help='relative widening of the observed radius range')
@click.option('--workers', type=int, default=8)
def main(cache, split, out, margin, workers):
    d = os.path.join(cache, split)
    paths = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith('.npz') and '.tmp' not in f)
    with multiprocessing.get_context('spawn').Pool(workers) as pool:
        radii = dict(pool.map(radius, paths))
    r = np.array(list(radii.values()))
    prior = {'split': split, 'n': len(r), 'margin': margin,
             'r_min': float(r.min() * (1 - margin)), 'r_max': float(r.max() * (1 + margin)),
             'r_observed': [float(r.min()), float(np.median(r)), float(r.max())], 'radii': radii}
    with open(out, 'w') as f:
        json.dump(prior, f, indent=1)
    print(f"{split}: n={len(r)} r min/median/max = {r.min()*1000:.1f}/{np.median(r)*1000:.1f}/{r.max()*1000:.1f} mm "
          f"-> prior [{prior['r_min']*1000:.1f}, {prior['r_max']*1000:.1f}] mm")


if __name__ == '__main__':
    main()
