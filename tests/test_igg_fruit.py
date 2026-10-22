"""Checks the IGGFruit replacement on real challenge data.

Needs the dataset (challenge layout) at $SC_DATA, e.g. in the pcdiff:cpu container:

  docker run --rm -e SC_DATA=/data -v <root>:/data:ro -v <repo>:/workspace -w /workspace pcdiff:cpu \
      python3 tests/test_igg_fruit.py

Skipped when $SC_DATA is not set.
"""
import os

import numpy as np
from scipy.spatial import cKDTree

from pcdiff.datasets.igg_fruit import IGGFruit

ROOT = os.environ.get('SC_DATA')
CASES = [('val', 'lab15'), ('val', 'p12'), ('test', 'p1')]


def load(split, fid):
    ds = IGGFruit(data_source=ROOT, split=split)
    idx = [s[1] for s in ds.samples].index(fid)
    return ds[idx]


def check(split, fid):
    sample = load(split, fid)
    part, gt = sample['points'], sample['extra']['gt_points']
    assert sample['extra']['fruit_id'] == fid

    # (N, 6) float XYZRGB partial; colours in [0, 1]
    assert part.ndim == 2 and part.shape[1] == 6 and len(part) > 1000, part.shape
    assert part.dtype == np.float64
    assert np.isfinite(part).all()
    assert part[:, 3:].min() >= 0 and part[:, 3:].max() <= 1

    # (M, 6) GT; colours in [0, 1] or exactly -1 where nothing was transferred
    assert gt.ndim == 2 and gt.shape[1] == 6, gt.shape
    col = gt[:, 3:]
    assert np.all(((col >= 0) & (col <= 1)) | (col == -1))

    # metres, same world frame: a fruit is a few cm across and the partial lies on the GT surface
    extent = np.ptp(gt[:, :3], axis=0)
    assert np.all(extent > 0.02) and np.all(extent < 0.3), extent
    d, _ = cKDTree(gt[:, :3]).query(part[:, :3], k=1)
    med = np.median(d)
    if split != 'test':
        assert med < 0.005, f'{fid}: median partial-to-GT distance {med * 1000:.1f} mm'
        assert (col[:, 0] >= 0).any(), 'no GT point got a colour'
    else:
        assert med == 0, 'test split: GT falls back to the partial'
    print(f'{split}/{fid}: partial {part.shape}, gt {gt.shape}, extent {np.round(extent * 1000)} mm, '
          f'median partial-GT {med * 1000:.2f} mm, coloured GT {np.mean(col[:, 0] >= 0):.2f}')


def test_igg_fruit():
    if ROOT is None:
        import pytest
        pytest.skip('SC_DATA not set')
    for split, fid in CASES:
        check(split, fid)


if __name__ == '__main__':
    assert ROOT, 'set SC_DATA to the dataset root'
    for split, fid in CASES:
        check(split, fid)
    print('ok')
