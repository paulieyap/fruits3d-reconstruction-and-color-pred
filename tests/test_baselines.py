"""Synthetic checks for the geometric baselines (no dataset needed):

  python3 -m pytest tests/test_baselines.py      (or: python3 tests/test_baselines.py)
"""
import numpy as np

from baselines import METHODS, N_POINTS


def cap(radii, frac, rng, n=20000, R=np.eye(3), c=np.array([0.1, -0.05, 0.4]), noise=0.0005):
    """Points on an ellipsoid surface seen from -z: keep the `frac` of it facing the camera."""
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    u = u[u[:, 2] < np.quantile(u[:, 2], frac)]
    return (u * radii) @ R.T + c + rng.normal(scale=noise, size=(len(u), 3))


def vol(radii):
    return 4 / 3 * np.pi * np.prod(radii)


def test_shapes_and_volume_on_sphere_cap():
    rng = np.random.default_rng(0)
    r = np.array([0.03, 0.03, 0.03])
    partial = cap(r, 0.4, rng)
    for name, f in METHODS.items():
        out = f(partial)
        assert out.points.shape == (N_POINTS, 3), name
        assert np.isfinite(out.points).all(), name
        err = out.volume / vol(r) - 1
        if name == 'convex_hull':
            assert out.volume < vol(r), name  # lower bound
        else:
            assert abs(err) < 0.15, (name, err)


def test_sphere_radius_and_hull_lower_bound_on_ellipsoid():
    rng = np.random.default_rng(1)
    r = np.array([0.035, 0.03, 0.025])
    partial = cap(r, 0.5, rng)
    s = METHODS['sphere'](partial)
    assert 0.02 < s.params['radius'] < 0.04
    assert METHODS['convex_hull'](partial).volume < vol(r)
    e = METHODS['ellipsoid_aa'](partial)
    assert abs(e.volume / vol(r) - 1) < 0.25, e.volume / vol(r)


if __name__ == '__main__':
    test_shapes_and_volume_on_sphere_cap()
    test_sphere_radius_and_hull_lower_bound_on_ellipsoid()
    rng = np.random.default_rng(0)
    for r, frac in [((0.03,) * 3, 0.4), ((0.035, 0.03, 0.025), 0.5), ((0.035, 0.03, 0.025), 0.25)]:
        r = np.array(r); p = cap(r, frac, rng)
        print(r, frac, {k: round(f(p).volume / vol(r), 3) for k, f in METHODS.items()})
    print('ok')
