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


def test_size_prior_bounds_radius():
    from baselines.geometric import prior_methods
    rng = np.random.default_rng(2)
    # a flat, noisy patch: unconstrained fits run off to huge radii
    partial = np.c_[rng.uniform(-0.03, 0.03, (5000, 2)), rng.normal(0, 0.002, 5000)]
    for name, f in prior_methods(0.03, 0.05).items():
        out = f(partial)
        if 'radii' in out.params and 'e1' in out.params:  # superquadric: volume also depends on the exponents
            assert np.all(out.params['radii'] <= 0.05 + 1e-9) and np.all(out.params['radii'] >= 0.03 - 1e-9), (name, out.params)
            continue
        assert out.volume <= 4 / 3 * np.pi * 0.05 ** 3 * 1.001, (name, out.volume)
        assert out.volume >= 4 / 3 * np.pi * 0.03 ** 3 * 0.999, (name, out.volume)


def test_superquadric_volume_and_boxy_shape():
    from baselines.geometric import superquadric_volume, _superquadric_points, superquadric
    a = np.array([0.03, 0.025, 0.02])
    assert abs(superquadric_volume(a, 1, 1) / vol(a) - 1) < 1e-9  # ellipsoid limit
    # Monte-Carlo check of the closed form for a boxy superquadric
    rng = np.random.default_rng(3)
    q = rng.uniform(-1, 1, (400000, 3)) * a
    from baselines.geometric import _sq_inside
    mc = (_sq_inside(q, a, 0.5, 0.6) <= 1).mean() * 8 * a.prod()
    assert abs(superquadric_volume(a, 0.5, 0.6) / mc - 1) < 0.02
    pts = _superquadric_points(np.zeros(3), np.eye(3), a, 0.5, 0.6, 20000, rng)
    V = superquadric_volume(a, 0.5, 0.6)
    # fully visible: shape recovered
    full = superquadric(pts + rng.normal(scale=0.0003, size=pts.shape))
    assert abs(full.params['e1'] - 0.5) < 0.1 and abs(full.params['e2'] - 0.6) < 0.1, full.params
    assert abs(full.volume / V - 1) < 0.03, full.volume / V
    # 57 % visible (top hidden): exponents still move below 1; the unseen radius is pulled towards the
    # mean radius by the shape prior, so volume is only within 25 %
    m = pts[:, 2] < 0.005
    out = superquadric(pts[m] + rng.normal(scale=0.0003, size=(int(m.sum()), 3)))
    assert out.params['e1'] < 0.95 and out.params['e2'] < 0.95, out.params
    assert abs(out.volume / V - 1) < 0.25, out.volume / V


if __name__ == '__main__':
    test_superquadric_volume_and_boxy_shape()
    test_size_prior_bounds_radius()
    test_shapes_and_volume_on_sphere_cap()
    test_sphere_radius_and_hull_lower_bound_on_ellipsoid()
    rng = np.random.default_rng(0)
    for r, frac in [((0.03,) * 3, 0.4), ((0.035, 0.03, 0.025), 0.5), ((0.035, 0.03, 0.025), 0.25)]:
        r = np.array(r); p = cap(r, frac, rng)
        print(r, frac, {k: round(f(p).volume / vol(r), 3) for k, f in METHODS.items()})
    print('ok')
