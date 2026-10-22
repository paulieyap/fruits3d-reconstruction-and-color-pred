"""Cheap geometric shape completion baselines for (roughly) round fruits.

Every method has the same interface as the DDPM pipeline's output:

    completion = METHODS[name](partial)      # partial: (N, 3+) array, XYZ in metres (extra columns ignored)
    completion.points                        # (n_points, 3) points on the completed surface
    completion.volume                        # enclosed volume in m^3 of the method's own surface
    completion.params                        # fitted parameters (centre, radii, ...) for inspection

`points` is sampled on the completed surface with the same number of points as the
DDPM output (30000), so all methods go through the same metric code.
"""
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull, cKDTree

N_POINTS = 30000
FIT_POINTS = 20000  # points used by the iterative fits


@dataclass
class Completion:
    points: np.ndarray
    volume: float
    params: dict = field(default_factory=dict)


def _rng(seed=0):
    return np.random.default_rng(seed)


def _subsample(xyz, n, seed):
    """Random subset for the iterative fits; fused partials have up to ~1M points."""
    if len(xyz) <= n:
        return xyz
    return xyz[_rng(seed).choice(len(xyz), n, replace=False)]


def _unit_sphere(n, rng):
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ---------------------------------------------------------------- sphere

def _sphere_algebraic(xyz):
    """Linear least squares: |p|^2 = 2 c.p + (r^2 - |c|^2)."""
    A = np.c_[2 * xyz, np.ones(len(xyz))]
    b = (xyz ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r = np.sqrt(max(sol[3] + c @ c, 1e-12))
    return c, r


def _bounds_x0(x0, lb, ub):
    """least_squares needs a start strictly inside the bounds."""
    eps = 1e-6
    return np.minimum(np.maximum(x0, np.asarray(lb) + eps), np.asarray(ub) - eps)


def fit_sphere(xyz, n_iter=200, inlier_thresh=0.002, seed=0, r_range=None):
    """RANSAC on 4-point algebraic fits, then a geometric (Huber) refinement on the inliers.
    With r_range=(r_min, r_max) (size prior), candidates outside the range are rejected and the
    refinement is bounded."""
    r_lo, r_hi = r_range if r_range is not None else (0.0, 0.2)  # no fruit here is larger than 40 cm
    rng = _rng(seed)
    xyz = _subsample(xyz, FIT_POINTS, seed)
    best_c, best_r = _sphere_algebraic(xyz)
    best_inl = np.abs(np.linalg.norm(xyz - best_c, axis=1) - best_r) < inlier_thresh
    if not r_lo <= best_r <= r_hi:
        best_inl[:] = False
    for _ in range(n_iter):
        idx = rng.choice(len(xyz), 4, replace=False)
        c, r = _sphere_algebraic(xyz[idx])
        if not np.isfinite(r) or not r_lo <= r <= r_hi:
            continue
        inl = np.abs(np.linalg.norm(xyz - c, axis=1) - r) < inlier_thresh
        if inl.sum() > best_inl.sum():
            best_c, best_r, best_inl = c, r, inl

    pts = xyz[best_inl] if best_inl.sum() >= 10 else xyz
    if r_range is None:
        lb, ub = -np.inf, np.inf
        x0 = np.r_[best_c, best_r]
    else:
        lb, ub = [-np.inf] * 3 + [r_lo], [np.inf] * 3 + [r_hi]
        x0 = _bounds_x0(np.r_[best_c, best_r], lb, ub)
    res = least_squares(lambda p: np.linalg.norm(pts - p[:3], axis=1) - p[3],
                        x0, bounds=(lb, ub), loss='huber', f_scale=inlier_thresh)
    return res.x[:3], abs(res.x[3]), float(best_inl.mean())


def sphere(partial, n_points=N_POINTS, seed=0, r_range=None):
    xyz = np.asarray(partial)[:, :3]
    c, r, inlier_ratio = fit_sphere(xyz, seed=seed, r_range=r_range)
    pts = c + r * _unit_sphere(n_points, _rng(seed))
    return Completion(pts, 4 / 3 * np.pi * r ** 3, {'center': c, 'radius': r, 'inlier_ratio': inlier_ratio})


# ---------------------------------------------------------------- ellipsoid

def _ellipsoid_points(c, R, radii, n, rng):
    # area-weighted sampling via rejection on the unit sphere (uniform on the ellipsoid surface)
    out = []
    while sum(len(o) for o in out) < n:
        u = _unit_sphere(4 * n, rng)
        # surface area element of x = radii * u is proportional to |u / radii| * prod(radii)
        w = np.linalg.norm(u / radii, axis=1) * radii.prod()
        keep = rng.random(len(u)) < w / w.max()
        out.append(u[keep] * radii)
    p = np.concatenate(out)[:n]
    return p @ R.T + c


def _radial_resid(q, a):
    """Distance (m) from each point to the ellipsoid surface along the ray from the centre."""
    s = np.sqrt(((q / a) ** 2).sum(1))
    return np.linalg.norm(q, axis=1) * (1 - 1 / np.maximum(s, 1e-9))


def _shape_prior(a, r0, n, weight):
    """Weak pull of the radii towards the sphere radius: a partial cap leaves the unseen axis
    almost unconstrained, and fruits are close to spheres. Scaled so that 1 mm of radius
    deviation costs `weight` mm on every point."""
    return weight * np.sqrt(n) * (a - r0)


def fit_ellipsoid_axis_aligned(xyz, init_c, init_r, prior=0.05, r_range=None):
    """Ellipsoid with axes along the world axes: radial distance + Huber, weak sphere prior,
    radii bounded by r_range (size prior) if given."""
    def resid(p):
        c, a = p[:3], np.abs(p[3:]) + 1e-6
        return np.r_[_radial_resid(xyz - c, a), _shape_prior(a, init_r, len(xyz), prior)]
    x0 = np.r_[init_c, [init_r] * 3]
    lb, ub = -np.inf, np.inf
    if r_range is not None:
        lb, ub = [-np.inf] * 3 + [r_range[0]] * 3, [np.inf] * 3 + [r_range[1]] * 3
        x0 = _bounds_x0(x0, lb, ub)
    res = least_squares(resid, x0, bounds=(lb, ub), loss='huber', f_scale=0.002)
    return res.x[:3], np.eye(3), np.abs(res.x[3:])


def _rot(v):
    """Rotation matrix from a rotation vector (Rodrigues)."""
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def fit_ellipsoid_free(xyz, init_c, init_r, seed=0, prior=0.05, r_range=None):
    """Free-orientation ellipsoid: centre, 3 radii and a rotation, started from the sphere fit
    and a few PCA-aligned orientations (partial views make this fit poorly conditioned)."""
    def resid(p):
        c, R, a = p[:3], _rot(p[3:6]), np.abs(p[6:]) + 1e-6
        return np.r_[_radial_resid((xyz - c) @ R, a), _shape_prior(a, init_r, len(xyz), prior)]

    # PCA of the partial gives candidate axes; try the identity too
    _, _, Vt = np.linalg.svd(xyz - xyz.mean(0), full_matrices=False)
    starts = [np.zeros(3)]
    R0 = Vt.T if np.linalg.det(Vt.T) > 0 else Vt.T @ np.diag([1, 1, -1])
    # rotation vector of R0
    ang = np.arccos(np.clip((np.trace(R0) - 1) / 2, -1, 1))
    if ang > 1e-6:
        axis = np.array([R0[2, 1] - R0[1, 2], R0[0, 2] - R0[2, 0], R0[1, 0] - R0[0, 1]]) / (2 * np.sin(ang))
        starts.append(axis * ang)

    lb, ub = -np.inf, np.inf
    if r_range is not None:
        lb, ub = [-np.inf] * 6 + [r_range[0]] * 3, [np.inf] * 6 + [r_range[1]] * 3
    best = None
    for rv in starts:
        x0 = np.r_[init_c, rv, [init_r] * 3]
        if r_range is not None:
            x0 = _bounds_x0(x0, lb, ub)
        res = least_squares(resid, x0, bounds=(lb, ub), loss='huber', f_scale=0.002)
        if best is None or res.cost < best.cost:
            best = res
    return best.x[:3], _rot(best.x[3:6]), np.abs(best.x[6:])


def _ellipsoid(partial, free, n_points, seed, max_ratio=2.0, r_range=None):
    xyz = np.asarray(partial)[:, :3]
    c0, r0, _ = fit_sphere(xyz, seed=seed, r_range=r_range)
    xyz = _subsample(xyz, FIT_POINTS, seed)
    if free:
        c, R, radii = fit_ellipsoid_free(xyz, c0, r0, seed=seed, r_range=r_range)
    else:
        c, R, radii = fit_ellipsoid_axis_aligned(xyz, c0, r0, r_range=r_range)
    # partial caps leave the unseen axis unconstrained; clamp radii relative to the sphere fit
    clamped = bool(np.any(radii > max_ratio * r0) or np.any(radii < r0 / max_ratio))
    radii = np.clip(radii, r0 / max_ratio, max_ratio * r0)
    pts = _ellipsoid_points(c, R, radii, n_points, _rng(seed))
    return Completion(pts, 4 / 3 * np.pi * radii.prod(),
                      {'center': c, 'rotation': R, 'radii': radii, 'clamped': clamped})


def ellipsoid_axis_aligned(partial, n_points=N_POINTS, seed=0, r_range=None):
    return _ellipsoid(partial, False, n_points, seed, r_range=r_range)


def ellipsoid_free(partial, n_points=N_POINTS, seed=0, r_range=None):
    return _ellipsoid(partial, True, n_points, seed, r_range=r_range)


# ---------------------------------------------------------------- symmetry

def _hull_completion(xyz, n_points, seed, params):
    hull = ConvexHull(xyz)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(xyz),
                                     o3d.utility.Vector3iVector(hull.simplices))
    o3d.utility.random.seed(seed)
    pts = np.asarray(mesh.sample_points_uniformly(n_points).points)
    return Completion(pts, float(hull.volume), params)


def symmetry_plane(partial, n_points=N_POINTS, seed=0):
    """Mirror the partial view about a plane through the fitted centre.

    The visible cap's centroid lies between the camera(s) and the fruit centre, so the plane
    normal is taken along (centre - centroid); this needs no camera poses. The completed
    surface is the convex hull of partial + mirrored points.
    """
    xyz = np.asarray(partial)[:, :3]
    c, r, _ = fit_sphere(xyz, seed=seed)
    d = c - xyz.mean(0)
    n = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-9 else np.array([0., 0., 1.])
    mirrored = xyz - 2 * ((xyz - c) @ n)[:, None] * n
    return _hull_completion(np.r_[xyz, mirrored], n_points, seed, {'center': c, 'normal': n})


def symmetry_point(partial, n_points=N_POINTS, seed=0):
    """Point reflection through the fitted centre (central symmetry), then convex hull."""
    xyz = np.asarray(partial)[:, :3]
    c, r, _ = fit_sphere(xyz, seed=seed)
    return _hull_completion(np.r_[xyz, 2 * c - xyz], n_points, seed, {'center': c})


# ---------------------------------------------------------------- convex hull

def convex_hull(partial, n_points=N_POINTS, seed=0):
    """Convex hull of the partial observation only: a lower bound, no completion at all."""
    return _hull_completion(np.asarray(partial)[:, :3], n_points, seed, {})


# ---------------------------------------------------------------- input cleaning

def largest_cluster(xyz, voxel=0.002, eps=0.006, min_points=5):
    """Keep the largest connected cluster of the partial view.

    Fused greenhouse views contain small, sparse clusters from neighbouring fruits or leaves
    10-15 cm away; they inflate hulls and pull the fits. DBSCAN runs on a 2 mm voxel grid, and
    every original point near a kept voxel survives.
    """
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz)).voxel_down_sample(voxel)
    labels = np.asarray(pcd.cluster_dbscan(eps, min_points))
    if labels.max() < 0:
        return xyz
    keep = np.asarray(pcd.points)[labels == np.bincount(labels[labels >= 0]).argmax()]
    d, _ = cKDTree(keep).query(xyz, k=1)
    return xyz[d < 1.5 * voxel]


def _cleaned(method, **kwargs):
    def run(partial, n_points=N_POINTS, seed=0):
        xyz = largest_cluster(np.asarray(partial, dtype=np.float64)[:, :3])
        comp = method(xyz, n_points=n_points, seed=seed, **kwargs)
        comp.params['n_input'] = len(partial)
        comp.params['n_used'] = len(xyz)
        return comp
    run.__doc__ = method.__doc__
    return run


# all methods see the partial view after largest_cluster(); the DDPM gets it unfiltered
METHODS = {
    'sphere': _cleaned(sphere),
    'ellipsoid_aa': _cleaned(ellipsoid_axis_aligned),
    'ellipsoid_free': _cleaned(ellipsoid_free),
    'symmetry_plane': _cleaned(symmetry_plane),
    'symmetry_point': _cleaned(symmetry_point),
    'convex_hull': _cleaned(convex_hull),
}


def prior_methods(r_min, r_max):
    """Variants with a fruit size prior: every fitted radius is bounded to [r_min, r_max] (m).
    Estimate the range on a split you do not evaluate on (baselines/size_prior.py)."""
    r = (r_min, r_max)
    return {
        'sphere_prior': _cleaned(sphere, r_range=r),
        'ellipsoid_aa_prior': _cleaned(ellipsoid_axis_aligned, r_range=r),
        'ellipsoid_free_prior': _cleaned(ellipsoid_free, r_range=r),
    }
