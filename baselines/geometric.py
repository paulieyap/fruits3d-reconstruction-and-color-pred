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
from scipy.spatial import ConvexHull

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


def fit_sphere(xyz, n_iter=200, inlier_thresh=0.002, seed=0):
    """RANSAC on 4-point algebraic fits, then a geometric (Huber) refinement on the inliers."""
    rng = _rng(seed)
    xyz = _subsample(xyz, FIT_POINTS, seed)
    best_c, best_r = _sphere_algebraic(xyz)
    best_inl = np.abs(np.linalg.norm(xyz - best_c, axis=1) - best_r) < inlier_thresh
    for _ in range(n_iter):
        idx = rng.choice(len(xyz), 4, replace=False)
        c, r = _sphere_algebraic(xyz[idx])
        if not np.isfinite(r) or r > 0.2:  # no fruit here is larger than 40 cm
            continue
        inl = np.abs(np.linalg.norm(xyz - c, axis=1) - r) < inlier_thresh
        if inl.sum() > best_inl.sum():
            best_c, best_r, best_inl = c, r, inl

    pts = xyz[best_inl] if best_inl.sum() >= 10 else xyz
    res = least_squares(lambda p: np.linalg.norm(pts - p[:3], axis=1) - p[3],
                        np.r_[best_c, best_r], loss='huber', f_scale=inlier_thresh)
    return res.x[:3], abs(res.x[3]), float(best_inl.mean())


def sphere(partial, n_points=N_POINTS, seed=0):
    xyz = np.asarray(partial)[:, :3]
    c, r, inlier_ratio = fit_sphere(xyz, seed=seed)
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


def fit_ellipsoid_axis_aligned(xyz, init_c, init_r, prior=0.05):
    """Ellipsoid with axes along the world axes: radial distance + Huber, weak sphere prior."""
    def resid(p):
        c, a = p[:3], np.abs(p[3:]) + 1e-6
        return np.r_[_radial_resid(xyz - c, a), _shape_prior(a, init_r, len(xyz), prior)]
    res = least_squares(resid, np.r_[init_c, [init_r] * 3], loss='huber', f_scale=0.002)
    return res.x[:3], np.eye(3), np.abs(res.x[3:])


def _rot(v):
    """Rotation matrix from a rotation vector (Rodrigues)."""
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def fit_ellipsoid_free(xyz, init_c, init_r, seed=0, prior=0.05):
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

    best = None
    for rv in starts:
        res = least_squares(resid, np.r_[init_c, rv, [init_r] * 3], loss='huber', f_scale=0.002)
        if best is None or res.cost < best.cost:
            best = res
    return best.x[:3], _rot(best.x[3:6]), np.abs(best.x[6:])


def _ellipsoid(partial, free, n_points, seed, max_ratio=2.0):
    xyz = np.asarray(partial)[:, :3]
    c0, r0, _ = fit_sphere(xyz, seed=seed)
    xyz = _subsample(xyz, FIT_POINTS, seed)
    if free:
        c, R, radii = fit_ellipsoid_free(xyz, c0, r0, seed=seed)
    else:
        c, R, radii = fit_ellipsoid_axis_aligned(xyz, c0, r0)
    # partial caps leave the unseen axis unconstrained; clamp radii relative to the sphere fit
    clamped = bool(np.any(radii > max_ratio * r0) or np.any(radii < r0 / max_ratio))
    radii = np.clip(radii, r0 / max_ratio, max_ratio * r0)
    pts = _ellipsoid_points(c, R, radii, n_points, _rng(seed))
    return Completion(pts, 4 / 3 * np.pi * radii.prod(),
                      {'center': c, 'rotation': R, 'radii': radii, 'clamped': clamped})


def ellipsoid_axis_aligned(partial, n_points=N_POINTS, seed=0):
    return _ellipsoid(partial, False, n_points, seed)


def ellipsoid_free(partial, n_points=N_POINTS, seed=0):
    return _ellipsoid(partial, True, n_points, seed)


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


METHODS = {
    'sphere': sphere,
    'ellipsoid_aa': ellipsoid_axis_aligned,
    'ellipsoid_free': ellipsoid_free,
    'symmetry_plane': symmetry_plane,
    'symmetry_point': symmetry_point,
    'convex_hull': convex_hull,
}
