"""Per-tomato shape-completion samples from harvester snapshot views.

Every SAM 3 tomato instance in a wrist / stacker view gives a REAL partial point
cloud (stereo depth inside the mask, real RGB). The harvester never sees the back
of a fruit, so complete ground truth comes from a procedural tomato placed where
the real one is (sphere fit on the real partial: centre + radius), oriented
stem-up in the world frame, and ray-cast from the REAL camera through the REAL
mask pixels with ZED-like depth noise. Input colours are the real image pixels.

Outputs (pcdiff cache layout, camera optical frame, metres):
  <out>/syn/<split>/<id>.npz   rgbd_pcd (N,6) synthetic-geometry partial + real RGB
                               aligned_data (M,6) complete GT, RGB from partial (<=1 cm) else -1
                               gt_points, gt_colors, real_partial (K,6) for reference, meta json
  <out>/real/<id>.npz          real wrist partial, matched stacker partial (held-out view) in the
                               wrist camera frame, both RGB; no complete GT exists
  <out>/index.csv              one row per instance with fit stats
Split is by snapshot (no fruit leaks between train / val).
"""
import argparse, glob, json, os, zlib
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import open3d as o3d

N_GT = 8192
MAX_PART = 12000


# ---------------------------------------------------------------- geometry

def backproject(depth, K, ys, xs):
    z = depth[ys, xs]
    x = (xs - K[0, 2]) * z / K[0, 0]
    y = (ys - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], 1)


def fit_sphere(p, iters=3):
    """Algebraic sphere fit with residual-based reweighting. Returns centre, radius, rms."""
    w = np.ones(len(p))
    for _ in range(iters):
        A = np.c_[2 * p, np.ones(len(p))]
        b = (p ** 2).sum(1)
        sw = np.sqrt(w)
        sol, *_ = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)
        c = sol[:3]
        r = np.sqrt(max(sol[3] + c @ c, 1e-12))
        res = np.linalg.norm(p - c, axis=1) - r
        s = 1.4826 * np.median(np.abs(res)) + 1e-4
        w = 1.0 / (1.0 + (res / (2 * s)) ** 2)
    return c, r, float(np.sqrt(np.mean(res ** 2)))


def tomato_mesh(rng, radius):
    """Procedural tomato: oblate spheroid, lobes, stem dimple, smooth noise. Local z = stem axis (up)."""
    m = o3d.geometry.TriangleMesh.create_sphere(1.0, resolution=48)
    v = np.asarray(m.vertices)
    th = np.arccos(np.clip(v[:, 2], -1, 1))  # 0 at stem end
    ph = np.arctan2(v[:, 1], v[:, 0])
    k = rng.uniform(0.72, 0.97)  # height / width
    n_lobe = rng.integers(4, 8)
    a_lobe = rng.uniform(0.0, 0.07)
    lobe = 1 + a_lobe * np.cos(n_lobe * ph + rng.uniform(0, 2 * np.pi)) * np.sin(th) ** 2 * (1 - 0.5 * np.cos(th))
    dimple = 1 - rng.uniform(0.0, 0.18) * np.exp(-(th / rng.uniform(0.25, 0.5)) ** 2)
    blossom = 1 - rng.uniform(0.0, 0.06) * np.exp(-((np.pi - th) / 0.5) ** 2)
    noise = np.ones(len(v))
    for _ in range(4):  # a few random low-order bumps
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        noise += rng.normal(0, 0.015) * np.exp(-(1 - v @ d) * rng.uniform(1.5, 4))
    s = lobe * dimple * blossom * noise
    out = v * s[:, None]
    out[:, 2] *= k
    out *= radius / np.percentile(np.linalg.norm(out[:, :2], axis=1), 95)  # radius = equatorial half-width
    out[:, 0] *= rng.uniform(0.94, 1.06)
    m.vertices = o3d.utility.Vector3dVector(out)
    m.compute_vertex_normals()
    return m


def random_up_rotation(rng, up, max_tilt_deg=35):
    """Rotation mapping local z to a direction within max_tilt of `up`, random spin."""
    tilt = np.deg2rad(rng.uniform(0, max_tilt_deg))
    az = rng.uniform(0, 2 * np.pi)
    a = np.cross(up, [1, 0, 0] if abs(up[0]) < 0.9 else [0, 1, 0])
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    z = np.cos(tilt) * up + np.sin(tilt) * (np.cos(az) * a + np.sin(az) * b)
    x = np.cross(z, a if abs(z @ a) < 0.9 else b)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.c_[x, y, z]
    spin = rng.uniform(0, 2 * np.pi)
    Rz = np.array([[np.cos(spin), -np.sin(spin), 0], [np.sin(spin), np.cos(spin), 0], [0, 0, 1]])
    return R @ Rz


# ---------------------------------------------------------------- per view

def load_view(vdir):
    K = np.array(json.load(open(os.path.join(vdir, "intrinsic.json")))["K"])
    T = np.loadtxt(os.path.join(vdir, "pose.txt"))
    depth = np.load(os.path.join(vdir, "depth.npy")).astype(np.float32)
    img = cv2.imread(os.path.join(vdir, "color.jpg"))[:, :, ::-1]
    inst = cv2.imread(os.path.join(vdir, "inst.png"), cv2.IMREAD_UNCHANGED)
    scores = json.load(open(os.path.join(vdir, "inst.json")))["scores"]
    return K, T, depth, img, inst, scores


def real_instances(vdir, a):
    """Cleaned real partial clouds (camera frame) for the usable instances of one view."""
    K, T, depth, img, inst, scores = load_view(vdir)
    H, W = depth.shape
    out = []
    ker = np.ones((5, 5), np.uint8)
    for k, sc in enumerate(scores, 1):
        if sc < a.min_score:
            continue
        m = inst == k
        area = int(m.sum())
        if area < a.min_area:
            continue
        ys0, xs0 = np.nonzero(m)
        if ys0.min() < 3 or xs0.min() < 3 or ys0.max() > H - 4 or xs0.max() > W - 4:
            continue  # cut by the image border
        me = cv2.erode(m.astype(np.uint8), ker) > 0
        ys, xs = np.nonzero(me & np.isfinite(depth) & (depth > 0.1))
        if len(ys) < a.min_points:
            continue
        p = backproject(depth, K, ys, xs)
        zmed = np.median(p[:, 2])
        if zmed > a.max_depth:
            continue
        keep = np.abs(p[:, 2] - zmed) < 0.05
        p, ys, xs = p[keep], ys[keep], xs[keep]
        if len(p) < a.min_points:
            continue
        c, r, rms = fit_sphere(p)
        ok = (np.linalg.norm(p - c, axis=1) - r) < 0.02  # drop points far outside the fruit
        p, ys, xs = p[ok], ys[ok], xs[ok]
        if len(p) < a.min_points or not (a.min_r < r < a.max_r) or rms > 0.35 * r or c[2] < zmed:
            continue
        rgb = img[ys, xs].astype(np.float32) / 255.0
        out.append(dict(k=k, score=float(sc), area=area, c=c, r=float(r), rms=rms, pts=p, rgb=rgb,
                        ys=ys, xs=xs, mask=m, n=len(p), zmed=float(zmed)))
    return (K, T, depth, img), out


def synth_sample(rng, K, T, img, inst, a):
    """Procedural tomato at the real fruit, ray-cast through the real mask pixels."""
    r = inst["r"] * rng.uniform(0.9, 1.1)
    c = inst["c"].copy()
    mesh = tomato_mesh(rng, r)
    up_cam = T[:3, :3].T @ np.array([0.0, 0.0, 1.0])  # world up in the camera frame
    R = random_up_rotation(rng, up_cam)
    # place the centre so the front surface sits where the real points are
    mesh.rotate(R, center=(0, 0, 0))
    mesh.translate(c)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    ys, xs = np.nonzero(inst["mask"])
    dirs = np.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], np.ones(len(xs))], 1)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    rays = np.c_[np.zeros_like(dirs), dirs].astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(rays))
    t = hit["t_hit"].numpy()
    ok = np.isfinite(t)
    if ok.sum() < a.min_points:
        return None
    p = dirs[ok] * t[ok, None]
    z = p[:, 2]
    # ZED-like noise: depth sigma grows ~z^2, plus smooth low-frequency bias and a few flying pixels
    sig = rng.uniform(0.6, 1.6) * 1e-3 * (z / 0.5) ** 2
    zn = z + rng.normal(size=len(z)) * sig
    zn += rng.normal(0, 0.6e-3) * ((ys[ok] - ys.mean()) / 50.0) + rng.normal(0, 0.6e-3) * ((xs[ok] - xs.mean()) / 50.0)
    fly = rng.random(len(z)) < rng.uniform(0.0, 0.015)
    zn[fly] += rng.uniform(-0.008, 0.008, fly.sum())
    p = p * (zn / z)[:, None]
    rgb = img[ys[ok], xs[ok]].astype(np.float32) / 255.0
    part = np.c_[p, rgb]
    if len(part) > MAX_PART:
        part = part[rng.choice(len(part), MAX_PART, replace=False)]
    gt = np.asarray(mesh.sample_points_uniformly(N_GT).points)
    # GT colours from the nearest partial point within 5 mm (thesis used 1 cm, which colours most of a
    # 2 cm-radius tomato from a front view), else -1
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(part[:, :3]))
    tree = o3d.geometry.KDTreeFlann(pc)
    col = -np.ones((len(gt), 3))
    for i, q in enumerate(gt):
        _, idx, d2 = tree.search_knn_vector_3d(q, 1)
        if d2[0] < 2.5e-5:
            col[i] = part[idx[0], 3:]
    return part.astype(np.float32), np.c_[gt, col].astype(np.float32), dict(radius=r, centre=c.tolist(), R=R.tolist())


def sub(x, n, rng):
    return x if len(x) <= n else x[rng.choice(len(x), n, replace=False)]


def align_views(Tw, Ts, wi, si, min_support=5):
    """Stacker->wrist camera transform. The TF chain leaves a per-snapshot offset of ~5-15 cm
    between the arm and stacker cameras, so the translation is voted from fruit centres and
    refined with ICP on the fruit points. None if no consistent offset is found."""
    W = np.array([Tw[:3, :3] @ w["c"] + Tw[:3, 3] for w in wi])
    S = np.array([Ts[:3, :3] @ s["c"] + Ts[:3, 3] for s in si])
    D = (S[None] - W[:, None]).reshape(-1, 3)
    D = D[np.linalg.norm(D, axis=1) < 0.3]
    if not len(D):
        return None
    cnt = np.array([(np.linalg.norm(D - d, axis=1) < 0.015).sum() for d in D])
    if cnt.max() < min_support:
        return None
    off = np.median(D[np.linalg.norm(D - D[cnt.argmax()], axis=1) < 0.015], axis=0)
    Td = np.eye(4)
    Td[:3, 3] = -off
    T0 = np.linalg.inv(Tw) @ Td @ Ts
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate([s["pts"] for s in si])))
    dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate([w["pts"] for w in wi])))
    src, dst = src.voxel_down_sample(0.003), dst.voxel_down_sample(0.003)
    reg = o3d.pipelines.registration.registration_icp(
        src, dst, 0.012, T0, o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
    if reg.fitness < 0.15:
        return None
    return reg.transformation


def process_snapshot(args):
    snap, a = args
    rows = []
    split = "val" if (zlib.crc32(os.path.basename(snap).encode()) % 100) < a.val_pct else "train"
    rng = np.random.default_rng(zlib.crc32(snap.encode()))
    views = {}
    for vdir in sorted(glob.glob(os.path.join(snap, "*"))):
        if not os.path.isfile(os.path.join(vdir, "inst.json")):
            continue
        try:
            views[os.path.basename(vdir)] = real_instances(vdir, a)
        except Exception as e:  # corrupt view: skip, keep the rest
            print("ERR", vdir, e, flush=True)
    sid = os.path.basename(snap).replace("execution_snapshot_", "")
    for cam, ((K, T, depth, img), insts) in views.items():
        for inst in insts:
            for rep in range(a.syn_per_fruit):
                s = synth_sample(rng, K, T, img, inst, a)
                if s is None:
                    continue
                part, gt, meta = s
                fid = f"{sid}_{cam.replace('zed_', '')}_{inst['k']:02d}_{rep}"
                d = os.path.join(a.out, "syn", split)
                os.makedirs(d, exist_ok=True)
                real = np.c_[inst["pts"], inst["rgb"]].astype(np.float32)
                np.savez_compressed(os.path.join(d, fid + ".npz"), rgbd_pcd=part, aligned_data=gt,
                                    gt_points=gt[:, :3], gt_colors=np.zeros((0, 3), np.float32),
                                    real_partial=sub(real, MAX_PART, rng),
                                    meta=json.dumps(dict(meta, snapshot=snap, cam=cam, inst=inst["k"],
                                                         score=inst["score"], r_fit=inst["r"])))
                rows.append([split, "syn", fid, cam, inst["k"], inst["score"], inst["area"], inst["n"],
                             round(inst["r"] * 1000, 2), round(inst["rms"] * 1000, 3), round(inst["zmed"], 3), len(part)])
    # real pairs: wrist instance + the same fruit seen by a stacker camera (held-out view)
    if "zed_wrist" in views:
        (Kw, Tw, _, imgw), wi = views["zed_wrist"]
        for cam, ((Ks, Ts, _, _), si) in views.items():
            if cam == "zed_wrist":
                continue
            if not si or not wi:
                continue
            Tws = align_views(Tw, Ts, wi, si)  # stacker cam -> wrist cam
            if Tws is None:
                continue
            sc = np.array([Tws[:3, :3] @ s["c"] + Tws[:3, 3] for s in si])
            for w in wi:
                if not len(sc):
                    break
                d = np.linalg.norm(sc - w["c"], axis=1)
                j = int(np.argmin(d))
                s = si[j]
                if d[j] > a.match_dist or abs(s["r"] - w["r"]) > 0.35 * w["r"]:
                    continue
                held = np.c_[s["pts"] @ Tws[:3, :3].T + Tws[:3, 3], s["rgb"]].astype(np.float32)
                fid = f"{sid}_wrist_{w['k']:02d}_{cam.replace('zed_', '')}_{s['k']:02d}"
                dd = os.path.join(a.out, "real", split)
                os.makedirs(dd, exist_ok=True)
                part = np.c_[w["pts"], w["rgb"]].astype(np.float32)
                ys, xs = np.nonzero(w["mask"])
                y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
                pad = int(0.6 * max(y1 - y0, x1 - x0)) + 10
                crop = imgw[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
                ok, jpg = cv2.imencode(".jpg", crop[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
                np.savez_compressed(os.path.join(dd, fid + ".npz"), rgbd_pcd=sub(part, MAX_PART, rng),
                                    heldout=sub(held, MAX_PART, rng), crop_jpg=np.frombuffer(jpg, np.uint8),
                                    meta=json.dumps(dict(snapshot=snap, wrist_inst=w["k"], cam=cam, inst=s["k"],
                                                         match_mm=float(d[j] * 1000), r_wrist=w["r"], r_other=s["r"],
                                                         bbox=[int(x0), int(y0), int(x1), int(y1)])))
                rows.append([split, "real", fid, cam, w["k"], w["score"], w["area"], w["n"],
                             round(w["r"] * 1000, 2), round(w["rms"] * 1000, 3), round(w["zmed"], 3), len(held)])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("views_root")
    ap.add_argument("out")
    ap.add_argument("-j", type=int, default=24)
    ap.add_argument("--max-snapshots", type=int, default=0)
    ap.add_argument("--val-pct", type=int, default=8)
    ap.add_argument("--syn-per-fruit", type=int, default=1)
    ap.add_argument("--min-score", type=float, default=0.5)
    ap.add_argument("--min-area", type=int, default=1500)
    ap.add_argument("--min-points", type=int, default=400)
    ap.add_argument("--max-depth", type=float, default=1.3)
    ap.add_argument("--min-r", type=float, default=0.012)
    ap.add_argument("--max-r", type=float, default=0.045)
    ap.add_argument("--match-dist", type=float, default=0.012)
    a = ap.parse_args()
    snaps = sorted(d for d in glob.glob(os.path.join(a.views_root, "*")) if os.path.isdir(d))
    if a.max_snapshots:
        snaps = snaps[:a.max_snapshots]
    os.makedirs(a.out, exist_ok=True)
    print(len(snaps), "snapshots", flush=True)
    rows = []
    with ProcessPoolExecutor(a.j) as ex:
        for i, r in enumerate(ex.map(process_snapshot, [(s, a) for s in snaps], chunksize=2)):
            rows += r
            if i % 50 == 0:
                print(i, len(rows), flush=True)
    with open(os.path.join(a.out, "index.csv"), "w") as f:
        f.write("split,kind,id,cam,inst,score,area_px,n_real,r_fit_mm,rms_mm,depth_m,n_part\n")
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")
    print("samples", len(rows))


if __name__ == "__main__":
    main()
