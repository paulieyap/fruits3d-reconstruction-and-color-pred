"""Extract harvester execution snapshots (mcap) into per-view RGB-D frames.

For every snapshot folder and every camera with colour + depth, writes
<out>/<snapshot>/<cam>/{color.jpg, depth.npy (float32 m, NaN = invalid),
conf.png, intrinsic.json, pose.txt (4x4 T_world_cam, optical frame)}.
"""
import argparse, glob, json, os, struct, sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numba
import numpy as np
from mcap_ros2.reader import read_ros2_messages
from scipy.spatial.transform import Rotation


@numba.njit(cache=True)
def rvl_decode(nibs, n):
    out = np.zeros(n, np.int64)
    p = 0
    i = 0
    prev = 0
    while i < n:
        v = 0
        sh = 0
        while True:
            x = nibs[p]
            p += 1
            v |= (x & 7) << sh
            sh += 3
            if (x & 8) == 0:
                break
        i += v
        if i >= n:
            break
        v = 0
        sh = 0
        while True:
            x = nibs[p]
            p += 1
            v |= (x & 7) << sh
            sh += 3
            if (x & 8) == 0:
                break
        for _ in range(v):
            if i >= n:
                break
            u = 0
            sh = 0
            while True:
                x = nibs[p]
                p += 1
                u |= (x & 7) << sh
                sh += 3
                if (x & 8) == 0:
                    break
            delta = (u >> 1) ^ -(u & 1)
            prev = prev + delta
            out[i] = prev
            i += 1
    return out


def rvl_nibbles(b):
    """RVL packs nibbles MSB-first into little-endian uint32 words."""
    b = b[: len(b) // 4 * 4]
    w = np.frombuffer(b, "<u4")
    sh = np.arange(28, -1, -4, dtype=np.uint32)
    return ((w[:, None] >> sh[None, :]) & 0xF).astype(np.int64).ravel()


def decode_compressed_depth(msg):
    data = bytes(msg.data)
    fmt = msg.format
    # ConfigHeader: int32 format, float depthQuantA, float depthQuantB
    _, qa, qb = struct.unpack("<iff", data[:12])
    body = data[12:]
    if "rvl" in fmt:
        cols, rows = struct.unpack("<II", body[:8])
        nibs = rvl_nibbles(body[8:] + b"\0" * 8)
        inv = rvl_decode(nibs, cols * rows).reshape(rows, cols)
    else:
        inv = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_UNCHANGED)
    if "32FC1" in fmt:
        inv = inv.astype(np.float32)
        depth = np.full(inv.shape, np.nan, np.float32)
        ok = inv > 0
        depth[ok] = qa / (inv[ok] + qb)  # encoder: inv = qa / depth - qb
        return depth
    d = inv.astype(np.float32) / 1000.0
    d[inv == 0] = np.nan
    return d


def tf_mat(tr):
    t = tr.transform.translation
    q = tr.transform.rotation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def chain(edges, target, root="world"):
    """T_root_target from parent->child edges (dict child -> (parent, T_parent_child))."""
    T = np.eye(4)
    f = target
    seen = 0
    while f != root:
        if f not in edges or seen > 50:
            return None
        p, Tp = edges[f]
        T = Tp @ T
        f = p
        seen += 1
    return T


def extract(snap_dir, out_root):
    mc = glob.glob(os.path.join(snap_dir, "*.mcap"))
    if not mc:
        return snap_dir, 0
    name = os.path.basename(snap_dir.rstrip("/"))
    edges = {}
    imgs, deps, confs, infos = {}, {}, {}, {}
    joints = None
    for m in read_ros2_messages(mc[0]):
        t = m.channel.topic
        msg = m.ros_msg
        if t in ("/tf", "/tf_static"):
            for tr in msg.transforms:
                edges[tr.child_frame_id] = (tr.header.frame_id, tf_mat(tr))
            continue
        if t == "/joint_states":
            joints = dict(zip(msg.name, [float(x) for x in msg.position]))
            continue
        if not t.startswith("/zed_"):
            continue
        cam = t.split("/")[1]
        if t.endswith("rgb/image_rect_color/compressed"):
            imgs[cam] = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
        elif t.endswith("depth_registered/compressedDepth"):
            deps[cam] = decode_compressed_depth(msg)
        elif "confidence" in t and t.endswith("compressed"):
            confs[cam] = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_UNCHANGED)
        elif t.endswith("rgb/camera_info"):
            infos[cam] = (np.array(msg.k).reshape(3, 3), msg.width, msg.height, msg.header.frame_id)
    n = 0
    for cam in imgs:
        if cam not in deps or cam not in infos:
            continue
        K, w, h, frame = infos[cam]
        T = chain(edges, frame)
        if T is None:
            continue
        d = os.path.join(out_root, name, cam)
        os.makedirs(d, exist_ok=True)
        img = imgs[cam]
        dep = deps[cam]
        if dep.shape[:2] != img.shape[:2]:
            sx = img.shape[1] / dep.shape[1]
            K = K.copy()
        cv2.imwrite(os.path.join(d, "color.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        np.save(os.path.join(d, "depth.npy"), dep.astype(np.float16))
        if cam in confs:
            cv2.imwrite(os.path.join(d, "conf.png"), confs[cam])
        json.dump({"K": K.tolist(), "width": w, "height": h, "frame": frame,
                   "depth_hw": list(dep.shape[:2]), "joints": joints},
                  open(os.path.join(d, "intrinsic.json"), "w"))
        np.savetxt(os.path.join(d, "pose.txt"), T)
        n += 1
    return snap_dir, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snap_glob")
    ap.add_argument("out")
    ap.add_argument("-j", type=int, default=16)
    a = ap.parse_args()
    snaps = sorted(glob.glob(a.snap_glob))
    print(len(snaps), "snapshots", flush=True)
    tot = 0
    with ProcessPoolExecutor(a.j) as ex:
        for i, (s, n) in enumerate(ex.map(extract, snaps, [a.out] * len(snaps))):
            tot += n
            if i % 50 == 0:
                print(i, s, n, flush=True)
    print("views", tot)


if __name__ == "__main__":
    main()
