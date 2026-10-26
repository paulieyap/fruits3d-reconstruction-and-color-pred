"""Prepare TomatoMAP-Seg (macro photographs with instance masks) for tomato instance segmentation and
width-keypoint training.

Reads the FiftyOne export (samples.json + data/) directly, without FiftyOne:

  python3 -m tomato2d.prepare TOMATOMAP_ROOT OUT_DIR [--long-side 1333] [--whole-fruit]

Writes OUT_DIR/images/<name>.jpg (downscaled), OUT_DIR/masks/<name>.npz (bool stack, one mask per fruit)
and OUT_DIR/annotations.json:

  {'classes': [...], 'images': [{'file', 'width', 'height', 'split', 'session', 'capture',
                                 'instances': [{'label', 'bbox' (x0, y0, x1, y1 px), 'width_kp' [[x, y], [x, y]],
                                                'width_px', 'width_valid', 'solidity', 'area_px', 'group', 'pieces'}]}]}

Width keypoints: the two ends of the widest horizontal chord of the mask (the camera is held upright
and fruits hang, so image-horizontal approximates the fruit's equatorial width). They are only trusted
(`width_valid`) when the mask is compact (solidity >= 0.92) and does not touch the image border;
occluded or cut-off fruits keep their mask but are excluded from the keypoint loss.

TomatoMAP labels a fruit crossed by a stem or leaf as several pieces that share an `isat_group` id.
By default each piece is one instance; `--whole-fruit` merges the pieces of a group into one instance
(label of the largest area; solidity measured after closing the gaps the occluder leaves).

Splits are made per capture session (photos less than 120 s apart), so near-identical shots of one fruit
cluster never end up in different splits.
"""
import base64
import io
import json
import os
import zlib
from datetime import datetime

import click
import cv2
import numpy as np

FRUIT_CLASSES = ['nascent', 'mini', 'unripe green tomato', 'semi ripe', 'fully ripe']
SOLIDITY_MIN = 0.92


def decode_mask(det):
    """FiftyOne stores masks as zlib-compressed .npy, cropped to the bounding box."""
    raw = det['mask']['$binary']['base64'] if isinstance(det['mask'], dict) else det['mask']
    return np.load(io.BytesIO(zlib.decompress(base64.b64decode(raw)))).astype(bool)


def full_mask(det, W, H):
    x, y, w, h = det['bounding_box']  # relative, top-left + size
    crop = decode_mask(det)
    x0, y0 = int(round(x * W)), int(round(y * H))
    m = np.zeros((H, W), bool)
    ch, cw = crop.shape
    m[y0:y0 + ch, x0:x0 + cw] = crop[:max(0, min(ch, H - y0)), :max(0, min(cw, W - x0))]
    return m


def width_keypoints(mask):
    rows = np.where(mask.any(1))[0]
    best = (-1, None)
    for r in rows:
        cols = np.where(mask[r])[0]
        w = cols[-1] - cols[0]
        if w > best[0]:
            best = (w, (r, cols[0], cols[-1]))
    if best[1] is None:
        return None, 0.0
    r, c0, c1 = best[1]
    return [[float(c0), float(r)], [float(c1), float(r)]], float(c1 - c0)


def solidity(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return 0.0
    c = max(cs, key=cv2.contourArea)
    hull = cv2.contourArea(cv2.convexHull(c))
    return float(cv2.contourArea(c) / hull) if hull > 0 else 0.0


@click.command()
@click.argument('root')
@click.argument('out')
@click.option('--long-side', default=1333, help='downscale images to this long side (px)')
@click.option('--seed', default=0)
@click.option('--split-frac', default=(0.7, 0.15, 0.15), nargs=3, type=float)
@click.option('--whole-fruit', is_flag=True, help='merge the labelled pieces of each fruit (same isat_group)')
def main(root, out, long_side, seed, split_frac, whole_fruit):
    os.makedirs(os.path.join(out, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out, 'masks'), exist_ok=True)
    samples = [s for s in json.load(open(os.path.join(root, 'samples.json')))['samples']
               if 'seg' in s['tags'] and 'labeled' in s['tags']]

    # capture sessions: consecutive photos less than 120 s apart
    def when(s):
        return datetime.fromisoformat(s['capture_datetime']['$date'].replace('Z', '+00:00')).timestamp()
    samples.sort(key=when)
    session, last, sessions = -1, None, []
    for s in samples:
        t = when(s)
        if last is None or t - last > 120:
            session += 1
        sessions.append(session)
        last = t
    rng = np.random.default_rng(seed)
    order = rng.permutation(session + 1)
    n_tr, n_va = int(split_frac[0] * len(order)), int(split_frac[1] * len(order))
    split_of = {int(g): ('train' if i < n_tr else 'val' if i < n_tr + n_va else 'test') for i, g in enumerate(order)}

    images, n_inst, n_valid = [], 0, 0
    for s, g in zip(samples, sessions):
        dets = [d for d in s['ground_truth']['detections'] if d['label'] in FRUIT_CLASSES and d.get('mask')]
        if not dets:
            continue
        img = cv2.imread(os.path.join(root, s['filepath']))
        H0, W0 = img.shape[:2]
        k = long_side / max(H0, W0)
        W, H = int(round(W0 * k)), int(round(H0 * k))
        name = os.path.splitext(os.path.basename(s['filepath']))[0]
        cv2.imwrite(os.path.join(out, 'images', name + '.jpg'), cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        pieces = []
        for d in dets:
            m = cv2.resize(full_mask(d, W0, H0).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            if m.sum() >= 16:
                pieces.append((d['label'], d.get('isat_group'), m))
        if whole_fruit:
            groups = {}
            for i, (lab, grp, m) in enumerate(pieces):
                groups.setdefault(('g', grp) if grp is not None else ('p', i), []).append((lab, grp, m))
            fruits = []
            for (_, _), ps in groups.items():
                m = np.logical_or.reduce([p[2] for p in ps])
                area = {}
                for lab, _, pm in ps:
                    area[lab] = area.get(lab, 0) + int(pm.sum())
                fruits.append((max(area, key=area.get), ps[0][1], m, len(ps)))
        else:
            fruits = [(lab, grp, m, 1) for lab, grp, m in pieces]

        masks, inst = [], []
        for label, grp, m, n_pieces in fruits:
            ys, xs = np.where(m)
            kp, wpx = width_keypoints(m)
            if n_pieces > 1:  # close the gaps a stem or leaf leaves between the pieces
                k = max(5, int(round(0.25 * np.sqrt(m.sum()))))
                sol = solidity(cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))))
            else:
                sol = solidity(m)
            border = xs.min() == 0 or ys.min() == 0 or xs.max() == W - 1 or ys.max() == H - 1
            valid = bool(kp is not None and sol >= SOLIDITY_MIN and not border and wpx >= 8)
            masks.append(m)
            inst.append({'label': label, 'bbox': [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
                         'width_kp': kp, 'width_px': wpx, 'width_valid': valid, 'solidity': sol, 'area_px': int(m.sum()),
                         'group': grp, 'pieces': n_pieces})
            n_valid += valid
        if not inst:
            continue
        np.savez_compressed(os.path.join(out, 'masks', name + '.npz'), masks=np.stack(masks))
        images.append({'file': name, 'width': W, 'height': H, 'split': split_of[g], 'session': int(g),
                       'capture': s['capture_datetime']['$date'], 'instances': inst})
        n_inst += len(inst)
        print(f'[{len(images)}] {name}: {len(inst)} fruits', flush=True)

    json.dump({'classes': FRUIT_CLASSES, 'images': images, 'long_side': long_side, 'solidity_min': SOLIDITY_MIN,
               'whole_fruit': whole_fruit},
              open(os.path.join(out, 'annotations.json'), 'w'))
    per = {sp: sum(1 for im in images if im['split'] == sp) for sp in ('train', 'val', 'test')}
    print(f'images {len(images)} {per}, fruit instances {n_inst}, width-valid {n_valid}, sessions {session + 1}')


if __name__ == '__main__':
    main()
