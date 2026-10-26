"""Turn RGB-D views with reference instance masks (e.g. SAM 3 "tomato" masks) into a training set in the
tomato2d/prepare.py format, as pseudo-labels for fine-tuning on our own cameras.

Each view directory holds color.jpg and inst.png + inst.json (uint16 ids, scores[k - 1] for id k). Masks with
score >= --min-score become instances labelled 'tomato': ripeness unknown, so they train detection, masks and
width keypoints but not the ripeness classifier (see tomato2d/model.py).

Each view is cut into the same overlapping 2x2 tiles that tomato2d/infer_views.py predicts on. Splits are by day
(snapshot date), so near-identical snapshots of one plant never end up in different splits.

  python3 -m tomato2d.prepare_views VIEWS_ROOT OUT_DIR --val-days 20260908,20261007 --test-days 20260825,20260917,20261015

VIEWS_ROOT holds <snapshot>/<camera>/ directories; snapshot names contain the date as the third '_' field
(execution_snapshot_YYYYMMDD_HHMMSS_mmm).
"""
import json
import os

import click
import cv2
import numpy as np

from tomato2d.prepare import FRUIT_CLASSES, SOLIDITY_MIN, width_keypoints
from tomato2d.score_fruit import closed_solidity

UNKNOWN_LABEL = 'tomato'


def tiles(H, W, overlap=0.1):
    """(y0, x0, th, tw) of the 2x2 tiles, as in infer_views.predict_tiled."""
    th, tw = int(H / 2 * (1 + overlap)), int(W / 2 * (1 + overlap))
    return [(y0, x0, th, tw) for y0 in (0, H - th) for x0 in (0, W - tw)]


def pick_views(root, stride, max_per, min_fruits, min_score):
    """Every `stride`-th view per (day, camera), at most `max_per`, with at least `min_fruits` reference masks."""
    groups = {}
    for snap in sorted(os.listdir(root)):
        if not os.path.isdir(os.path.join(root, snap)) or len(snap.split('_')) < 3:
            continue
        for cam in sorted(os.listdir(os.path.join(root, snap))):
            view = os.path.join(root, snap, cam)
            if os.path.exists(os.path.join(view, 'inst.json')):
                groups.setdefault((snap.split('_')[2], cam), []).append(view)
    out = []
    for (day, cam), views in sorted(groups.items()):
        kept = []
        for v in views[::stride]:
            sc = json.load(open(os.path.join(v, 'inst.json'))).get('scores', [])
            if sum(s >= min_score for s in sc) >= min_fruits:
                kept.append(v)
            if len(kept) >= max_per:
                break
        out += [(day, cam, v) for v in kept]
    return out


@click.command()
@click.argument('root')
@click.argument('out')
@click.option('--val-days', required=True, help='comma-separated YYYYMMDD')
@click.option('--test-days', required=True, help='comma-separated YYYYMMDD')
@click.option('--min-score', default=0.5, help='min reference mask score to become an instance')
@click.option('--min-area', default=64, help='min instance area (px) inside a tile')
@click.option('--stride', default=4, help='take every n-th view per day and camera')
@click.option('--max-per', default=40, help='max views per day and camera')
@click.option('--min-fruits', default=3, help='skip views with fewer reference masks')
def main(root, out, val_days, test_days, min_score, min_area, stride, max_per, min_fruits):
    os.makedirs(os.path.join(out, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out, 'masks'), exist_ok=True)
    split_of = {**{d: 'val' for d in val_days.split(',')}, **{d: 'test' for d in test_days.split(',')}}
    images, n_inst, n_valid = [], 0, 0
    for day, cam, view in pick_views(root, stride, max_per, min_fruits, min_score):
        img = cv2.imread(os.path.join(view, 'color.jpg'))
        lab = cv2.imread(os.path.join(view, 'inst.png'), cv2.IMREAD_UNCHANGED)
        sc = json.load(open(os.path.join(view, 'inst.json'))).get('scores', [])
        ids = [k + 1 for k, s in enumerate(sc) if s >= min_score]
        snap = os.path.basename(os.path.dirname(view))
        for ti, (y0, x0, th, tw) in enumerate(tiles(*img.shape[:2])):
            sub = lab[y0:y0 + th, x0:x0 + tw]
            masks, inst = [], []
            for k in ids:
                m = sub == k
                if m.sum() < min_area:
                    continue
                ys, xs = np.where(m)
                kp, wpx = width_keypoints(m)
                sol = closed_solidity(m)
                border = xs.min() == 0 or ys.min() == 0 or xs.max() == tw - 1 or ys.max() == th - 1
                valid = bool(kp is not None and sol >= SOLIDITY_MIN and not border and wpx >= 8)
                masks.append(m)
                inst.append({'label': UNKNOWN_LABEL, 'bbox': [float(xs.min()), float(ys.min()), float(xs.max() + 1),
                                                              float(ys.max() + 1)],
                             'width_kp': kp, 'width_px': wpx, 'width_valid': valid, 'solidity': sol,
                             'area_px': int(m.sum()), 'ref_id': k, 'ref_score': float(sc[k - 1])})
                n_valid += valid
            if not inst:
                continue
            name = f'{snap}__{cam}__t{ti}'
            cv2.imwrite(os.path.join(out, 'images', name + '.jpg'), img[y0:y0 + th, x0:x0 + tw],
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            np.savez_compressed(os.path.join(out, 'masks', name + '.npz'), masks=np.stack(masks))
            images.append({'file': name, 'width': tw, 'height': th, 'split': split_of.get(day, 'train'), 'day': day,
                           'camera': cam, 'view': view, 'tile': [y0, x0], 'instances': inst})
            n_inst += len(inst)
        print(f'[{len(images)}] {snap}/{cam}', flush=True)

    json.dump({'classes': FRUIT_CLASSES, 'images': images, 'source': 'reference masks', 'min_score': min_score,
               'solidity_min': SOLIDITY_MIN}, open(os.path.join(out, 'annotations.json'), 'w'))
    per = {sp: sum(1 for im in images if im['split'] == sp) for sp in ('train', 'val', 'test')}
    print(f'tiles {len(images)} {per}, instances {n_inst}, width-valid {n_valid}')


if __name__ == '__main__':
    main()
