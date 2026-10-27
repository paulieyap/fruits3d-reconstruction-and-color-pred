"""Turn truss-level ripeness grades into fruit-level training labels.

The Harvesting tomato set (COCO: peduncle, stem, truss) grades each truss on the colour scale
unripe / 4 / 5 / 6 / 7, judged on one tomato: the lowest one of the truss. Its fruit mask is not exported,
only its mean Lab colour (`lab_gt`, OpenCV 8-bit Lab). So:

1. A fruit checkpoint finds the tomatoes (2x2 tiles, as in tomato2d/infer_views.py).
2. A fruit belongs to a graded truss when at least half of its mask lies inside the truss mask.
3. The lowest fruit of the truss (largest centroid y) gets the truss grade; every other fruit is 'tomato'
   (ripeness unknown, see tomato2d/model.py).
4. As a check, the lowest fruit's median Lab is compared with `lab_gt`. With --max-lab-dist the grade is only
   kept when that distance is small, so a missed or wrongly picked lowest fruit does not get the grade.

  python3 -m tomato2d.prepare_grades COCO_ROOT CKPT OUT_DIR [--max-lab-dist 25]

COCO_ROOT holds annotations/instances_{train,val}2017.json and {train,val}2017/. Output is in the
tomato2d/prepare.py format with classes ['unripe', '4', '5', '6', '7'], one image per tile; train2017 -> train,
val2017 -> val and test (alternate images, so both get every source camera).
"""
import json
import os

import click
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

from tomato2d.infer_views import predict_tiled
from tomato2d.model import load
from tomato2d.prepare import SOLIDITY_MIN, width_keypoints
from tomato2d.prepare_views import UNKNOWN_LABEL, tiles
from tomato2d.score_fruit import closed_solidity

GRADE_CLASSES = ['unripe', '4', '5', '6', '7']
TRUSS = 'truss'


def polygon_mask(seg, H, W):
    m = np.zeros((H, W), np.uint8)
    for poly in seg:
        cv2.fillPoly(m, [np.asarray(poly, np.float32).reshape(-1, 2).round().astype(np.int32)], 1)
    return m.astype(bool)


def assign(fruits, trusses, lab_img, inside_frac=0.5):
    """fruits: list of bool masks; trusses: list of (mask, grade, lab_gt). Returns {fruit index: (grade, lab_dist,
    n_fruits_in_truss, lab_rank)} for the lowest fruit of each graded truss."""
    out = {}
    for tm, grade, lab_gt in trusses:
        members = [i for i, f in enumerate(fruits) if (f & tm).sum() >= inside_frac * f.sum()]
        if not members:
            continue
        cy = [np.where(fruits[i])[0].mean() for i in members]
        low = members[int(np.argmax(cy))]
        d = {i: float(np.linalg.norm(np.median(lab_img[fruits[i]], 0) - np.asarray(lab_gt))) for i in members}
        rank = sorted(d, key=d.get).index(low)
        if low not in out or d[low] < out[low][1]:
            out[low] = (grade, d[low], len(members), rank)
    return out


@click.command()
@click.argument('root')
@click.argument('ckpt')
@click.argument('out')
@click.option('--score-thr', default=0.5)
@click.option('--max-lab-dist', default=0.0, help='drop a grade when the lowest fruit is further than this from lab_gt (0 = keep all)')
@click.option('--min-area', default=64)
def main(root, ckpt, out, score_thr, max_lab_dist, min_area):
    dev = 'cuda'
    os.makedirs(os.path.join(out, 'images'), exist_ok=True)
    os.makedirs(os.path.join(out, 'masks'), exist_ok=True)
    model, _ = load(ckpt, dev)
    images, checks = [], []
    for coco_split in ('train', 'val'):
        coco = json.load(open(os.path.join(root, 'annotations', f'instances_{coco_split}2017.json')))
        truss_id = next(c['id'] for c in coco['categories'] if c['name'] == TRUSS)
        anns = {}
        for a in coco['annotations']:
            if a['category_id'] == truss_id and a.get('ripeness_grade') in GRADE_CLASSES and isinstance(a['segmentation'], list):
                anns.setdefault(a['image_id'], []).append(a)
        for n, im in enumerate(coco['images']):
            split = 'train' if coco_split == 'train' else ('val' if n % 2 == 0 else 'test')
            bgr = cv2.imread(os.path.join(root, f'{coco_split}2017', im['file_name']))
            if bgr is None:
                continue
            H, W = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            lab_img = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            m, s, _ = predict_tiled(model, TF.to_tensor(rgb), dev, score_thr)
            fruits = [x for x in m.cpu().numpy() if x.sum() >= min_area]
            trusses = [(polygon_mask(a['segmentation'], H, W), a['ripeness_grade'], a['lab_gt']) for a in anns.get(im['id'], [])]
            graded = assign(fruits, trusses, lab_img)
            for i, (g, dist, k, rank) in graded.items():
                checks.append({'file': im['file_name'], 'split': split, 'grade': g, 'lab_dist': dist, 'fruits_in_truss': k,
                               'lab_rank_of_lowest': rank})
            labels = [UNKNOWN_LABEL] * len(fruits)
            for i, (g, dist, _, _) in graded.items():
                if not max_lab_dist or dist <= max_lab_dist:
                    labels[i] = g
            name0 = os.path.splitext(im['file_name'])[0]
            for ti, (y0, x0, th, tw) in enumerate(tiles(H, W)):
                masks, inst = [], []
                for f, lab in zip(fruits, labels):
                    sub = f[y0:y0 + th, x0:x0 + tw]
                    if sub.sum() < min_area:
                        continue
                    if sub.sum() < 0.5 * f.sum():  # a graded fruit keeps its grade only in the tile holding most of it
                        lab = UNKNOWN_LABEL
                    ys, xs = np.where(sub)
                    kp, wpx = width_keypoints(sub)
                    sol = closed_solidity(sub)
                    border = xs.min() == 0 or ys.min() == 0 or xs.max() == tw - 1 or ys.max() == th - 1
                    masks.append(sub)
                    inst.append({'label': lab, 'bbox': [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
                                 'width_kp': kp, 'width_px': wpx, 'solidity': sol, 'area_px': int(sub.sum()),
                                 'width_valid': bool(kp is not None and sol >= SOLIDITY_MIN and not border and wpx >= 8)})
                if not inst:
                    continue
                name = f'{name0}__t{ti}'
                cv2.imwrite(os.path.join(out, 'images', name + '.jpg'), bgr[y0:y0 + th, x0:x0 + tw], [cv2.IMWRITE_JPEG_QUALITY, 92])
                np.savez_compressed(os.path.join(out, 'masks', name + '.npz'), masks=np.stack(masks))
                images.append({'file': name, 'width': tw, 'height': th, 'split': split, 'source': im['file_name'],
                               'tile': [y0, x0], 'instances': inst})
            print(f'[{coco_split} {n + 1}/{len(coco["images"])}] {im["file_name"]}: {len(fruits)} fruits, '
                  f'{len(trusses)} graded trusses, {len(graded)} graded fruits', flush=True)

    json.dump({'classes': GRADE_CLASSES, 'images': images, 'source': 'truss grades on the lowest fruit',
               'max_lab_dist': max_lab_dist, 'solidity_min': SOLIDITY_MIN},
              open(os.path.join(out, 'annotations.json'), 'w'))
    json.dump(checks, open(os.path.join(out, 'grade_checks.json'), 'w'))
    d = np.array([c['lab_dist'] for c in checks])
    print(f'tiles {len(images)}, graded fruits {len(checks)}, lab dist median {np.median(d):.1f} p90 {np.percentile(d, 90):.1f}, '
          f'lowest is the closest-colour fruit in {np.mean([c["lab_rank_of_lowest"] == 0 for c in checks]):.0%}')


if __name__ == '__main__':
    main()
