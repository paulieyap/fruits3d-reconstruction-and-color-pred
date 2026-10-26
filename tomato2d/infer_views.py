"""Run a checkpoint on RGB-D views from another camera and measure fruit widths in mm.

Each view directory holds color.jpg, depth.npy (metres, NaN = invalid), intrinsic.json ({"K": 3x3}) and optionally
inst.png + inst.json (reference instance masks, uint16 ids, scores[k - 1] for id k) that predictions are compared to.

Two passes per view: the whole image, and overlapping 2x2 tiles (fruit twice as many pixels, closer to the
TomatoMAP close-ups the model was trained on), merged by mask NMS.

Width in mm = widest horizontal chord (px) x median depth inside the eroded mask / fx. Only fruits whose mask is
compact (closed solidity >= 0.92) and clear of the image border get a width, as in training.

  python3 -m tomato2d.infer_views CKPT VIEW_LIST OUT_DIR [--renders 16]

VIEW_LIST is a text file with one view directory per line.
"""
import json
import os

import click
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

from tomato2d.metrics import mask_iou, widest_chord
from tomato2d.model import load
from tomato2d.score_fruit import closed_solidity
from tomato2d.test import paint


@torch.no_grad()
def predict(model, img, dev, score_thr):
    o = model([img.to(dev)])[0]
    k = o['scores'] >= score_thr
    return (o['masks'][k, 0] > 0.5), o['scores'][k], o['labels'][k]


@torch.no_grad()
def predict_tiled(model, img, dev, score_thr, overlap=0.1, nms_iou=0.3):
    _, H, W = img.shape
    th, tw = int(H / 2 * (1 + overlap)), int(W / 2 * (1 + overlap))
    masks, scores, labels = [], [], []
    for y0 in (0, H - th):
        for x0 in (0, W - tw):
            m, s, l = predict(model, img[:, y0:y0 + th, x0:x0 + tw], dev, score_thr)
            full = torch.zeros(len(m), H, W, dtype=torch.bool, device=dev)
            full[:, y0:y0 + th, x0:x0 + tw] = m
            masks.append(full); scores.append(s); labels.append(l)
    m, s, l = torch.cat(masks), torch.cat(scores), torch.cat(labels)
    if not len(m):
        return m, s, l
    order = torch.argsort(s, descending=True)
    m, s, l = m[order], s[order], l[order]
    area = m.flatten(1).sum(1).float()
    inter = m.flatten(1).float() @ m.flatten(1).float().T
    iou = inter / (area[:, None] + area[None] - inter).clamp(min=1)
    contain = inter / area[None].clamp(min=1)  # [i, j]: share of j inside i
    keep = []
    for j in range(len(m)):
        if all(iou[i, j] < nms_iou and contain[i, j] < 0.7 for i in keep):
            keep.append(j)
    return m[keep], s[keep], l[keep]


def width_mm(mask, depth, fx):
    """mask (H, W) bool numpy -> (width_mm, width_px, z_m, chord) or None when the width is not trustworthy."""
    H, W = mask.shape
    ys, xs = np.where(mask)
    if len(xs) < 50 or xs.min() == 0 or ys.min() == 0 or xs.max() == W - 1 or ys.max() == H - 1:
        return None
    if closed_solidity(mask) < 0.92:
        return None
    k = max(3, int(0.15 * np.sqrt(mask.sum())))
    inner = cv2.erode(mask.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    z = depth[inner & np.isfinite(depth) & (depth > 0.1)]
    if len(z) < 20:
        return None
    zm = float(np.median(z))
    wpx, chord = widest_chord(torch.from_numpy(mask))
    return wpx * zm / fx * 1000, wpx, zm, chord


def reference(view, ref_thr):
    p = os.path.join(view, 'inst.png')
    if not os.path.exists(p):
        return None
    lab = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    sc = json.load(open(os.path.join(view, 'inst.json'))).get('scores', [])
    return [lab == k + 1 for k, s in enumerate(sc) if s >= ref_thr and (lab == k + 1).any()]


@click.command()
@click.argument('ckpt')
@click.argument('view_list')
@click.argument('out')
@click.option('--score-thr', default=0.5)
@click.option('--ref-thr', default=0.5, help='min score of a reference mask')
@click.option('--near-px', default=60, help='reference fruits at least this wide (px) count as near')
@click.option('--renders', default=16)
def main(ckpt, view_list, out, score_thr, ref_thr, near_px, renders):
    dev = 'cuda'
    os.makedirs(os.path.join(out, 'renders'), exist_ok=True)
    model, ck = load(ckpt, dev)
    model.eval()
    classes = ['nascent', 'mini', 'unripe green tomato', 'semi ripe', 'fully ripe']
    views = [l.strip() for l in open(view_list) if l.strip()]
    pick = set(np.linspace(0, len(views) - 1, min(renders, len(views))).round().astype(int).tolist())
    rows, fruits = [], []
    for vi, view in enumerate(views):
        name = view.rstrip('/').replace('/', '__')
        rgb = np.array(Image.open(os.path.join(view, 'color.jpg')).convert('RGB'))
        depth = np.load(os.path.join(view, 'depth.npy')).astype(np.float32)
        fx = json.load(open(os.path.join(view, 'intrinsic.json')))['K'][0][0]
        img = TF.to_tensor(rgb)
        ref = reference(view, ref_thr)
        row = {'view': view}
        for mode, fn in (('full', predict), ('tiled', predict_tiled)):
            m, s, l = fn(model, img, dev, score_thr)
            row[f'{mode}_n'] = len(m)
            if ref is not None:
                g = torch.from_numpy(np.stack(ref)).to(dev) if ref else torch.zeros(0, *rgb.shape[:2], dtype=torch.bool, device=dev)
                near = [i for i, r in enumerate(ref) if widest_chord(torch.from_numpy(r))[0] >= near_px]
                iou = mask_iou(m, g) if len(m) and len(g) else torch.zeros(len(m), len(g), device=dev)
                best_g = iou.max(0).values if len(m) and len(g) else torch.zeros(len(g), device=dev)
                best_p = iou.max(1).values if len(m) and len(g) else torch.zeros(len(m), device=dev)
                row.update({f'{mode}_ref_n': len(g), f'{mode}_ref_near_n': len(near),
                            f'{mode}_ref_found': int((best_g >= 0.5).sum()),
                            f'{mode}_ref_near_found': int((best_g[near] >= 0.5).sum()) if near else 0,
                            f'{mode}_pred_matched': int((best_p >= 0.5).sum())})
            if mode == 'tiled':
                mn = m.cpu().numpy()
                lines, labs = [], []
                for i in range(len(mn)):
                    w = width_mm(mn[i], depth, fx)
                    f = {'view': view, 'score': float(s[i]), 'label': classes[int(l[i]) - 1], 'area_px': int(mn[i].sum())}
                    if w:
                        f.update(width_mm=w[0], width_px=w[1], z_m=w[2])
                        lines.append(w[3]); labs.append((w[3], f'{w[0]:.0f}'))
                    fruits.append(f)
                if vi in pick:
                    left = paint(rgb, ref or [], [5] * len(ref or []), [])
                    right = paint(rgb, list(mn), [int(x) for x in l.cpu()], lines)
                    d = ImageDraw.Draw(right)
                    for (p0, p1), t in labs:
                        x, y = (p0[0] + p1[0]) / 2, p0[1] - 22
                        d.rectangle([x - 22, y - 2, x + 22, y + 18], fill=(255, 255, 255))
                        d.text((x - 18, y), t + 'mm', fill=(0, 0, 0))
                    both = Image.new('RGB', (2 * rgb.shape[1] + 12, rgb.shape[0]), (255, 255, 255))
                    both.paste(left, (0, 0)); both.paste(right, (rgb.shape[1] + 12, 0))
                    both = both.resize((1600, round(both.size[1] * 1600 / both.size[0])), Image.LANCZOS)
                    both.save(os.path.join(out, 'renders', name + '.jpg'), quality=82)
                    row['render'] = name + '.jpg'
        rows.append(row)
        print(vi, json.dumps({k: v for k, v in row.items() if k != 'view'}), flush=True)

    tot = lambda k: sum(r.get(k, 0) for r in rows)
    summary = {'checkpoint': ckpt, 'epoch': ck['epoch'], 'views': len(rows), 'score_thr': score_thr, 'ref_thr': ref_thr,
               'near_px': near_px}
    for mode in ('full', 'tiled'):
        summary[mode] = {'pred': tot(f'{mode}_n'), 'ref': tot(f'{mode}_ref_n'), 'ref_found': tot(f'{mode}_ref_found'),
                         'ref_near': tot(f'{mode}_ref_near_n'), 'ref_near_found': tot(f'{mode}_ref_near_found'),
                         'pred_matched': tot(f'{mode}_pred_matched')}
    w = np.array([f['width_mm'] for f in fruits if 'width_mm' in f])
    summary['widths'] = {'n': len(w), **({'median_mm': float(np.median(w)), 'p10_mm': float(np.percentile(w, 10)),
                                          'p90_mm': float(np.percentile(w, 90))} if len(w) else {})}
    summary['labels'] = {c: sum(f['label'] == c for f in fruits) for c in classes}
    json.dump({'summary': summary, 'views': rows, 'fruits': fruits}, open(os.path.join(out, 'views_metrics.json'), 'w'), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
