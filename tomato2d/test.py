"""Score a trained checkpoint on a split and render predictions next to ground truth.

  python3 -m tomato2d.test DATA_DIR CKPT OUT_DIR [--split test] [--renders 12]

Writes OUT_DIR/<split>_metrics.json (headline metrics, ripeness confusion matrix, per-session metrics,
per-fruit width rows) and OUT_DIR/renders/<split>_<file>.jpg (left: ground truth, right: prediction).
"""
import json
import os

import click
import numpy as np
import torch
from PIL import Image, ImageDraw

from tomato2d.data import TomatoSeg
from tomato2d.metrics import evaluate, mask_iou, widest_chord
from tomato2d.model import load

COLORS = [(46, 160, 67), (163, 190, 50), (240, 180, 40), (240, 110, 40), (210, 40, 40)]  # by ripeness class index


def outline(mask):
    m = mask
    inner = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
    edge = m.copy()
    edge[1:-1, 1:-1] &= ~inner
    return edge


def paint(img, masks, labels, lines, alpha=0.35):
    """img (H, W, 3) uint8; masks list of bool (H, W); labels 1-based; lines list of ((x0, y0), (x1, y1)) or None."""
    out = img.astype(np.float32)
    for m, l in zip(masks, labels):
        c = np.array(COLORS[l - 1], np.float32)
        out[m] = out[m] * (1 - alpha) + c * alpha
        out[outline(m)] = c
    pil = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(pil)
    for ln in lines:
        if ln is not None:
            d.line([ln[0], ln[1]], fill=(255, 255, 255), width=3)
            d.line([ln[0], ln[1]], fill=(20, 20, 20), width=1)
    return pil


@torch.no_grad()
def confusion_and_renders(model, ds, device, out, split, n_renders, score_thr=0.5, iou_thr=0.5):
    K = len(ds.classes)
    conf = np.zeros((K, K), int)  # rows: true, cols: predicted (matched fruits only)
    missed = np.zeros(K, int)
    per_image = []
    os.makedirs(os.path.join(out, 'renders'), exist_ok=True)
    pick = set(np.linspace(0, len(ds) - 1, min(n_renders, len(ds))).round().astype(int).tolist())
    for i in range(len(ds)):
        img, t = ds[i]
        o = model([img.to(device)])[0]
        keep = o['scores'] >= score_thr
        pm = (o['masks'][keep, 0] > 0.5)
        pl = o['labels'][keep].cpu().numpy()
        pk = o['keypoints'][keep, :, :2].cpu().numpy()
        gt = t['masks'].to(device).bool()
        gl = t['labels'].numpy()
        iou = mask_iou(pm, gt) if len(pm) and len(gt) else torch.zeros(len(pm), len(gt), device=device)
        used_p, used_g = set(), set()
        if iou.numel():
            for flat in torch.argsort(iou.flatten(), descending=True).tolist():
                p, g = divmod(flat, len(gt))
                if iou[p, g] < iou_thr:
                    break
                if p in used_p or g in used_g:
                    continue
                used_p.add(p); used_g.add(g)
                conf[gl[g] - 1, pl[p] - 1] += 1
        for g in range(len(gt)):
            if g not in used_g:
                missed[gl[g] - 1] += 1
        per_image.append({'file': t['file'], 'n_gt': len(gt), 'n_pred': int(keep.sum()), 'tp': len(used_g)})

        if i in pick:
            base = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            gm = [m.cpu().numpy() for m in gt]
            glines = [tuple(map(tuple, t['keypoints'][g, :, :2].tolist())) if bool(t['width_valid'][g]) else None
                      for g in range(len(gt))]
            left = paint(base, gm, gl.tolist(), glines)
            pmn = [m.cpu().numpy() for m in pm]
            plines = [tuple(map(tuple, k.tolist())) for k in pk]
            right = paint(base, pmn, pl.tolist(), plines)
            W, H = left.size
            both = Image.new('RGB', (2 * W + 8, H), (255, 255, 255))
            both.paste(left, (0, 0)); both.paste(right, (W + 8, 0))
            s = 1400 / both.size[0]
            if s < 1:
                both = both.resize((round(both.size[0] * s), round(both.size[1] * s)), Image.LANCZOS)
            both.save(os.path.join(out, 'renders', f'{split}_{t["file"]}.jpg'), quality=82)
            per_image[-1]['render'] = f'{split}_{t["file"]}.jpg'
    return conf, missed, per_image


@click.command()
@click.argument('data')
@click.argument('ckpt')
@click.argument('out')
@click.option('--split', default='test')
@click.option('--renders', default=12)
def main(data, ckpt, out, split, renders):
    dev = 'cuda'
    os.makedirs(out, exist_ok=True)
    model, ck = load(ckpt, dev)
    ds = TomatoSeg(data, split)
    m = evaluate(model, ds, dev, keep_instances=True)
    conf, missed, per_image = confusion_and_renders(model, ds, dev, out, split, renders)

    sessions = {}
    for im in ds.images:
        sessions.setdefault(im['session'], []).append(im)
    per_session = {}
    for s, ims in sessions.items():
        sub = TomatoSeg(data, split)
        sub.images = ims
        r = evaluate(model, sub, dev)
        per_session[str(s)] = {'images': len(ims), 'date': ims[0]['capture'][:10], **{
            k: r[k] for k in ('ap50', 'f1', 'mean_iou', 'class_acc', 'width_kp_medabs_pct', 'width_mask_medabs_pct', 'n_gt')}}

    res = {'split': split, 'checkpoint': ckpt, 'epoch': ck['epoch'], 'classes': ds.classes, 'images': len(ds),
           **{k: v for k, v in m.items() if k != 'width_rows'}, 'confusion': conf.tolist(), 'missed': missed.tolist(),
           'per_session': per_session, 'per_image': per_image, 'width_rows': m['width_rows']}
    json.dump(res, open(os.path.join(out, f'{split}_metrics.json'), 'w'), indent=1)
    print(split, ' '.join(f'{k} {res[k]:.4f}' for k in ('ap50', 'f1', 'mean_iou', 'class_acc', 'width_kp_medabs_pct',
                                                         'width_mask_medabs_pct', 'width_kp_bias_pct', 'width_mask_bias_pct')))
    print('per session', json.dumps(per_session))


if __name__ == '__main__':
    main()
