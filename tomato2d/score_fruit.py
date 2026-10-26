"""Score a checkpoint per whole fruit (data from prepare.py --whole-fruit).

TomatoMAP labels a fruit crossed by a stem as several pieces, and a model trained on pieces predicts pieces.
Three ways of turning its predictions into fruits are scored against the whole-fruit ground truth:

  raw     predictions as they are
  rule    pieces whose dilated masks touch are merged when the union stays compact (closed solidity >= --min-solidity);
          needs no labels, so it is what would run on the robot
  oracle  each prediction joins the ground-truth fruit holding most of its pixels (upper bound for any grouping)

A merged fruit takes the highest score, the label covering the most pixels, and its width from the widest chord of
the union. The keypoint width is only reported for fruits made of one prediction.

  python3 -m tomato2d.score_fruit DATA_FRUIT CKPT OUT_DIR [--split test]
"""
import json
import os

import click
import cv2
import numpy as np
import torch

from tomato2d.data import TomatoSeg
from tomato2d.metrics import average_precision, mask_iou, widest_chord
from tomato2d.model import load
from tomato2d.prepare import solidity


def closed_solidity(m):
    k = max(5, int(round(0.25 * np.sqrt(m.sum()))))
    return solidity(cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))))


def dilate(m, k):
    return cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)


def combine(masks, scores, labels, kps, groups):
    """groups: list of index lists -> fruits as (mask, score, label, keypoints or None, n_pieces)."""
    out = []
    for g in groups:
        m = np.logical_or.reduce([masks[i] for i in g])
        area = {}
        for i in g:
            area[labels[i]] = area.get(labels[i], 0) + int(masks[i].sum())
        out.append((m, max(scores[i] for i in g), max(area, key=area.get), kps[g[0]] if len(g) == 1 else None, len(g)))
    return out


def group_rule(masks, scores, min_solidity, gap):
    """Greedy agglomeration of touching pieces, best-scoring first, while the union stays fruit-shaped."""
    n = len(masks)
    groups = [[i] for i in range(n)]
    union = [m.copy() for m in masks]
    changed = True
    while changed:
        changed = False
        order = sorted(range(len(groups)), key=lambda a: -max((scores[i] for i in groups[a]), default=-1))
        for a in order:
            if not groups[a]:
                continue
            da = dilate(union[a], gap)
            best = None
            for b in order:
                if b == a or not groups[b] or not (da & union[b]).any():
                    continue
                u = union[a] | union[b]
                s = closed_solidity(u)
                if s >= min_solidity and (best is None or s > best[1]):
                    best = (b, s)
            if best:
                b = best[0]
                groups[a] += groups[b]; union[a] |= union[b]
                groups[b] = []
                changed = True
    return [g for g in groups if g]


def group_oracle(masks, gt, min_frac=0.5, gap=7):
    gd = [dilate(g, gap) for g in gt]
    by_fruit, alone = {}, []
    for i, m in enumerate(masks):
        a = max(int(m.sum()), 1)
        frac = [int((m & g).sum()) / a for g in gd]
        j = int(np.argmax(frac)) if frac else -1
        if j >= 0 and frac[j] >= min_frac:
            by_fruit.setdefault(j, []).append(i)
        else:
            alone.append([i])
    return list(by_fruit.values()) + alone


def match(fruits, t, score_thr, iou_thr, dev):
    """AP inputs (all fruits) and operating-point rows (score >= score_thr) against whole-fruit ground truth."""
    gt = t['masks'].to(dev).bool()
    if fruits:
        pm = torch.from_numpy(np.stack([f[0] for f in fruits])).to(dev)
        iou = mask_iou(pm, gt) if len(gt) else torch.zeros(len(fruits), 0, device=dev)
    else:
        iou = torch.zeros(0, len(gt), device=dev)
    sc = [f[1] for f in fruits]
    taken, ap_tp = set(), []
    for p in np.argsort(-np.asarray(sc)):
        hit = False
        if len(gt):
            cand = iou[p].clone()
            cand[list(taken)] = -1
            g = int(cand.argmax())
            if cand[g] >= iou_thr:
                taken.add(g); hit = True
        ap_tp.append(hit)
    ap_sc = [sc[p] for p in np.argsort(-np.asarray(sc))]

    keep = [p for p in range(len(fruits)) if sc[p] >= score_thr]
    used_p, used_g, rows = set(), set(), []
    if keep and len(gt):
        sub = iou[keep]
        for flat in torch.argsort(sub.flatten(), descending=True).tolist():
            q, g = divmod(flat, len(gt))
            if sub[q, g] < iou_thr:
                break
            p = keep[q]
            if p in used_p or g in used_g:
                continue
            used_p.add(p); used_g.add(g)
            m, s, lab, kp, npc = fruits[p]
            r = {'file': t['file'], 'iou': float(iou[p, g]), 'cls_ok': lab == int(t['labels'][g]), 'pred_pieces': npc}
            if bool(t['width_valid'][g]):
                gw = float(torch.linalg.norm(t['keypoints'][g, 1, :2] - t['keypoints'][g, 0, :2]))
                mw, _ = widest_chord(torch.from_numpy(m))
                r.update(gt_px=gw, mask_err_pct=(mw / gw - 1) * 100)
                if kp is not None:
                    r['kp_err_pct'] = (float(np.linalg.norm(kp[1] - kp[0])) / gw - 1) * 100
            rows.append(r)
    return ap_sc, ap_tp, rows, len(keep), len(gt)


def summarise(acc):
    tp = len(acc['rows'])
    prec, rec = tp / max(acc['n_pred'], 1), tp / max(acc['n_gt'], 1)
    mk = np.array([r['mask_err_pct'] for r in acc['rows'] if 'mask_err_pct' in r])
    kp = np.array([r['kp_err_pct'] for r in acc['rows'] if 'kp_err_pct' in r])
    med = lambda a, f: float(f(a)) if len(a) else float('nan')
    return {'ap50': average_precision(acc['sc'], acc['tp'], acc['n_gt']), 'precision': prec, 'recall': rec,
            'f1': 2 * prec * rec / max(prec + rec, 1e-9), 'n_gt': acc['n_gt'], 'n_pred': acc['n_pred'], 'tp': tp,
            'mean_iou': med([r['iou'] for r in acc['rows']], np.mean),
            'class_acc': med([r['cls_ok'] for r in acc['rows']], np.mean),
            'merged_tp': sum(r['pred_pieces'] > 1 for r in acc['rows']),
            'n_width': len(mk), 'width_mask_medabs_pct': med(np.abs(mk), np.median), 'width_mask_bias_pct': med(mk, np.median),
            'width_mask_p90_pct': med(np.abs(mk), lambda a: np.percentile(a, 90)),
            'n_width_kp': len(kp), 'width_kp_medabs_pct': med(np.abs(kp), np.median)}


@click.command()
@click.argument('data')
@click.argument('ckpt')
@click.argument('out')
@click.option('--split', default='test')
@click.option('--score-thr', default=0.5)
@click.option('--min-solidity', default=0.92)
@click.option('--gap', default=7, help='dilation (px) for pieces to count as touching')
@torch.no_grad()
def main(data, ckpt, out, split, score_thr, min_solidity, gap):
    dev = 'cuda'
    os.makedirs(out, exist_ok=True)
    model, ck = load(ckpt, dev)
    model.eval()
    ds = TomatoSeg(data, split)
    multi = {im['file']: sum(x.get('pieces', 1) > 1 for x in im['instances']) for im in ds.images}
    modes = ('raw', 'rule', 'oracle')
    acc = {k: {'sc': [], 'tp': [], 'rows': [], 'n_pred': 0, 'n_gt': 0} for k in modes}
    for i in range(len(ds)):
        img, t = ds[i]
        o = model([img.to(dev)])[0]
        masks = list((o['masks'][:, 0] > 0.5).cpu().numpy())
        scores = o['scores'].cpu().tolist()
        labels = o['labels'].cpu().tolist()
        kps = list(o['keypoints'][:, :, :2].cpu().numpy())
        gt = list(t['masks'].bool().numpy())
        hi = [p for p in range(len(masks)) if scores[p] >= score_thr]
        lo = [[p] for p in range(len(masks)) if scores[p] < score_thr]  # low-score ones only feed AP, never merged
        sel = lambda idx: [masks[p] for p in idx]
        rule = [[hi[j] for j in g] for g in group_rule(sel(hi), [scores[p] for p in hi], min_solidity, gap)] + lo
        orc = [[hi[j] for j in g] for g in group_oracle(sel(hi), gt, gap=gap)] + lo
        for k, groups in (('raw', [[p] for p in range(len(masks))]), ('rule', rule), ('oracle', orc)):
            sc, tp, rows, n_pred, n_gt = match(combine(masks, scores, labels, kps, groups), t, score_thr, 0.5, dev)
            acc[k]['sc'] += sc; acc[k]['tp'] += tp; acc[k]['rows'] += rows
            acc[k]['n_pred'] += n_pred; acc[k]['n_gt'] += n_gt

    res = {'split': split, 'checkpoint': ckpt, 'epoch': ck['epoch'], 'images': len(ds), 'score_thr': score_thr,
           'min_solidity': min_solidity, 'gap': gap, 'gt_multi_piece_fruits': sum(multi.values()),
           **{k: summarise(acc[k]) for k in modes}, 'rows': {k: acc[k]['rows'] for k in modes}}
    json.dump(res, open(os.path.join(out, f'{split}_fruit_metrics.json'), 'w'), indent=1)
    for k in modes:
        s = res[k]
        print(f'{split} {k:6s} ' + ' '.join(f'{c} {s[c]:.4f}' if isinstance(s[c], float) else f'{c} {s[c]}' for c in s))


if __name__ == '__main__':
    main()
