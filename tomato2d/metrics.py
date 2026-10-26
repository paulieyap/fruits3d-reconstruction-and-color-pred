"""Evaluation for tomato instances: detection AP@0.5 (mask IoU), mask quality, ripeness class accuracy and
width error (keypoint head vs widest horizontal chord of the predicted mask)."""
import numpy as np
import torch


def widest_chord(mask):
    """Width (px) and endpoints of the widest horizontal chord of a boolean mask."""
    rows = torch.nonzero(mask.any(1)).flatten()
    if len(rows) == 0:
        return 0.0, None
    sub = mask[rows]
    W = mask.shape[1]
    idx = torch.arange(W, device=mask.device)
    left = torch.where(sub, idx, W).min(1).values
    right = torch.where(sub, idx, -1).max(1).values
    w = right - left
    k = int(w.argmax())
    return float(w[k]), ((float(left[k]), float(rows[k])), (float(right[k]), float(rows[k])))


def mask_iou(a, b):
    """a (P, H, W) bool, b (G, H, W) bool -> (P, G)."""
    a = a.flatten(1).float()
    b = b.flatten(1).float()
    inter = a @ b.T
    return inter / (a.sum(1)[:, None] + b.sum(1)[None] - inter).clamp(min=1)


def average_precision(scores, tp, n_gt):
    if n_gt == 0:
        return float('nan')
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(tp, dtype=float)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]  # precision envelope
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


@torch.no_grad()
def evaluate(model, dataset, device='cuda', score_thr=0.5, iou_thr=0.5, keep_instances=False):
    model.eval()
    all_scores, all_tp, n_gt = [], [], 0
    tp = fp = fn = 0
    ious, cls_ok, w_rows = [], [], []
    for i in range(len(dataset)):
        img, t = dataset[i]
        out = model([img.to(device)])[0]
        gt = t['masks'].to(device).bool()
        n_gt += len(gt)
        pm = out['masks'][:, 0] > 0.5
        sc = out['scores']
        iou = mask_iou(pm, gt) if len(pm) and len(gt) else torch.zeros(len(pm), len(gt), device=device)

        # AP: all predictions, greedy by score
        taken = torch.zeros(len(gt), dtype=torch.bool, device=device)
        for p in torch.argsort(sc, descending=True).tolist():
            hit = False
            if len(gt):
                cand = iou[p].clone()
                cand[taken] = -1
                g = int(cand.argmax())
                if cand[g] >= iou_thr:
                    taken[g] = True
                    hit = True
            all_scores.append(float(sc[p]))
            all_tp.append(hit)

        # operating point: predictions above score_thr, greedy by IoU
        keep = torch.nonzero(sc >= score_thr).flatten()
        sub = iou[keep] if len(keep) else torch.zeros(0, len(gt), device=device)
        used_p, used_g, matches = set(), set(), []
        if sub.numel():
            for flat in torch.argsort(sub.flatten(), descending=True).tolist():
                p, g = divmod(flat, len(gt))
                if sub[p, g] < iou_thr:
                    break
                if p in used_p or g in used_g:
                    continue
                used_p.add(p); used_g.add(g); matches.append((int(keep[p]), g))
        tp += len(matches)
        fp += len(keep) - len(matches)
        fn += len(gt) - len(matches)
        for p, g in matches:
            ious.append(float(iou[p, g]))
            if int(t['labels'][g]) <= len(dataset.classes):  # skip fruits of unknown ripeness
                cls_ok.append(int(out['labels'][p]) == int(t['labels'][g]))
            if bool(t['width_valid'][g]):
                gw = float(torch.linalg.norm(t['keypoints'][g, 1, :2] - t['keypoints'][g, 0, :2]))
                kp = out['keypoints'][p, :, :2]
                kw = float(torch.linalg.norm(kp[1] - kp[0]))
                mw, _ = widest_chord(pm[p])
                w_rows.append({'file': t['file'], 'gt_px': gw, 'kp_px': kw, 'mask_px': mw,
                               'kp_err_pct': (kw / gw - 1) * 100, 'mask_err_pct': (mw / gw - 1) * 100,
                               'label': int(t['labels'][g]), 'iou': float(iou[p, g])})

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    kp_err = np.array([r['kp_err_pct'] for r in w_rows]) if w_rows else np.array([np.nan])
    mk_err = np.array([r['mask_err_pct'] for r in w_rows]) if w_rows else np.array([np.nan])
    res = {
        'ap50': average_precision(all_scores, all_tp, n_gt), 'precision': prec, 'recall': rec,
        'f1': 2 * prec * rec / max(prec + rec, 1e-9), 'mean_iou': float(np.mean(ious)) if ious else float('nan'),
        'class_acc': float(np.mean(cls_ok)) if cls_ok else float('nan'), 'n_gt': n_gt, 'n_width': len(w_rows),
        'width_kp_medabs_pct': float(np.median(np.abs(kp_err))), 'width_kp_bias_pct': float(np.median(kp_err)),
        'width_mask_medabs_pct': float(np.median(np.abs(mk_err))), 'width_mask_bias_pct': float(np.median(mk_err)),
        'width_kp_p90_pct': float(np.percentile(np.abs(kp_err), 90)), 'width_mask_p90_pct': float(np.percentile(np.abs(mk_err), 90)),
    }
    if keep_instances:
        res['width_rows'] = w_rows
    return res
