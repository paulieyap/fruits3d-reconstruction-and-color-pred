"""Train the tomato mask + width-keypoint model on TomatoMAP-Seg (prepared by tomato2d/prepare.py).

  python3 -m tomato2d.train DATA_DIR OUT_DIR [--epochs 40] [--batch-size 4] [--max-iters N]

Writes OUT_DIR/last.pt, OUT_DIR/best.pt (best val AP@0.5) and OUT_DIR/metrics.csv (one row per evaluation).
COCO-pretrained weights are downloaded to $TORCH_HOME on first use.
"""
import csv
import math
import os
import time

import click
import torch
from torch.utils.data import DataLoader

from tomato2d.data import TomatoSeg, collate
from tomato2d.metrics import evaluate
from tomato2d.model import build_model, model_targets


@click.command()
@click.argument('data')
@click.argument('out')
@click.option('--epochs', default=40)
@click.option('--batch-size', default=4)
@click.option('--lr', default=0.01)
@click.option('--eval-every', default=4)
@click.option('--workers', default=4)
@click.option('--max-iters', default=0, help='stop after this many iterations (smoke test)')
@click.option('--seed', default=0)
def main(data, out, epochs, batch_size, lr, eval_every, workers, max_iters, seed):
    torch.manual_seed(seed)
    os.makedirs(out, exist_ok=True)
    dev = 'cuda'
    train_ds, val_ds = TomatoSeg(data, 'train', train=True), TomatoSeg(data, 'val')
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate,
                        persistent_workers=workers > 0, drop_last=True)
    model = build_model(num_classes=len(train_ds.classes)).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    total = epochs * len(loader)
    warm = min(300, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: (i + 1) / warm if i < warm else 0.5 * (1 + math.cos(math.pi * (i - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler('cuda')
    print(f'train {len(train_ds)} images ({len(loader)} it/epoch), val {len(val_ds)}', flush=True)

    best, it, rows = -1.0, 0, []
    for ep in range(epochs):
        model.train()
        t0, run = time.time(), {}
        for imgs, targets in loader:
            imgs = [x.to(dev) for x in imgs]
            with torch.autocast('cuda', dtype=torch.float16):
                losses = model(imgs, model_targets(targets, dev))
            loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in losses.items():
                run[k] = run.get(k, 0) + float(v)
            it += 1
            if max_iters and it >= max_iters:
                break
        n = max(1, min(len(loader), it - ep * len(loader)))
        msg = ' '.join(f'{k.replace("loss_", "")} {v / n:.3f}' for k, v in run.items())
        print(f'epoch {ep + 1}/{epochs} {time.time() - t0:.0f}s lr {sched.get_last_lr()[0]:.4f} | {msg}', flush=True)

        last = (ep + 1 == epochs) or (max_iters and it >= max_iters)
        if (ep + 1) % eval_every == 0 or last:
            m = evaluate(model, val_ds, dev)
            row = {'epoch': ep + 1, 'iter': it, **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
                   **{k: round(v / n, 4) for k, v in run.items()}}
            rows.append(row)
            print('  val ' + ' '.join(f'{k} {row[k]}' for k in ('ap50', 'f1', 'mean_iou', 'class_acc',
                                                                   'width_kp_medabs_pct', 'width_mask_medabs_pct')), flush=True)
            ck = {'model': model.state_dict(), 'classes': train_ds.classes, 'epoch': ep + 1, 'val': m}
            torch.save(ck, os.path.join(out, 'last.pt'))
            if m['ap50'] > best:
                best = m['ap50']
                torch.save(ck, os.path.join(out, 'best.pt'))
            with open(os.path.join(out, 'metrics.csv'), 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[-1].keys()))
                w.writeheader()
                w.writerows(rows)
        if max_iters and it >= max_iters:
            break
    print(f'done, best val AP50 {best:.4f}', flush=True)


if __name__ == '__main__':
    main()
