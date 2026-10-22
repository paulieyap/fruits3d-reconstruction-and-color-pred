"""Load converted shape-completion samples (any crop) through pcdiff's own loaders and check alignment.

Run in the pcdiff container with the dataset mounted at /data:
  python3 tools/check_samples.py /data OUT_DIR
"""
import os
import sys
import json
import contextlib
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from pcdiff.dataloader import ShapeCompletionDataset
from pcdiff.datasets.igg_fruit import IGGFruit
try:  # pulls in MinkowskiEngine, which may need the GPU runtime
    from pcdiff.train import FruitCollation
except Exception as e:
    print("!! collation check skipped:", type(e).__name__, str(e)[:120])
    FruitCollation = None

root, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
rows = []
for split in ("train", "val", "test"):
    if not os.path.isdir(os.path.join(root, split)):
        continue
    with contextlib.redirect_stdout(io.StringIO()):
        scd = ShapeCompletionDataset(data_source=root, split=split)
    igg = IGGFruit(root, split=split)
    for i in range(len(scd)):
        with contextlib.redirect_stdout(io.StringIO()):
            item = scd[i]
        fid = item["fruit_id"]
        part = item["rgbd_pcd"]
        gt = np.asarray(item["groundtruth_pcd"].points)
        d_part_gt, _ = cKDTree(gt).query(part[:, :3])      # input should sit on the gt
        d_gt_part, _ = cKDTree(part[:, :3]).query(gt)      # how much of the gt the view covers
        colored = float((item["aligned_data"][:, 3] != -1).mean())
        rows.append({"split": split, "id": fid, "n_part": len(part), "n_gt": len(gt),
                     "part_to_gt_med_mm": 1000 * float(np.median(d_part_gt)),
                     "part_to_gt_p90_mm": 1000 * float(np.percentile(d_part_gt, 90)),
                     "gt_covered_frac": float((d_gt_part < 0.002).mean()),
                     "gt_colored_frac": colored,
                     "gt_extent_mm": (1000 * (gt.max(0) - gt.min(0))).round(1).tolist()})
        igg_s = igg[[s[1] for s in igg.samples].index(fid)]
        assert igg_s["points"].shape[1] == 6 and igg_s["extra"]["gt_points"].shape[1] == 6
        if FruitCollation and split == "train" and i == 0:
            batch = FruitCollation()([item])  # the exact path training takes
            print("collation ok:", {k: tuple(v.shape) for k, v in batch.items()
                                    if hasattr(v, "shape") and k.startswith("pcd")})
        if len([r for r in rows if r["split"] == split]) <= 4:  # a few previews per split
            fig = plt.figure(figsize=(12, 4))
            for j, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
                ax = fig.add_subplot(1, 3, j + 1)
                ax.scatter(gt[:, a] * 1000, gt[:, b] * 1000, s=1, c="0.7", label="gt (fused)")
                ax.scatter(part[:, a] * 1000, part[:, b] * 1000, s=1, c=np.clip(part[:, 3:], 0, 1),
                           label="input view")
                ax.set_aspect("equal"); ax.set_xlabel("xyz"[a] + " mm"); ax.set_ylabel("xyz"[b] + " mm")
            fig.axes[0].legend(markerscale=8, loc="upper left", fontsize=7)
            fig.suptitle(f"{split}/{fid}")
            fig.tight_layout(); fig.savefig(os.path.join(out, f"{split}__{fid}.png"), dpi=110)
            plt.close(fig)

json.dump(rows, open(os.path.join(out, "check.json"), "w"), indent=1)
for k in ("n_part", "n_gt", "part_to_gt_med_mm", "part_to_gt_p90_mm", "gt_covered_frac",
          "gt_colored_frac"):
    v = np.array([r[k] for r in rows])
    print(f"{k:20s} min {v.min():8.3f}  median {np.median(v):8.3f}  max {v.max():8.3f}")
ext = np.array([r["gt_extent_mm"] for r in rows])
print("gt extent mm median", np.median(ext, 0).round(1), " samples:",
      {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")})
