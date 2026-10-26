"""SAM 3 text-prompted tomato instances for every extracted view.

Writes <view>/inst.png (uint16, 0 = background, k = instance k, higher score wins
overlaps) and <view>/inst.json ({"scores": [...]}, index k-1 = instance k).
"""
import argparse, glob, json, os, sys

import cv2
import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("views_root")
    ap.add_argument("--prompt", default="tomato")
    ap.add_argument("--thr", type=float, default=0.4)
    a = ap.parse_args()
    ck = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--facebook--sam3/snapshots/*/sam3.pt"))[0]
    model = build_sam3_image_model(checkpoint_path=ck, load_from_HF=False)
    proc = Sam3Processor(model, confidence_threshold=a.thr)
    views = sorted(glob.glob(os.path.join(a.views_root, "*", "*", "color.jpg")))
    todo = [v for v in views if not os.path.isfile(v.replace("color.jpg", "inst.json"))]
    print(len(views), "views,", len(todo), "to do", flush=True)
    for i, p in enumerate(todo):
        img = Image.open(p).convert("RGB")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            st = proc.set_image(img)
            out = proc.set_text_prompt(state=st, prompt=a.prompt)
        masks = out["masks"].squeeze(1).cpu().numpy() > 0
        scores = out["scores"].float().cpu().numpy()
        order = np.argsort(scores)  # paint low score first, high score overwrites
        lab = np.zeros(masks.shape[1:] if len(masks) else img.size[::-1], np.uint16)
        for k in order:
            lab[masks[k]] = k + 1
        d = os.path.dirname(p)
        cv2.imwrite(os.path.join(d, "inst.png"), lab)
        json.dump({"scores": scores.round(4).tolist(), "prompt": a.prompt}, open(os.path.join(d, "inst.json"), "w"))
        if i % 200 == 0:
            print(i, p, len(scores), flush=True)


if __name__ == "__main__":
    main()
