"""TomatoMAP-Seg samples (from tomato2d/prepare.py) as torchvision detection targets."""
import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class TomatoSeg(Dataset):
    """image (3, H, W) float in [0, 1]; target with boxes (N, 4) xyxy, labels (N,) in 1..5,
    masks (N, H, W) uint8, keypoints (N, 2, 3) = two width endpoints (x, y, visibility), visibility 0
    where the width is not trusted (occluded or cut off), so those fruits add no keypoint loss.
    A label outside `classes` (e.g. 'tomato' from tomato2d/prepare_views.py) means ripeness unknown and maps to
    len(classes) + 1. `classes` overrides the set's own list (e.g. to train TomatoMAP photos under the ripeness
    grades of another set: their labels are then all unknown)."""

    def __init__(self, root, split, train=False, classes=None):
        ann = json.load(open(os.path.join(root, 'annotations.json')))
        self.root, self.train, self.classes = root, train, classes or ann['classes']
        self.images = [im for im in ann['images'] if im['split'] == split]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        im = self.images[i]
        img = TF.to_tensor(Image.open(os.path.join(self.root, 'images', im['file'] + '.jpg')).convert('RGB'))
        masks = torch.from_numpy(np.load(os.path.join(self.root, 'masks', im['file'] + '.npz'))['masks'].astype(np.uint8))
        boxes = torch.tensor([x['bbox'] for x in im['instances']], dtype=torch.float32)
        labels = torch.tensor([self.classes.index(x['label']) + 1 if x['label'] in self.classes else len(self.classes) + 1
                               for x in im['instances']], dtype=torch.int64)
        kps = torch.zeros(len(im['instances']), 2, 3)
        for k, x in enumerate(im['instances']):
            if x['width_kp'] is not None:
                kps[k, :, :2] = torch.tensor(x['width_kp'])
                kps[k, :, 2] = 2.0 if x['width_valid'] else 0.0
        valid = torch.tensor([bool(x['width_valid']) for x in im['instances']])

        if self.train:
            if random.random() < 0.5:  # horizontal flip; keep the left endpoint first
                W = img.shape[-1]
                img, masks = img.flip(-1), masks.flip(-1)
                boxes = torch.stack([W - boxes[:, 2], boxes[:, 1], W - boxes[:, 0], boxes[:, 3]], 1)
                kps[:, :, 0] = W - 1 - kps[:, :, 0]
                kps = kps.flip(1)
            img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
            img = TF.adjust_saturation(img, random.uniform(0.8, 1.2))
            img = TF.adjust_hue(img, random.uniform(-0.03, 0.03))  # small: hue carries ripeness

        return img, {'boxes': boxes, 'labels': labels, 'masks': masks, 'keypoints': kps,
                     'width_valid': valid, 'file': im['file']}


def collate(batch):
    return tuple(zip(*batch))
