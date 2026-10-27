"""Mask R-CNN (torchvision, ResNet-50 FPN v2) with an extra 2-point keypoint head for the width endpoints.

torchvision's RoIHeads runs mask and keypoint branches side by side when both are set, so one model
predicts box, ripeness class, mask and the two width keypoints for every tomato.

Fruits whose ripeness is unknown (pseudo-labels from another camera) carry label num_classes + 1. For them the
classifier only learns "some fruit class, not background" (log of the summed fruit probabilities), the box
regression uses the predicted fruit class and the mask loss is averaged over all fruit-class mask channels.
With no such labels both losses equal torchvision's.

CLASS_WEIGHT (set by tomato2d/train.py --balance) optionally weights the classification loss of known labels per
class (index 0 = background), so rare ripeness classes count more.
"""
import torch
import torch.nn.functional as F
from torchvision.models.detection import roi_heads
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNHeads, KeypointRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign

CLASS_WEIGHT = None  # (num_classes + 1,) tensor or None


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    N, K = class_logits.shape
    unknown = labels >= K
    logp = F.log_softmax(class_logits.float(), 1)
    known_nll = -logp.gather(1, labels.clamp(max=K - 1)[:, None])[:, 0]
    if CLASS_WEIGHT is not None:
        known_nll = known_nll * CLASS_WEIGHT.to(known_nll.device)[labels.clamp(max=K - 1)]
    classification_loss = torch.where(unknown, -torch.logsumexp(logp[:, 1:], 1), known_nll).mean()

    pos = torch.where(labels > 0)[0]
    labels_pos = torch.where(unknown[pos], class_logits[pos, 1:].detach().argmax(1) + 1, labels[pos])
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
    box_loss = F.smooth_l1_loss(box_regression[pos, labels_pos], regression_targets[pos], beta=1 / 9, reduction='sum')
    return classification_loss, box_loss / labels.numel()


def maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
    S, K = mask_logits.shape[-1], mask_logits.shape[1]
    labels = torch.cat([gl[idx] for gl, idx in zip(gt_labels, mask_matched_idxs)], dim=0)
    targets = torch.cat([roi_heads.project_masks_on_boxes(m, p, i, S)
                         for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)], dim=0)
    if targets.numel() == 0:
        return mask_logits.sum() * 0
    unknown = labels >= K
    known = F.binary_cross_entropy_with_logits(
        mask_logits[torch.arange(len(labels), device=labels.device), labels.clamp(max=K - 1)], targets,
        reduction='none').mean((1, 2))
    every = F.binary_cross_entropy_with_logits(
        mask_logits[:, 1:], targets[:, None].expand(-1, K - 1, -1, -1), reduction='none').mean((1, 2, 3))
    return torch.where(unknown, every, known).mean()


roi_heads.fastrcnn_loss = fastrcnn_loss  # RoIHeads.forward looks these up at call time
roi_heads.maskrcnn_loss = maskrcnn_loss


def build_model(num_classes=5, num_keypoints=2, pretrained=True, min_size=(640, 704, 768, 832, 896), max_size=1333):
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
    model = maskrcnn_resnet50_fpn_v2(weights=weights, min_size=min_size, max_size=max_size)
    rh = model.roi_heads
    rh.box_predictor = FastRCNNPredictor(rh.box_predictor.cls_score.in_features, num_classes + 1)
    rh.mask_predictor = MaskRCNNPredictor(256, 256, num_classes + 1)
    rh.keypoint_roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=14, sampling_ratio=2)
    rh.keypoint_head = KeypointRCNNHeads(256, tuple(512 for _ in range(8)))
    rh.keypoint_predictor = KeypointRCNNPredictor(512, num_keypoints)
    return model


def model_targets(targets, device):
    """Only the keys torchvision's detection heads use."""
    keep = ('boxes', 'labels', 'masks', 'keypoints')
    return [{k: t[k].to(device) for k in keep} for t in targets]


def load(path, device='cuda'):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(num_classes=len(ck['classes']), pretrained=False)
    model.load_state_dict(ck['model'])
    return model.to(device).eval(), ck
