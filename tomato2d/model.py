"""Mask R-CNN (torchvision, ResNet-50 FPN v2) with an extra 2-point keypoint head for the width endpoints.

torchvision's RoIHeads runs mask and keypoint branches side by side when both are set, so one model
predicts box, ripeness class, mask and the two width keypoints for every tomato.
"""
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNHeads, KeypointRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign


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
