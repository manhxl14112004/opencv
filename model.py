from __future__ import annotations

import random
import numpy as np
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

CLASSES = ["cube", "cylinder"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class EdgeTruncation:
    """Remove up to 22% at one edge; applied to TRAIN samples only."""
    def __init__(self, maximum=0.22):
        self.maximum=maximum
    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > 0.55:
            return image
        w, h = image.size
        edge, amount = random.randrange(4), random.uniform(0.04, self.maximum)
        bounds = [0, 0, w, h]
        if edge == 0: bounds[0] = int(w * amount)
        elif edge == 1: bounds[1] = int(h * amount)
        elif edge == 2: bounds[2] = int(w * (1 - amount))
        else: bounds[3] = int(h * (1 - amount))
        return image.crop(bounds)


class LightingVariation:
    """Train-only exposure, gamma, white balance and sensor noise variation."""
    def __call__(self,image:Image.Image)->Image.Image:
        rgb=np.asarray(image,dtype=np.float32)/255
        gamma=random.uniform(0.65,1.9)
        exposure=random.uniform(0.5,1.35)
        # Include both warm household lamps and cool phone/LED lighting.
        warmth=random.uniform(-0.25,0.25)
        gains=np.array([1+warmth,random.uniform(0.92,1.08),1-warmth],np.float32)
        rgb=np.power(rgb,gamma)*exposure*gains
        if random.random()<0.35:
            rgb+=np.random.normal(0,random.uniform(0.002,0.018),rgb.shape)
        return Image.fromarray(np.uint8(np.clip(rgb,0,1)*255))


def preprocess(training: bool = False, foreground: bool = False, low_light: bool = False):
    steps = []
    if training:
        if low_light:steps.append(LightingVariation())
        steps += [EdgeTruncation(0.38 if foreground else 0.22), transforms.RandomHorizontalFlip(),
                  transforms.ColorJitter(0.22, 0.22, 0.35, 0.08) if foreground else transforms.ColorJitter(0.18, 0.18, 0.12, 0.02),
                  transforms.RandomRotation(8,fill=(128,128,128) if foreground else 0)]
        if foreground:steps += [transforms.RandomGrayscale(p=0.25)]
    steps += [transforms.Resize((224, 224)), transforms.ToTensor(),
              transforms.Normalize(MEAN, STD)]
    if training:
        steps += [transforms.RandomErasing(p=0.25, scale=(0.02, 0.12))]
    return transforms.Compose(steps)


class SortNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(576, len(CLASSES))

    def embedding(self, x):
        return nn.functional.normalize(self.pool(self.features(x)).flatten(1), dim=1)

    def forward(self, x):
        return self.head(self.embedding(x))


def decision(probabilities, embedding, centroids, distance_limits, threshold=0.85, margin=0.25):
    values, indices = probabilities.sort(descending=True)
    winner = int(indices[0])
    confidence, separation = float(values[0]), float(values[0] - values[1])
    distance = float(1.0 - torch.dot(embedding, centroids[winner]))
    reasons = []
    if confidence < threshold: reasons.append("low_confidence")
    if separation < margin: reasons.append("low_margin")
    if distance > float(distance_limits[winner]): reasons.append("embedding_outlier")
    return {
        "class_code": 3 if reasons else winner + 1,
        "product_type": "unknown" if reasons else CLASSES[winner],
        "confidence": confidence, "margin": separation,
        "embedding_distance": distance, "rejection_reasons": reasons,
    }
