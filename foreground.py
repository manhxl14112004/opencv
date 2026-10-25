"""Locate the coloured plastic products before classifying their shape.

HSV ranges locate both classes together; they never assign a class label.
These defaults are for the blue/yellow products in the supplied dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image

DEFAULTS = {
    "roi": [0.0, 0.42, 0.80, 1.0],
    "hsv_ranges": [
        {"lower": [15, 45, 130], "upper": [42, 255, 255]},
        {"lower": [80, 235, 105], "upper": [132, 255, 255]},
    ],
    "minimum_area_fraction": 0.006,
    "minimum_fill": 0.40,
    "expected_center": [0.43, 0.86],
    "padding_fraction": 0.12,
}

# Deployment-camera preset for warm lamps, blue shadows and flash. The
# saturation/geometry checks exclude the floor and objects behind CB1.
# Kept separate so existing checkpoints retain their original preprocessing.
LOW_LIGHT_DEFAULTS = {
    **DEFAULTS,
    "hsv_ranges": [
        {"lower": [15, 205, 100], "upper": [35, 255, 255]},
        {"lower": [94, 100, 45], "upper": [132, 255, 255]},
        {"lower": [80, 170, 205], "upper": [94, 255, 255]},
        {"lower": [80, 235, 105], "upper": [132, 255, 255]},
        {"lower": [15, 45, 130], "upper": [42, 255, 255]},
    ],
    "maximum_area_fraction": 0.12,
    "maximum_width_fraction": 0.38,
    "maximum_height_fraction": 0.60,
    "minimum_center_y": 0.65,
    "separate_color_masks": True,
    "reject_left_edge_background": True,
    "minimum_product_color_fraction": 0.40,
}


@dataclass
class Foreground:
    image: Image.Image
    bbox: tuple[int, int, int, int]
    area_fraction: float
    touches_frame_edge: bool


def locate_product(image: Image.Image, settings: dict | None = None) -> Foreground | None:
    config = {**DEFAULTS, **(settings or {})}
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    scale = min(1.0, 640 / width)
    small = cv2.resize(rgb, (round(width * scale), round(height * scale))) if scale < 1 else rgb
    h, w = small.shape[:2]
    roi = config["roi"]
    if len(roi) != 4 or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
        raise ValueError("foreground.roi must be [left, top, right, bottom] fractions")
    x0, y0, x1, y1 = round(roi[0]*w), round(roi[1]*h), round(roi[2]*w), round(roi[3]*h)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    masks = [cv2.inRange(hsv,np.array(limits["lower"],np.uint8),np.array(limits["upper"],np.uint8)) for limits in config["hsv_ranges"]]
    if not config.get("separate_color_masks",False):
        masks=[np.bitwise_or.reduce(masks)]
    allowed = np.zeros((h,w),dtype=np.uint8); allowed[y0:y1, x0:x1] = 255
    candidates = []
    expected_x, expected_y = config["expected_center"]
    for mask in masks:
        mask &= allowed
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            cx, cy = (x+cw/2)/w, (y+ch/2)/h
            if area < w*h*config["minimum_area_fraction"] or min(cw, ch) < 12:
                continue
            if (area > w*h*config.get("maximum_area_fraction",1)
                or cw > w*config.get("maximum_width_fraction",1)
                or ch > h*config.get("maximum_height_fraction",1)
                or cy < config.get("minimum_center_y",0)):
                continue
            if area/(cw*ch) < config["minimum_fill"]:
                continue
            # Discard clothing entering through the right edge of the search area.
            if cx > roi[2] - 0.10 and x+cw >= x1-2:
                continue
            # A blue belt/roller entering from the left must not become the
            # product when a green unknown object occupies the CB1 position.
            if config.get("reject_left_edge_background",False) and x<=x0+1 and cx<expected_x-0.15:
                continue
            proximity = ((cx-expected_x)/0.28)**2 + ((cy-expected_y)/0.38)**2
            candidates.append((area/(1+3*proximity), contour, (x,y,cw,ch), area))
    if not candidates:
        return None
    _, contour, (x,y,cw,ch), area = max(candidates, key=lambda item:item[0])
    foreground = np.zeros((h,w), np.uint8)
    cv2.drawContours(foreground, [contour], -1, 255, cv2.FILLED)
    if config.get("minimum_product_color_fraction",0)>0:
        # A warm-lit hand plus a green unknown box can join the permissive
        # yellow mask. Require enough blue/yellow pigment before shape inference.
        colors=hsv[foreground!=0];colors=colors[colors[:,1]>30]
        hue,saturation=colors[:,0],colors[:,1]
        valid=((hue>=18)&(hue<=33)&(saturation>=80))|((hue>=80)&(hue<=132)&(saturation>=100))
        if len(colors)==0 or float(valid.mean())<config["minimum_product_color_fraction"]:
            return None
    foreground = cv2.resize(foreground, (width,height), interpolation=cv2.INTER_NEAREST)
    isolated = np.full_like(rgb, 128)
    isolated[foreground != 0] = rgb[foreground != 0]
    left, top = round(x/scale), round(y/scale)
    right, bottom = round((x+cw)/scale), round((y+ch)/scale)
    side = max(right-left, bottom-top)*(1+2*config["padding_fraction"])
    center_x, center_y = (left+right)/2, (top+bottom)/2
    box = (round(center_x-side/2), round(center_y-side/2), round(center_x+side/2), round(center_y+side/2))
    # Preserve the visible object's aspect ratio even when it touches an image edge.
    crop = Image.new("RGB", (box[2]-box[0],box[3]-box[1]), (128,128,128))
    clipped = (max(0,box[0]),max(0,box[1]),min(width,box[2]),min(height,box[3]))
    crop.paste(Image.fromarray(isolated).crop(clipped), (clipped[0]-box[0],clipped[1]-box[1]))
    return Foreground(crop, (left,top,right,bottom), area/(w*h), x <= 1 or y <= 1 or x+cw >= w-1 or y+ch >= h-1)


def draw_search_region(frame, settings: dict | None = None):
    config = {**DEFAULTS, **(settings or {})}
    h,w = frame.shape[:2]
    left,top,right,bottom = config["roi"]
    cv2.rectangle(frame, (round(left*w),round(top*h)), (min(w-1,round(right*w)),min(h-1,round(bottom*h))), (0,220,255), 2)
    cv2.putText(frame, "VUNG NHAN DIEN CB1", (round(left*w)+6,round(top*h)+22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,255), 1, cv2.LINE_AA)
    return frame
