import Vot as Vot
import os
import cv2
import yaml
import pprint
import sys
import logging
import cv2
from easydict import EasyDict as edict
import numpy as np
from siamfc import TrackerSiamFC



net_path = '/home/vision/Desktop/Master/siamfc-pytorch/pretrained/siamfc_alexnet_e50.pth'
tracker = TrackerSiamFC(net_path=net_path)
handle = Vot.VOT("rectangle")

image_file = handle.frame()

if not image_file:
    sys.exit(0)

frame = cv2.imread(image_file)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
bbox = handle.region()
lx, ly, w, h = bbox.x, bbox.y, bbox.width, bbox.height
# cx, cy = lx + w/2, ly + h/2
init_bbox = lx, ly, w, h

target_pos = np.array([lx + w/2, ly + h/2])
target_sz = np.array([w, h])

state = tracker.init(frame, init_bbox)

while True:
    image_file = handle.frame()
    if not image_file:
        print("of")
        break
    frame = cv2.imread(image_file)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    location = tracker.update(frame)

    x1, y1, x2, y2 = int(location[0]), int(location[1]), int(location[0] + location[2]), int(location[1] + location[3])
    #x1, y1, x2, y2 = center2corner((cx, cy, w, h))
    handle.report(Vot.Rectangle(x1, y1, (x2-x1), (y2-y1)))