
import cv2
import numpy as np
import os

def load_image(path, target_size=(256, 256)):
    img = cv2.imread(path)
    img = cv2.resize(img, target_size)
    return img.astype(np.float32) / 255.0

def load_mask(path, target_size=(256, 256), num_classes=27):
    mask = cv2.imread(path, 0)
    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    mask[mask == 255] = 26

    mask = np.eye(num_classes, dtype=np.uint8)[mask]
    return mask


