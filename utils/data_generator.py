import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
from preprocessing import load_image, load_mask

class IDDSegmentationGenerator(Sequence):
    def __init__(self, image_paths, mask_paths, batch_size=8, target_size=(256, 256), num_classes=27):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.target_size = target_size
        self.num_classes = num_classes

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_images = self.image_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_masks = self.mask_paths[idx * self.batch_size:(idx + 1) * self.batch_size]

        X = [load_image(img, self.target_size) for img in batch_images]
        Y = [load_mask(msk, self.target_size, self.num_classes) for msk in batch_masks]

        return np.array(X), np.array(Y)
