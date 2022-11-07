import os
import numpy as np
import cv2
from PIL import Image

from dataset.base_dataset import BaseDataset

from configs.config import config

class GTAV(BaseDataset):

    def transform_mask(self, label):
        if isinstance(label, Image.Image):
            label = np.array(label)
        id_map = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
        for k, v in id_map.items():
            label_copy[label == k] = v
        return label_copy
        