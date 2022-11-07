import os
import numpy as np
import cv2
from PIL import Image

from dataset.base_dataset import BaseDataset

from configs.config import config

class Synthia(BaseDataset):
    def read_label(self, label_path):
        label = cv2.imread(label_path, -1)
        label = Image.fromarray(label[:,:,-1])
        return label

    def transform_mask(self, label):
        label = np.array(label, dtype=np.uint8)
        id_map = {3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5,
                  15: 6, 9: 7, 6: 8, 16: 9, 1: 10, 10: 11, 17: 12,
                  8: 13, 18: 14, 19: 15, 20: 16, 12: 17, 11: 18}
        label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
        for k, v in id_map.items():
            label_copy[label == k] = v
        return label_copy