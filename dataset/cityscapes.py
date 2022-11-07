import imp
import os
from tkinter.messagebox import NO

from torch import uint8
import torch
from torch.utils import data
from .base_dataset import BaseDataset
import random
import numpy as np
import PIL
from PIL import Image
import math
import cv2


class Cityscapes(BaseDataset):

    def transform_mask(self, label):
        if isinstance(label, PIL.Image.Image):
            label = np.array(label, dtype=np.uint8)
        return label

    def __getitem__(self, item):
        image_path = self.image_list[item]
        label_path = self.label_list[item]
        image = self.read_image(image_path)
        label = self.read_label(label_path)
        label = self.transform_mask(np.array(label))
        if self.domain == 'source':
            trans_dict = self.transform(image=image, mask=label)
        else:
            full_image = np.array(image).astype(np.float32)
            full_image = cv2.resize(full_image, dsize=(2048,1024), interpolation=cv2.INTER_LINEAR)
            # if self.phase == 'train': label = self.class_balance_drop(label)
            trans_dict = self.transform(image=image, mask=label,full_image=full_image)
            
        data_dict = {
            'name': label_path,
        }
        data_dict.update(trans_dict)
        return data_dict

    def class_balance_drop(self, label):
        # rate = np.zeros(19)
        alpha = 0.5
        label_size = len(label[label!=255])
        for i in range(19):
            num_label_i = len(label[label==i])
            if num_label_i > 0:
                rate = num_label_i/label_size
                drop_rate = math.exp(alpha*rate)-1
                if drop_rate < 0.1: continue
                drop_num = num_label_i*drop_rate
                cord_x, cord_y = np.where(label == i)
                cord_x, cord_y = np.expand_dims(cord_x, axis=1), np.expand_dims(cord_y, axis=1)
                if(drop_num > len(cord_x)-1): drop_num = len(cord_x)-1
                cord = np.concatenate([cord_x, cord_y], axis=1)
                cord = cord.tolist()
                s_cord = random.sample(cord, int(drop_num)+1)
                s_cord = np.array(s_cord)
                s_cord_x, s_cord_y = s_cord[:,0], s_cord[:,1]
                label[s_cord_x, s_cord_y] = 255
        return label

