import os
import cv2
import numpy as np
from PIL import Image
from torch import zero_
from torch.utils.data import Dataset
import dataset.augmentation as A
import torch


class BaseDataset(Dataset):
    def __init__(self, config, phase='train', domain='source'):
        self.config = config
        self.phase = phase
        self.domain = domain
        self.transform = None
        if domain == 'source':
            self.transform = self.get_data_transform(config.source_input, phase)
        else:
             self.transform = self.get_data_transform(config.target_input, phase)
        self.weak_transform = None
        self.strong_transform = None
        self.image_list = None
        self.label_list = None
        if phase == 'val':
            self.image_list, self.label_list = self.get_image_mask(config.val_dataset)
        else:
            if domain == 'source':
                self.image_list, self.label_list = self.get_image_mask(config.source_dataset)
            elif domain == 'pseudo':
                self.image_list, self.label_list = self.get_image_mask(config.pseudo_dataset)
            else:
                self.image_list, self.label_list = self.get_image_mask(config.target_dataset)
        print("find images:", len(self.image_list))
        print("find masks:", len(self.label_list))

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
            trans_dict = self.transform(image=image, mask=label, full_image=full_image)
        data_dict = {
            'name': label_path,
        }
        data_dict.update(trans_dict)
        return data_dict

    def __len__(self):
        return len(self.image_list)

    def read_image(self, image_path):
        image = Image.open(image_path)
        return image

    def read_label(self, label_path):
        '''
        npy_path = label_path.replace('label.png', 'softmax.npy')
        if os.path.exists(npy_path) == True and self.phase=='train':
            label = np.load(npy_path)
        else:
        ''' 
        label = Image.open(label_path)
        return label

    def data_aug(self, image, label):
        return image, label

    def transform_mask(self, label):
        return label

    def get_data_transform(self, opt, phase):
        transform_list = []
        if phase == 'train':
            if opt.resize is not None:
                transform_list.append(A.Resize((opt.resize[0], opt.resize[1])))
            if opt.gamma is not None:
                transform_list.append(A.AdjustGamma(opt.gamma[0], opt.gamma[1]))
            if opt.brightness is not None:
                transform_list.append(A.AdjustBrightness(opt.brightness[0], opt.brightness[1]))
            if opt.contrast is not None:
                transform_list.append(A.AdjustContrast(opt.contrast[0], opt.contrast[1]))
            if opt.saturation is not None:
                transform_list.append(A.AdjustSaturation(opt.saturation[0], opt.saturation[1]))
            if opt.hue is not None:
                transform_list.append(A.AdjustHue(opt.hue[0], opt.hue[1]))
            if opt.random_scale is not None:
                transform_list.append(A.RandomScale((opt.random_scale[0], opt.random_scale[1])))
            if opt.random_crop is not None:
                transform_list.append(A.RandomCrop((opt.random_crop[0], opt.random_crop[1])))
            if opt.random_flip is not None:
                transform_list.append(A.RandomHorizontallyFlip(p=0.5))
        else:
            transform_list.append(A.Resize((opt.val_size[0], opt.val_size[1])))
        transform_list.append(A.ToTensor(opt.mean, opt.std, opt.use_caffe))
        transform_func = A.Compose(transform_list)
        return transform_func

    def get_image_mask(self, opt):
        dataset_dir = opt.dataset_dir
        image_dir = os.path.join(dataset_dir, opt.image_dir)
        mask_dir = os.path.join(dataset_dir, opt.mask_dir)
        #print("mask dir:", mask_dir)
        image_list = os.listdir(image_dir)
        mask_list = os.listdir(mask_dir)
        image_list.sort()
        mask_list.sort()
        image_list = [os.path.join(image_dir, item) for item in image_list]
        mask_list = [os.path.join(mask_dir, item) for item in mask_list]
        return image_list, mask_list

