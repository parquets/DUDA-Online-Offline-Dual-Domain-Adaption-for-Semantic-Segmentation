import math
import numbers
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tf
import albumentations as A
from PIL import Image, ImageOps



class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, image, mask, params):
        assert image.size == mask.size
        # print("input size:",image.size)
        w, h = image.size
        th, tw = self.size
        if w == tw and h == th:
            return image, mask
 
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        #print("crop x1:",x1,"crop y1:",y1)

        croped_image = image.crop((x1, y1, x1 + tw, y1 + th))
        croped_mask = mask.crop((x1, y1, x1 + tw, y1 + th))
        params['RandomCrop'] = (y1, y1 + th, x1, x1 + tw)
        # print("croped size:", croped_image.size)
        return croped_image, croped_mask, params


class RandomHorizontallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, image, mask, params):
        if random.random() < self.p:
            params['RandomHorizontallyFlip'] = True
            fliped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
            fliped_mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            return fliped_image, fliped_mask, params
        else:
            params['RandomHorizontallyFlip'] = False
        return image, mask, params


class RandomScale(object):
    def __init__(self, scale_limit) -> None:
        self.min_scale = scale_limit[0]
        self.max_scale = scale_limit[1]

    def __call__(self, image, mask, params):
        # print("image size:", image.size)
        w,h = image.size
        scale = random.uniform(self.min_scale, self.max_scale)
        new_w, new_h = int(scale*w), int(scale*h)
        # print("resize w:", new_w, "resize h",new_h)
        scaled_image = image.resize((new_w, new_h),Image.ANTIALIAS)
        scaled_mask = mask.resize((new_w, new_h), Image.NEAREST)
        params['RandomScale'] = (new_w,new_h)
        return scaled_image, scaled_mask, params


class Resize(object):
    def __init__(self, size) -> None:
        self.h = size[0]
        self.w = size[1]

    def __call__(self, image, mask=None, params=None):
        resized_image = image.resize((self.w, self.h), Image.ANTIALIAS)
        resized_mask = None
        if mask is not None:
            resized_mask = mask.resize((self.w,self.h), Image.NEAREST)
        if params is not None:
            params['Resize'] = (self.h, self.w)
        return resized_image, resized_mask, params


class AdjustGamma(object):
    def __init__(self, gamma, p=0.2):
        self.gamma = gamma
        self.p = p

    def __call__(self, image, mask, params):
        assert image.size == mask.size
        if random.random() < self.p:
            image = tf.adjust_gamma(image, random.uniform(1-self.gamma, 1 + self.gamma))
        return image, mask, params


class AdjustSaturation(object):
    def __init__(self, saturation, p=0.2):
        self.saturation = saturation
        self.p = p

    def __call__(self, image, mask, params):
        assert image.size == mask.size
        if random.random() < self.p:
            image = tf.adjust_saturation(image, random.uniform(1 - self.saturation, 1 + self.saturation))
        return image, mask, params


class AdjustHue(object):
    def __init__(self, hue, p=0.2):
        self.hue = hue
        self.p = p

    def __call__(self, image, mask, params):
        assert image.size == mask.size
        if random.random() < self.p:
            image = tf.adjust_hue(image, random.uniform(-self.hue, self.hue))
        return image, mask, params

class AdjustBrightness(object):
    def __init__(self, bf, p=0.2):
        self.bf = bf
        self.p = p

    def __call__(self, image, mask, params):
        assert image.size == mask.size
        if random.random() < self.p:
            image = tf.adjust_brightness(image, random.uniform(1 - self.bf, 1 + self.bf))
        return image, mask, params

class AdjustContrast(object):
    def __init__(self, cf, p=0.2):
        self.cf = cf
        self.p = p

    def __call__(self, image, mask, params):
        assert image.size == mask.size
        if random.random() < self.p:
            image = tf.adjust_contrast(image, random.uniform(1 - self.cf, 1 + self.cf))
        return image, mask, params


class ToTensor(object):
    def __init__(self, mean, std, use_caffe=False):
        self.mean = mean
        self.std = std
        self.use_caffe = use_caffe
    
    def __call__(self, image, mask=None, params=None):
        if isinstance(image, Image.Image):
            image = np.array(image).astype(np.float32)
        if not self.use_caffe:
            image /= 255
        else:
            image = image[:,:,::-1].copy()
        image -= self.mean
        image /= self.std
        image = torch.from_numpy(image).float().permute((2,0,1))
        if mask is not None:
            if isinstance(mask, Image.Image):
                mask = np.array(mask)
            mask = torch.from_numpy(mask).long()
        return image, mask, params

class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.PIL2Numpy = False

    def __call__(self, image, mask, full_image=None):
        params = {}
        
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.uint8(image))
        if (full_image is not None) and (not isinstance(full_image, Image.Image)):
            full_image = Image.fromarray(np.uint8(full_image))
        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(np.uint8(mask))
        for a in self.augmentations:
            if isinstance(a, ToTensor) and (full_image is not None):
                full_image, _, _ = a(full_image)
            image, mask, params = a(image, mask, params)
        
        # image, mask = np.array(image), np.array(mask)
        output_dict = {
            'image': image,
            'label': mask,
            'param': params,
        }
        if full_image is not None:
            output_dict.update({'full_image': full_image})
        return output_dict