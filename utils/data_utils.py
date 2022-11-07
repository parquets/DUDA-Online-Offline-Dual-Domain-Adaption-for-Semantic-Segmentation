import os
import json
import torch
import torch.nn.functional as F
from torchvision import transforms
from configs.config import config
import numpy as np


def read_annos(dataset_dir, anno_file):
    assert anno_file.endswith('.json')
    with open(anno_file, 'r') as load_f:
        data_dict = json.load(load_f)
    image_list = [item_dict['image_name'] for item_dict in data_dict if item_dict['has_target']]
    label_list = [item_dict['mask_name'] for item_dict in data_dict if item_dict['has_target']]
    image_list = [os.path.join(dataset_dir, item) for item in image_list]
    label_list = [os.path.join(dataset_dir, item) for item in label_list]
    return image_list, label_list

def trans(image, label=None):
    if not config.network.use_caffe:
        image_out = transforms.ToTensor()(image)
        image_out = transforms.Normalize(
            config.network.mean,
            config.network.std
        )(image_out)
    else:
        image = image[:,:,::-1]
        image -= np.array(config.network.mean)
        image = torch.from_numpy(image).float()
        image = image.permute((2,0,1))

    
    label_out = None
    if label is not None:
        label_out = torch.from_numpy(label)
        if len(label.shape) == 2:
            label_out = label_out.long()
    return image_out, label_out



def apply_aug_from_param(logits, param):
    '''
    logits: A tensor predict from the network->(1, num_classes, h, w)
    param: A dict
    '''
    assert(logits, torch.Tensor)

    logits = F.interpolate(logits, size=(1024,2048), mode='bilinear', align_corners=True)

    flip = param['flip']
    s_h, s_w, crop_h, crop_w = param['crop']
    h_size, w_size = param['resize']

    bs, _, _, _ = logits.shape
    logits_list = []

    for i in range(bs):
        flip_i = flip[i]
        s_h_i, s_w_i, crop_h_i, crop_w_i = s_h[i], s_w[i], crop_h[i], crop_w[i]
        h, w = h_size[i], w_size[i]
        logit = logits[i]

        if flip_i:
            logit = torch.flip(logit, dims=[2])
        logit = logit[:,s_h_i:s_h_i+crop_h_i, s_w_i:s_w_i+crop_w_i]
        logit = F.interpolate(logit.unsqueeze(dim=0), size=(int(h),int(w)), mode='bilinear', align_corners=True)
        logits_list.append(logit)
    logits = torch.cat(logits_list, dim=0)
    return logits