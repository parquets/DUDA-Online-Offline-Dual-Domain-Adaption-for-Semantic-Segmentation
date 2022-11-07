import os
from numpy.lib.index_tricks import AxisConcatenator
from numpy.lib.twodim_base import eye
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import json


def get_confident(pred_probs):
    return np.max(pred_probs, axis=1)

def get_entropy(pred_probs):
    entropy = pred_probs*np.log(pred_probs+1e-8)
    entropy = -1*np.sum(entropy, axis=1)
    return entropy

def get_distance(pred_probs):
    bs,ch, h,w = pred_probs.shape
    sorted_probs = np.sort(pred_probs, axis=1)
    max_probs = sorted_probs[:,-1,:,:]
    sed_probs = sorted_probs[:,-2,:,:]
    dis = max_probs-sed_probs
    dis = 1 - dis
    return dis

def get_input(pred_probs):
    pred_probs = pred_probs.data.cpu().numpy()
    conf_input = get_confident(pred_probs)
    ent_input = get_entropy(pred_probs)
    dis_input = get_distance(pred_probs)
    conf_input = np.expand_dims(conf_input, axis=1)
    ent_input = np.expand_dims(ent_input, axis=1)
    dis_input = np.expand_dims(dis_input, axis=1)
    input = np.concatenate([conf_input, ent_input, dis_input], axis=1)
    return input


def generate_error_label(config, model, loader, input_save_dir=None, mask_save_dir=None):
    model.eval()
    for i, data_dict in enumerate(loader):
        print("solving:",i,"th image")
        image = data_dict['image'].cuda()
        pred_logits = model(image)['primary_decoder_logits']
        inputs = get_input(F.softmax(pred_logits, dim=1))
        pred_logits = pred_logits.data.cpu().numpy()
        pred_label = np.argmax(pred_logits, axis=1)
        label = data_dict['label'].data.cpu().numpy()
        mask = label != pred_label
        mask = mask.astype(np.uint8)
        names = data_dict['name']
        for i_, name in enumerate(names):
            file_name = name.split('/')[-1]
            file_name = file_name.replace('.png', '_error.png')
            save_path = os.path.join(mask_save_dir, file_name)
            cv2.imwrite(save_path, mask[i_])
            file_name = file_name.replace('.png', '.npy')
            save_path = os.path.join(input_save_dir, file_name)
            np.save(save_path, inputs[i_])


def split():
    pass