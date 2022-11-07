from cProfile import label
import os
from pyexpat import model
from statistics import mode

from cv2 import threshold
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from configs.config import config
from configs.parse_args import parse_args

args = parse_args()
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast
from models.GeneralSegmentor import GeneralSegmentor
from utils.metric import confusion_matrix, intersectionAndUnionGPU
from utils.display_utils import result_list2dict, print_top, print_iou_list, itv2time, print_loss_dict
from dataset import Cityscapes, Synthia
from pseudo.generate_pseudo_label_cbst import get_threshold_cbst
import tqdm
import cv2
from models.deeplabv2 import ResNet101, Deeplab
'''
python eval.py --cfg ./configs/syn2cityscapes/source_only_res101_multi.yaml
python eval.py --cfg ./configs/syn2cityscapes/adv_train_res101_multi.yaml
python eval.py --cfg ./configs/syn2cityscapes/IAST/self_train_res101.yaml
'''

confusion_matrix = np.zeros((19, 19))

def cityscapes_decoder(pred_label, file_name):
    pred_label[pred_label != 255] += 1
    pred_label[pred_label == 255] = 0
    colormap =[
        [  0,   0,   0], # 0
        [128, 64, 128], # 1
        [244, 35, 232], # 2
        [70, 70, 70], # 3
        [102, 102, 156], # 4
        [190, 153, 153], # 5
        [153, 153, 153], # 6
        [250, 170, 30], # 7
        [220, 220, 0], # 8
        [107, 142, 35], # 9
        [152, 251, 152], # 10
        [70, 130, 180], # 11
        [220, 20, 60], #12
        [255, 0, 0], # 13
        [0, 0, 142], # 14
        [0, 0, 70], # 15
        [0, 60, 100], # 16
        [0, 80, 100], # 17
        [0, 0, 230], # 18
        [119, 11, 32], # 19
    ]
    cm = np.array(colormap).astype('uint8')
    pred_label = cm[pred_label][:, :, ]
    pred_label = pred_label[:,:,::-1]
    cv2.imwrite('./vis/tsne'+file_name, pred_label)


def save_batch_pred(pred_logits, save_path):
    label_pred = pred_logits.max(dim=1)[1]
    label_pred_np = label_pred.data.cpu().numpy()
    bs = label_pred_np.shape[0]
    for i in range(bs):
        label_i = label_pred_np[i]
        label_path = save_path[i]

        label_name = label_path.split('/')[-1]
        # label_path = os.path.join('./data/Cityscapes/pseudo_label_distill_stage1', label_name)
        # cv2.imwrite(label_path, label_i)
        cityscapes_decoder(label_i, label_name)
    # exit(0)

def bgr2rbg(_data):
    if isinstance(_data, torch.Tensor):
        _data = _data.data.cpu().numpy()
    _data = _data[:,::-1,:,:]
    _data = torch.from_numpy(_data.copy()).float()
    return _data

def filter_pseudo_and_save(pred_logits, save_path):
    pred_logits = F.interpolate(pred_logits, (1024,2048), mode='bilinear', align_corners=True)
    pred_scores = F.softmax(pred_logits, dim=1)
    probs, labels = torch.max(pred_scores, dim=1)
    labels_np = labels.data.cpu().numpy()
    probs_np = probs.data.cpu().numpy()
    bs = labels_np.shape[0]
    threshold = np.array(cbst_threshold)

    for i in range(bs):
        label_i = labels_np[i]
        prob_i = probs_np[i]
        label_cls_thresh = threshold[label_i]
        ignore_index = prob_i < label_cls_thresh
        label_path = save_path[i]
        label_i[ignore_index] = 255
        label_name = label_path.split('/')[-1]
        label_name = label_name.replace('gtFine_labelTrainIds.png','leftImg8bit.png')
        label_path = os.path.join('./data/Cityscapes/syn_pseudo_label_0.45', label_name)
        cv2.imwrite(label_path, label_i)

def update_confusion_matrix(max_label, sed_label, ratio_matrix, threshold=0.5):
    index = ratio_matrix > threshold
    max_label_sel = max_label[index]
    sed_label_sel = sed_label[index]
    confusion_matrix[max_label_sel][sed_label_sel] += 1

def get_prob_label(pred_scores_np):
    '''
    pred_scores_np [bs, num_classes, h, w]
    '''
    argsort_pred = np.argsort(pred_scores_np, axis=1)
    sort_pred = np.sort(pred_scores_np, axis=1)
    argsort_pred = argsort_pred[:,::-1,:,:]
    sort_pred = sort_pred[:,::-1,:,:]
    max_pred_label = argsort_pred[:,0,:,:]
    max_pred_prob = sort_pred[:,0,:,:]
    sed_pred_label = argsort_pred[:,1,:,:]
    sed_pred_prob = sort_pred[:,1,:,:]
    return (max_pred_prob, max_pred_label), (sed_pred_prob, sed_pred_label)

def sed_ratio_max(sed_prob, max_prob):
    return sed_prob/max_prob

def eval_model(model, eval_loader):
    feature_list = []
    label_list = []
    model.eval()
    n_class = config.dataset.num_classes
    intersection_sum = 0
    union_sum = 0
    pbar = tqdm.tqdm(total=len(eval_loader))
    with autocast():
        with torch.no_grad():
            for i, eval_data_dict in enumerate(eval_loader):
                pbar.update(1)
                if i== 1:
                    break
                images, labels = eval_data_dict['image'], eval_data_dict['label']
                # images = bgr2rbg(images)/255
                images, labels = images.cuda(), labels.cuda()
                im_path = eval_data_dict['name']

                pred = model(images)
                logits = pred['decoder_logits']
                

                bs, ch, h, w = pred['backbone_layer4'].size()
                if h != 69 and w != 129:
                    pred['backbone_layer4'] = F.interpolate(pred['backbone_layer4'], size=(69,129), mode='bilinear')
                features = pred['backbone_layer4'].data.cpu().numpy()

                bs, ch, h, w = features.shape
                # print(labels.shape)
                ref_label = F.interpolate(labels.unsqueeze(1).float(), (h, w), mode='nearest').long().data.cpu().numpy()
                save_batch_pred(logits, im_path)
                feature_list.append(features)
                label_list.append(ref_label)
                flip_logits = model(torch.flip(images, [3]))['decoder_logits']
                flip_logits = torch.flip(flip_logits, [3])
                # logits = mid_logits+mid_flip_logits+large_logits+large_flip_logits
                logits = logits+flip_logits
                label_pred = logits.max(dim=1)[1]
                intersection, union = intersectionAndUnionGPU(label_pred, labels, n_class)
                intersection_sum += intersection
                union_sum += union

            pbar.close()
            intersection_sum = intersection_sum.cpu().numpy()
            union_sum = union_sum.cpu().numpy()
            iu = intersection_sum / (union_sum + 1e-10)
            mean_iu = np.mean(iu)

            result_item = {}
            result_item.update({'iou': mean_iu})
            result_item.update(result_list2dict(iu,'iou'))
            report = 'val_miou: {:.6f}'.format(mean_iu) + print_iou_list(iu)
            print(report)
            
    features_array = np.concatenate(feature_list, axis=0)
    label_array = np.concatenate(label_list, axis=0)

    return mean_iu

def eval_fusion(model_A, model_B, eval_loader):
    model_A.eval()
    model_B.eval()
    n_class = config.dataset.num_classes
    intersection_sum = 0
    union_sum = 0
    pbar = tqdm.tqdm(total=len(eval_loader))
    with torch.no_grad():
        for i, eval_data_dict in enumerate(eval_loader):
            pbar.update(1)
            images, labels = eval_data_dict['image'], eval_data_dict['label']
            im_path = eval_data_dict['name']
            images, labels = images.cuda(), labels.cuda()
            pos_forward = model_A(images)
            logits_A = pos_forward['decoder_logits']
            features = logits_A['backbone_layer4']
            flip_logits_A = model_A(torch.flip(images, [3]))['decoder_logits']
            logits_B = model_B(images)['decoder_logits']
            flip_logits_B = model_B(torch.flip(images, [3]))['decoder_logits']
            flip_logits_A = torch.flip(flip_logits_A, [3])
            flip_logits_B = torch.flip(flip_logits_B, [3])
            # label_pred = (F.softmax(logits_A, dim=1)+F.softmax(logits_B, dim=1))/2
            label_pred = (logits_A+flip_logits_A)/2+(logits_B+flip_logits_B)/2
            # save_batch_pred(label_pred, im_path)
            # filter_pseudo_and_save(label_pred, im_path)
            label_pred = label_pred.max(dim=1)[1]
            intersection, union = intersectionAndUnionGPU(label_pred, labels, n_class)
            intersection_sum += intersection
            union_sum += union

        pbar.close()
        intersection_sum = intersection_sum.cpu().numpy()
        union_sum = union_sum.cpu().numpy()
        iu = intersection_sum / (union_sum + 1e-10)
        mean_iu = np.mean(iu)

        result_item = {}
        result_item.update({'iou': mean_iu})
        result_item.update(result_list2dict(iu,'iou'))
        report = 'val_miou: {:.6f}'.format(mean_iu) + print_iou_list(iu)
        print(report)
    
    return mean_iu


def eval_dir_model(model, eval_loader, dir_path):
    files = os.listdir(dir_path)
    pth_files = [os.path.join(dir_path,f) for f in files if f.endswith('.pth')]
    pth_files.sort()
    print("find:", len(pth_files), "weight files")
    mean_iou_list = []
    for weight_file in pth_files:
        print("load weight:", weight_file)
        model.load_pretrained(torch.load(weight_file))
        mean_iou = eval_model(model, eval_loader)
        mean_iou_list.append(mean_iou)
    print("max_miou:", max(mean_iou_list))


def main():

    weight_A = './checkpoints/self_train/gtav2cityscapes/stage1/MFA/last_iter_0402.pth'

    model_A = GeneralSegmentor(config, config.dataset.num_classes).cuda()
    model_A.load_pretrained(weight_A)
    # model_B.load_pretrained(weight_B)
    #model_B.load_pretrained(torch.load(weight_B))
    target_val_set = Cityscapes(config, 'val', 'target')

    target_val_loader = DataLoader(dataset=target_val_set, batch_size=6, 
                                    num_workers=0, shuffle=False)

    eval_model(model_A, target_val_loader)


if __name__ == '__main__':
    main()
