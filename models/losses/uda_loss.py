from cv2 import threshold
import torch
import torch.nn as nn
import torch.nn.functional as F
from .seg_loss import CrossEntropyLoss2d, SymmetricCrossEntropyLoss2d

class MaxSquareLoss(nn.Module):
    def __init__(self, ignore_index= -1, num_class=19):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_class = num_class
    
    def forward(self, pred_label, pred_prob):
        """
        :param pred: predictions (N, 1, H, W)
        :param prob: probability of pred (N, C, H, W)
        :return: maximum squares loss
        """
        # prob -= 0.5
        mask = (pred_label != self.ignore_index)    
        loss = -torch.sum(torch.sum(torch.pow(pred_prob, 2),dim=1)*mask)/(2*mask.sum())
        return loss

class ImageWeightedMaxSquareloss(nn.Module):
    def __init__(self, ignore_label= -1, num_class=19, ratio=0.2):
        super().__init__()
        self.ignore_label = ignore_label
        self.num_class = num_class
        self.ratio = ratio
    
    def forward(self, pred, prob, label=None):
        """
        :param pred: predictions (N, C, H, W)
        :param prob: probability of pred (N, C, H, W)
        :param label(optional): the map for counting label numbers (N, C, H, W)
        :return: maximum squares loss with image-wise weighting factor
        """
        # prob -= 0.5
        N, C, H, W = prob.size()
        mask = (prob != self.ignore_label)
        maxpred, argpred = torch.max(prob, 1)
        mask_arg = (maxpred != self.ignore_label)
        argpred = torch.where(mask_arg, argpred, torch.ones(1).to(prob.device, dtype=torch.long)*self.ignore_label)
        if label is None:
            label = argpred
        weights = []
        batch_size = prob.size(0)
        for i in range(batch_size):
            hist = torch.histc(label[i].cpu().data.float(), 
                            bins=self.num_class+1, min=-1,
                            max=self.num_class-1).float()
            hist = hist[1:]
            weight = (1/torch.max(torch.pow(hist, self.ratio)*torch.pow(hist.sum(), 1-self.ratio), torch.ones(1))).to(argpred.device)[argpred[i]].detach()
            weights.append(weight)
        weights = torch.stack(weights, dim=0)
        mask = mask_arg.unsqueeze(1).expand_as(prob)
        loss = -torch.sum((torch.pow(prob, 2)*weights)[mask]) / (batch_size*self.num_class)
        return loss

class ConfDiffTreatLoss(nn.Module):
    def __init__(self, threshold=0.9, ignore_label=255, num_classes=19):
        super().__init__()
        self.ignore_index = ignore_label
        self.num_classes = num_classes
        self.threshold = threshold
        self.ce_loss_fn = CrossEntropyLoss2d(ignore_label=ignore_label)
        self.sce_loss_fn = SymmetricCrossEntropyLoss2d(ignore_index=ignore_label, num_classes=num_classes)

    def forward(self, pred, label):
        '''
        pred: [bs, num_classes, h, w], logits without softmax
        label: [bs, h, w] index of each classes
        '''
        pred_prob, pred_label = torch.max(F.softmax(pred.detach(), dim=1), dim=1)
        ce_label = label[pred_prob <= self.threshold] = self.ignore_index
        ce_loss = self.ce_loss_fn(pred, ce_label)
        sce_label = label[pred_prob > self.threshold] = self.ignore_index
        sce_loss = self.sce_loss_fn(pred, sce_label)
        loss = ce_loss + sce_loss
        self.update_threshold()
        return loss

    def update_threshold(self):
        x = self.threshold - 0.000005
        self.threshold = max(self.threshold, 0.6)

class MultiLabelClassfierLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.bce_loss = nn.BCELoss(weight=None, size_average=True)
    def forward(self, input, target):
        '''
        input: [bs, num_classes]
        target: [bs, num_classes]
        '''
        prob = torch.sigmoid(input)
        loss = self.bce_loss(prob, target.float())
        return loss


class MaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input, target):
        '''
        input: [bs, 2, h, w]
        target: [bs,h,w]
        '''
        return F.cross_entropy(input, target)

def generate_neg_label(pred_labels):
    neg_label = None
    return neg_label

def negative_loss(pred_logits, pred_labels, mask):
    pred_scores = F.softmax(pred_logits, dim=1)
    neg_label = pred_labels
    neg_label_onehot = F.one_hot(neg_label, 19).permute(0,3,1,2).detach()
    weight_map = torch.max(pred_scores*neg_label_onehot, dim=1)[0].detach()
    loss = -torch.sum(mask*weight_map*torch.sum(torch.log(1-pred_scores+1e-10)*neg_label_onehot, dim=1))/(mask.sum())
    return loss