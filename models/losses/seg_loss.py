import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_index

    def compute_entropy(self, fcn_pred_prob):
        entropy = fcn_pred_prob*torch.log(fcn_pred_prob+1e-30)
        entropy = -torch.sum(entropy, dim=1)
        entropy = torch.mean(entropy)
        return entropy

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(2))

        loss = F.cross_entropy(predict, target, weight=weight, ignore_index=self.ignore_label)
        return loss



class SymmetricCrossEntropyLoss2d(nn.Module):
    def __init__(self, alpha=0.2, beta=1, num_classes=19, size_average=True, ignore_index=255):
        super(SymmetricCrossEntropyLoss2d, self).__init__()
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.size_average = size_average
        self.ignore_index = ignore_index

    def rce(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        mask = (labels != 255).float()
        labels[labels==255] = self.num_classes
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes + 1).float()
        label_one_hot = torch.clamp(label_one_hot.permute(0,3,1,2)[:,:-1,:,:], min=1e-4, max=1.0).to(labels.device)
        rce = -(torch.sum(pred * torch.log(label_one_hot), dim=1) * mask).sum() / (mask.sum() + 1e-6)
        return rce

    def forward(self, predict, target, weight=None):
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(2))
 
        ce = F.cross_entropy(predict, target, weight=weight, ignore_index=self.ignore_index)
        rce = self.rce(predict, target.clone())
        loss = self.alpha*ce + self.beta*rce
        return loss


class FocalLoss2d(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss
