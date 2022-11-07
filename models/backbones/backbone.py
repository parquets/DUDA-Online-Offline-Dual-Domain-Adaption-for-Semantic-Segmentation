import torch
import torch.nn as nn

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

    def load_pretrained(self, input_state_dict):
        pass

    def fix_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                for p in m.parameters():
                    p.requires_grad = False
    