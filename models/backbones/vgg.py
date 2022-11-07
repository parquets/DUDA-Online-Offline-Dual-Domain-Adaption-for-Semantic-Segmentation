import torch
import torch.nn as nn
from .backbone import Backbone

class VGG(Backbone):
    def __init__(self, depth=16):
        super().__init__()

    def forward(self, data):
        pass
