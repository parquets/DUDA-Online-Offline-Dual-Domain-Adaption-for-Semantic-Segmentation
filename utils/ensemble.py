import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def ema_model_update(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1. - alpha, param.data)
    
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.data = buffer.data.clone()