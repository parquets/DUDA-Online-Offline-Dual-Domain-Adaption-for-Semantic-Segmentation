import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones import build_backbone
from models.decoders import build_decoder
from models.predictors import build_predictor

class GeneralSegmentor(nn.Module):
    def __init__(self, config, num_classes):
        super(GeneralSegmentor, self).__init__()

        self.backbone = build_backbone(config)
        self.decoder = build_decoder(config, self.backbone.out_channels, num_classes)
        self.predictor = build_predictor(config, self.decoder.out_channels, num_classes)
        '''
        self.image_classfier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.Linear(in_features=2048, out_features=config.dataset.num_classes)
        )
        '''

    def forward(self, data):
        features_dict = self.backbone(data)
        decoder_dict = self.decoder(features_dict)
        # image_classfier_logits = self.image_classfier(features_dict['backbone_layer4'])
        output_dict = self.predictor(decoder_dict, data)
        output_dict.update(features_dict)
        # output_dict.update({'image_classfier_logit':image_classfier_logits})
        return output_dict
    
    def load_pretrained(self, weith_path, gpu_id=0):
        print("resume from:", weith_path)
        state_dict = torch.load(weith_path, map_location="cuda:"+str(gpu_id))
        model_dict = self.state_dict()
        if 'n_averaged' in state_dict:
            del state_dict['n_averaged']
        if "module." in list(state_dict.keys())[0]:
            pretrained_dict = {k[7:]: v for k, v in state_dict.items() if k[7:] in model_dict}
            for k1, k2 in zip(state_dict, model_dict):
                pretrained_dict[k2] = state_dict[k1]
        else:
            pretrained_dict = {}
            for k1, k2 in zip(state_dict, model_dict):
                pretrained_dict[k2] = state_dict[k1]
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def get_params(self, opt):
        params = []
        for key, val in self.named_parameters():
            if val.requires_grad == False:
                continue
            lr = opt.lr
            if 'backbone' in key:
                lr *= 0.1
            params += [{'params':[val], "lr": lr, "initial_lr":lr, "weight_decay":opt.wd}]
        return params