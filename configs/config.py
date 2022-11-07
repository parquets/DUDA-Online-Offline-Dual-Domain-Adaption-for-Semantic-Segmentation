import yaml
import numpy as np
from easydict import EasyDict as edict

config = edict()
config.use_ema = False
config.use_distill = False
config.adv_train = False
config.stage = None
config.experiment_name = ''
config.model_name = ''
config.network = edict()
config.network.backbone_type = "ResNet50"
config.network.backbone_fix_bn = True
config.network.backbone_with_ibn = False
config.network.backbone_layers = [3,4,6,3]
config.network.backbone_strides = [1,2,1,1]
config.network.backbone_dilations = [1,1,2,4]
config.network.backbone_pretrain = True
config.network.deocoder_type = None
config.network.decoder_fix_bn = True
config.network.predictor_type = None
config.network.discriminator = None
config.network.dropout_p = 0.5

config.input = edict()
config.input.use_source_data = False
config.input.resize = None
config.input.random_scale = None
config.input.random_crop = None
config.input.random_flip = None
config.input.val_size = [768, 1536]
config.input.mean = [0.485, 0.456, 0.406]
config.input.std = [0.229, 0.224, 0.225]
config.input.use_caffe = False

config.source_input = edict()
config.source_input.resize = None
config.source_input.gamma = None
config.source_input.brightness = None
config.source_input.contrast = None
config.source_input.saturation = None
config.source_input.hue = None
config.source_input.random_scale = None
config.source_input.random_crop = None
config.source_input.random_flip = None
config.source_input.mean = [0.485, 0.456, 0.406]
config.source_input.std = [0.229, 0.224, 0.225]
config.source_input.use_caffe = False

config.target_input = edict()
config.target_input.resize = None
config.target_input.gamma = None
config.target_input.brightness = None
config.target_input.contrast = None
config.target_input.saturation = None
config.target_input.hue = None
config.target_input.random_scale = None
config.target_input.random_crop = None
config.target_input.random_flip = None
config.target_input.val_size = [768, 1536]
config.target_input.mean = [0.485, 0.456, 0.406]
config.target_input.std = [0.229, 0.224, 0.225]
config.target_input.use_caffe = False

config.loader = edict()
config.loader.bs = 2
config.loader.nw = 2

config.source_loader = edict()
config.source_loader.bs = 2
config.source_loader.nw = 2

config.target_loader = edict()
config.target_loader.bs = 2
config.target_loader.nw = 2

config.val_loader = edict()
config.val_loader.bs = 2
config.val_loader.num_workers = 2

config.train = edict()
config.train.lr_scheduler = 'poly'
config.train.optimizer = 'SGD'
config.train.lr = 2e-4
config.train.wd = 0
config.train.use_source = True
config.train.resume_from = ''
config.train.begin_iteration = 0
config.train.max_iteration = 90000
config.train.warmup_iteration = 2000
config.train.scheduler_gamma = 0.1
config.train.max_epoch = -1
config.train.display_iteration = 50
config.train.eval_iteration = 1000
config.train.source_loss = "CE"
config.train.target_loss = "CE"

config.dataset = edict()
config.dataset.num_classes = 19

config.source_dataset = edict()
config.source_dataset.dataset_type = None
config.source_dataset.dataset_dir = ''
config.source_dataset.image_dir = ''
config.source_dataset.mask_dir = ''

config.target_dataset = edict()
config.target_dataset.dataset_type = None
config.target_dataset.dataset_dir = ''
config.target_dataset.image_dir = ''
config.target_dataset.mask_dir = ''

config.val_dataset = edict()
config.val_dataset.dataset_type = None
config.val_dataset.dataset_dir = ''
config.val_dataset.image_dir = ''
config.val_dataset.mask_dir = ''

config.pseudo = edict()
config.pseudo.pseudo_strategy = None
config.pseudo.iast_alpha = 1
config.pseudo.iast_beta = 1
config.pseudo.iast_gamma = 1
config.pseudo.proportion = 0.2
config.pseudo.online = False
config.pseudo.offline = True

config.pseudo_dataset = edict()
config.pseudo_dataset.dataset_type = None
config.pseudo_dataset.dataset_dir = ''
config.pseudo_dataset.train_anno_file = ''

config.logger = edict()
config.logger.train_log_save_dir = ''
config.logger.test_log_save_dir = ''

config.save = edict()
config.save.save_dir = ''

config.distribution = edict()
config.distribution.num_gpus = 2
config.distribution.num_nodes = 1
config.distribution.rank_within_nodes = 0
config.distribution.world_size = config.distribution.num_gpus*config.distribution.num_nodes
config.distribution.port = '12355'

def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.safe_load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    if k == 'train':
                        if 'bbox_weights' in v:
                            v['bbox_weights'] = np.array(v['bbox_weights'])
                    elif k == 'network':
                        if 'pixel_means' in v:
                            v['pixel_means'] = np.array(v['pixel_means'])
                    for vk, vv in v.items():
                        config[k][vk] = vv
                else:
                    config[k] = v
            else:
                config[k] = v
