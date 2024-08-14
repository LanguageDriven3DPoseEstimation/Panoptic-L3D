import torch

from datasets.PanopticL3DStage1 import PanopticL3DStage1
from datasets.PanopticL3DStage2 import PanopticL3DStage2


def build_dataset(image_set, dataset_file, cfg):
    if dataset_file == 'panoptic-L3D':
        if cfg.TRAINING_STAGE == 'stage1':
            return PanopticL3DStage1(cfg, image_set)
        elif cfg.TRAINING_STAGE == 'stage2' or cfg.RUNNING_MODE == 'test' or cfg.TRAINING_STAGE == 'stage3':
            return PanopticL3DStage2(cfg, image_set)
        else:
            raise ValueError('Unknown training stage {}'.format(cfg.TRAINING_STAGE))
    else:
        raise ValueError('Unknown dataset {}'.format(dataset_file))
