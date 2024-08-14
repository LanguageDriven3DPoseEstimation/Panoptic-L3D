import os
from os import path
import datetime
import shutil
import torch
import numpy as np


def flatten_temporal_batch_dims(outputs, targets):
    for k in outputs.keys():
        if k == 'pred_logit' or k == 'text_sentence_feature' or k == 'pred_cls':
            continue
        if isinstance(outputs[k], torch.Tensor):
            outputs[k] = outputs[k].flatten(0, 1)
        else:  # list
            outputs[k] = [i for step_t in outputs[k] for i in step_t]
    targets = [frame_t_target for step_t in targets for frame_t_target in step_t] #[({}, {})]
    return outputs, targets


def create_output_dir(config):
    root = '/root/socpose'
    output_dir_path = path.join(root, 'runs', config.dataset_name, config.version)
    os.makedirs(output_dir_path, exist_ok=True)
    shutil.copyfile(src=config.config_path, dst=path.join(output_dir_path, 'config.yaml'))
    return output_dir_path


def create_checkpoint_dir(output_dir_path):
    checkpoint_dir_path = path.join(output_dir_path, 'checkpoints')
    os.makedirs(checkpoint_dir_path, exist_ok=True)
    return checkpoint_dir_path

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster

def toDevice(sample, device):
    if torch.is_tensor(sample):
        return sample.to(device, non_blocking=True)
    elif isinstance(sample, dict):
        return {k: toDevice(v, device) for k, v in sample.items()}
    elif isinstance(sample, list):
        return [toDevice(s, device) for s in sample]
    elif isinstance(sample, tuple):
        return tuple([toDevice(s, device) for s in sample])
    else:
        return sample

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count