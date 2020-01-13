import os
import logging
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from mmcv import Config
from mmcv.runner import DistSamplerSeedHook, Runner


def parse_args():
    pass


def get_root_logger(log_level=logging.INFO):
    pass


def train_network(model, dataloaders, logger, distributed=False):
    # Start training
    if distributed:
        _dist_train(model, dataloaders, logger)
    else:
        _non_dist_train(model, dataloaders, logger)


def _dist_train(model, dataloaders, logger):
    # setup model for distributed computing
    model = 


