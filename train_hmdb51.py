import os
import logging
from argparse import ArgumentParser
from collections import OrderedDict

from torch.nn.parallel import DataParallel
from mmcv import Config

from .train_common import (
    dali_batch_processor, train_network
)
from .torchdcvd.dataloader import VideoLoader
from .torchdcvd.models import MultiFiberNet3d


def get_root_logger(log_level=logging.INFO):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s', level=log_level)
    logger = logging.getLogger()

    return logger


def parse_args():
    parser = ArgumentParser(description='Train HMDB51 classification')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--pretrained_file', required=True, help='pretraining file')
    parser.add_argument('--distributed', default=False, action='store_true', 
        help='whether to launch distributed training')
    parser.add_argument('--local_rank', type=int, default=0)

    return parser.parse_args()


def build_dataloader(file_list):
    dataloader = VideoLoader(file_list, 
                            batch_size=2,
                            sequence_length=16,
                            crop_size=(224,224),
                            random_shuffle=False)
    return dataloader


def load_multifiber_net(model, pretrained_file, saved_with_parallel=True):
    if saved_with_parallel:
        model = DataParallel(model)

    model_dict = model.state_dict()
    
    pretrained_dict = torch.load(pretrained_file)['state_dict']
    pretrained_dict = {k : v for k, v in pretrained_dict.items()
                       if k in model_dict}
    pretrained_dict['module.bn_1.bn.weight'] = pretrained_dict['module.tail.bn.weight']

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    if saved_with_parallel:
        model = model.module

    return model


def main():
    # setup and configs
    args = parse_args()
    cfg = Config.fromfile(args.config)
    logger = get_logger(cfg.log_level)

    # init distributed environment if necessary
    if args.distributed == True:
        init_dist(**cfg.dist_params)
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        if rank != 0:
            logger.setLevel('ERROR')
        logger.info('Enabled distributed training.')

    model = MultiFiberNet3d(num_classes=51)
    model = load_multifiber_net(model, args.pretrained_file)
