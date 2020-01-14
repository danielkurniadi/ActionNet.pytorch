import os
import logging
from collections import OrderedDict

from mmcv.runner import Runner
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def parse_loss_and_acc(loss, acc_top1, acc_top5):
    log_vars = OrderedDict()
    log_vars['loss'] = loss.item()
    log_vars['acc_top1'] = acc_top1.item()
    log_vars['acc_top5'] = acc_top5.item()

    return dict(loss=loss, log_vars=log_vars, num_samples=img.size(0))


def dali_batch_processor(model, sequence, train_mode=True):
    data, labels = sequence[-1]['data'], sequence[-1]['labels']
    data, labels = data.cuda(), target2onehot(labels).cuda()
    
    preds = model(data)
    loss = F.cross_entropy(preds, labels)

    acc_top1, acc_top5 = accuracy(preds, labels, topk=(1,5))
    outputs = parse_loss_and_acc(loss, acc_top1, acc_top5)

    return outputs


def train_network(model, dataloaders, cfg, distributed=False):
    # Start training
    if distributed:
        _dist_train(model, dataloaders, cfg)
    else:
        _non_dist_train(model, dataloaders, cfg)


def _dist_train(model, dataloaders, cfg):
    # setup model for distributed computing on multiple device
    model = MMDistributedDataParallel(model.cuda())

    # build runner
    runner = Runner(model, dali_batch_processor, cfg.optimizer, cfg.work_dir,
                    cfg.log_level)

    # register hooks
    optimizer_config = DistOptimizerHook(**cfg.optimizer_config)
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    # resume from epoch and/or load model checkpoint when available
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    # run training session
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def _non_dist_train(model, dataloaders, cfg):
    # setup model for data parallel across gpus
    model = MMDataParallel(model.cuda())

    # build runner
    runner = Runner(model, dali_batch_processor, cfg.optimizer, cfg.work_dir,
                    cfg.log_level)

    # register hooks
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    # register eval hooks
    runner.register_hook()

    # resume from epoch and/or load model checkpoint when available
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    # run training session
    runner.run(dataloaders, cfg.workflow, cfg.total_epochs)


