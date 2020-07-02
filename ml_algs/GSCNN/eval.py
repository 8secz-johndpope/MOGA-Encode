"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
Modified by Oscar Andersson
"""

from __future__ import absolute_import
from __future__ import division
import argparse
from functools import partial
from config import cfg, assert_and_infer_cfg
import logging
import math
import os
import sys
import gc

import torch
import numpy as np

from utils.misc import AverageMeter, prep_experiment, evaluate_eval, fast_hist
from utils.f_boundary import eval_mask_boundary
import datasets
import loss
import network
import optimizer

# Argument Parser
parser = argparse.ArgumentParser(description='GSCNN')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--arch', type=str, default='network.gscnn.GSCNN')
parser.add_argument('--dataset', type=str, default='cityscapes')
parser.add_argument('--cv', type=int, default=0,
                    help='cross validation split')
parser.add_argument('--joint_edgeseg_loss', action='store_true', default=True,
                    help='joint loss')
parser.add_argument('--img_wt_loss', action='store_true', default=False,
                    help='per-image class-weighted loss')
parser.add_argument('--batch_weighting', action='store_true', default=False,
                    help='Batch weighting for class')
parser.add_argument('--eval_thresholds', type=str, default='0.0005,0.001875,0.00375,0.005',
                    help='Thresholds for boundary evaluation')
parser.add_argument('--rescale', type=float, default=1.0,
                    help='Rescaled LR Rate')
parser.add_argument('--repoly', type=float, default=1.5,
                    help='Rescaled Poly')

parser.add_argument('--edge_weight', type=float, default=1.0,
                    help='Edge loss weight for joint loss')
parser.add_argument('--seg_weight', type=float, default=1.0,
                    help='Segmentation loss weight for joint loss')
parser.add_argument('--att_weight', type=float, default=1.0,
                    help='Attention loss weight for joint loss')
parser.add_argument('--dual_weight', type=float, default=1.0,
                    help='Dual loss weight for joint loss')

parser.add_argument('--evaluate', action='store_true', default=True)

parser.add_argument("--local_rank", default=0, type=int)

parser.add_argument('--sgd', action='store_true', default=True)
parser.add_argument('--sgd_finetuned',action='store_true',default=False)
parser.add_argument('--adam', action='store_true', default=False)
parser.add_argument('--amsgrad', action='store_true', default=False)

parser.add_argument('--trunk', type=str, default='resnet101',
                    help='trunk model, can be: resnet101 (default), resnet50')
parser.add_argument('--max_epoch', type=int, default=175)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--color_aug', type=float,
                    default=0.25, help='level of color augmentation')
parser.add_argument('--rotate', type=float,
                    default=0, help='rotation')
parser.add_argument('--gblur', action='store_true', default=True)
parser.add_argument('--bblur', action='store_true', default=False) 
parser.add_argument('--lr_schedule', type=str, default='poly',
                    help='name of lr schedule: poly')
parser.add_argument('--poly_exp', type=float, default=1.0,
                    help='polynomial LR exponent')
parser.add_argument('--bs_mult', type=int, default=1)
parser.add_argument('--bs_mult_val', type=int, default=1)
parser.add_argument('--crop_size', type=int, default=720,
                    help='training crop size')
parser.add_argument('--pre_size', type=int, default=None,
                    help='resize image shorter edge to this before augmentation')
parser.add_argument('--scale_min', type=float, default=0.5,
                    help='dynamically scale training images down to this size')
parser.add_argument('--scale_max', type=float, default=2.0,
                    help='dynamically scale training images up to this size')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--restore_optimizer', action='store_true', default=False)
parser.add_argument('--exp', type=str, default='default',
                    help='experiment directory name')
parser.add_argument('--tb_tag', type=str, default='',
                    help='add tag to tb dir')
parser.add_argument('--ckpt', type=str, default='logs/ckpt')
parser.add_argument('--tb_path', type=str, default='logs/tb')
parser.add_argument('--syncbn', action='store_true', default=True,
                    help='Synchronized BN')
parser.add_argument('--dump_augmentation_images', action='store_true', default=False,
                    help='Synchronized BN')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='minimum testing (1 epoch run ) to verify nothing failed')
parser.add_argument('-wb', '--wt_bound', type=float, default=1.0)
parser.add_argument('--maxSkip', type=int, default=0)
args = parser.parse_args()
args.best_record = {'epoch': -1, 'iter': 0, 'val_loss': 1e10, 'acc': 0,
                        'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}

#Enable CUDNN Benchmarking optimization
torch.backends.cudnn.benchmark = True
args.world_size = 1
#Test Mode run two epochs with a few iterations of training and val
if args.test_mode:
    args.max_epoch = 2

if 'WORLD_SIZE' in os.environ:
    args.world_size = int(os.environ['WORLD_SIZE'])
    print("Total world size: ", int(os.environ['WORLD_SIZE']))



#Set up the Arguments, Tensorboard Writer, Dataloader, Loss Fn, Optimizer
assert_and_infer_cfg(args)
writer = prep_experiment(args,parser)

default_eval_epoch = 1


def main(eval_args=None):
    '''
    Main Function

    '''
    # Parse arguments from rest_communication.py
    #args = parser.parse_args(eval_args)
    if args.snapshot == None:
        args.snapshot = "checkpoints/best_cityscapes_checkpoint.pth"
    
    train_loader, val_loader, train_obj = datasets.setup_loaders(args)
    criterion, criterion_val = loss.get_loss(args)
    net = network.get_net(args, criterion)
    net = restore_snapshot(net)
    torch.cuda.empty_cache()

    return evaluate(val_loader, net)


def evaluate(val_loader, net):
    '''
    Runs the evaluation loop and prints F score
    val_loader: Data loader for validation
    net: thet network
    return: 
    '''
    net.eval()
    # 0.0005   13.0 it/sec
    # 0.001875 4.80 it/sec
    # 0.00375  1.70 it/sec
    # 0.005    1.03 it/sec
    thresh = 0.0001

    mf_score1 = AverageMeter()
    mf_pc_score1 = AverageMeter()
    ap_score1 = AverageMeter()
    ap_pc_score1 = AverageMeter()
    IOU_acc = 0
    Fpc = np.zeros((args.dataset_cls.num_classes))
    Fc = np.zeros((args.dataset_cls.num_classes))
    for vi, data in enumerate(val_loader):
        input, mask, edge, img_names = data
        assert len(input.size()) == 4 and len(mask.size()) == 3
        assert input.size()[2:] == mask.size()[1:]
        h, w = mask.size()[1:]

        batch_pixel_size = input.size(0) * input.size(2) * input.size(3)
        input, mask_cuda, edge_cuda = input.cuda(), mask.cuda(), edge.cuda()

        with torch.no_grad():
            seg_out, edge_out = net(input)

        seg_predictions = seg_out.data.max(1)[1].cpu()
        edge_predictions = edge_out.max(1)[0].cpu()

        logging.info('evaluating: %d / %d' % (vi + 1, len(val_loader)))
        '''
        _Fpc, _Fc = eval_mask_boundary(seg_predictions.numpy(), mask.numpy(), args.dataset_cls.num_classes, bound_th=float(thresh))
        Fc += _Fc
        Fpc += _Fpc
        logging.info('F_Score: ' + str(np.sum(Fpc/Fc)/args.dataset_cls.num_classes))
        '''
  
        IOU_acc += fast_hist(seg_predictions.numpy().flatten(), mask.numpy().flatten(),
                             args.dataset_cls.num_classes)

        del seg_out, edge_out, vi, data

    
    acc = np.diag(IOU_acc).sum() / IOU_acc.sum()
    acc_cls = np.diag(IOU_acc) / IOU_acc.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(IOU_acc) / (IOU_acc.sum(axis=1) + IOU_acc.sum(axis=0) - np.diag(IOU_acc))
    freq = IOU_acc.sum(axis=1) / IOU_acc.sum()
    mean_iu = np.nanmean(iu)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    #logging.info('F_Score: ' + str(np.sum(Fpc/Fc)/args.dataset_cls.num_classes))
    #logging.info('F_Score (Classwise): ' + str(Fpc/Fc))
    results ={
        "mean_iu": mean_iu,
        "acc": acc,
        "acc_cls": acc_cls,
        "fwavacc": fwavacc
    } 

    return results


def restore_snapshot(net):
    checkpoint = torch.load(args.snapshot, map_location=torch.device('cpu'))
    logging.info("Load Compelete")
    if 'state_dict' in checkpoint:
        net = forgiving_state_restore(net, checkpoint['state_dict'])
    else:
        net = forgiving_state_restore(net, checkpoint)
    return net


def forgiving_state_restore(net, loaded_dict):
    # Handle partial loading when some tensors don't match up in size.
    # Because we want to use models that were trained off a different
    # number of classes.
    net_state_dict = net.state_dict()
    new_loaded_dict = {}
    for k in net_state_dict:
        if k in loaded_dict and net_state_dict[k].size() == loaded_dict[k].size():
            new_loaded_dict[k] = loaded_dict[k]
        else:
            logging.info('Skipped loading parameter {}'.format(k))
    net_state_dict.update(new_loaded_dict)
    net.load_state_dict(net_state_dict)
    return net


if __name__ == '__main__':
    results = main()
    print(str(results))




