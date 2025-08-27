# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
import datetime
import time
import math
import json
import numpy as np
import utils_ibot
import models
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from tensorboardX import SummaryWriter
from models.head import iBOTHead
from loader import ImageFolderMask
from evaluation.unsupervised.unsup_cls import eval_pred
from loader_vidor_odis import DataAugmentationSingleObject
from loader_wtvenice_odis import ODISWTVeniceFrameDataset
import wandb
import utils


def get_args_parser():
    parser = argparse.ArgumentParser("iBOT", add_help=False)

    # Model parameters
    parser.add_argument(
        "--arch",
        default="vit_small",
        type=str,
        choices=[
            "vit_tiny",
            "vit_small",
            "vit_base",
            "vit_large",
            "deit_tiny",
            "deit_small",
            "swin_tiny",
            "swin_small",
            "swin_base",
            "swin_large",
        ],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""",
    )
    parser.add_argument(
        "--patch_size",
        default=16,
        type=int,
        help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""",
    )
    parser.add_argument(
        "--num_object_tokens", default=4, type=int, help="""Object tokens per view."""
    )
    parser.add_argument(
        "--single_embed_obj_pos",
        default=False,
        type=utils_ibot.bool_flag,
        help="""(Default: False)""",
    )
    parser.add_argument(
        "--object_sampling_strategy",
        default="random_area",
        type=str,
        choices=["random_area", "random_area_w_repl", "random", "largest", "smallest"],
        help="""Object sampling strategy.""",
    )
    parser.add_argument(
        "--patch_obj_assignment_strategy",
        default="one_to_many_BB",
        type=str,
        choices=["one_to_one", "one_to_many", "one_to_many_BB"],
        help="""Patch-object assignment strategy.""",
    )
    parser.add_argument(
        "--obj_loss_only_global",
        default=False,
        type=utils_ibot.bool_flag,
        help="""(Default: False)""",
    )
    parser.add_argument(
        "--use_different_obj_tokens",
        default=False,
        type=utils_ibot.bool_flag,
        help="""(Default: False)""",
    )
    parser.add_argument(
        "--obj_divide_to_instances",
        default=False,
        type=utils_ibot.bool_flag,
        help="""(Default: False)""",
    )
    parser.add_argument(
        "--obj_aware_lc_loader",
        default=True,
        type=utils_ibot.bool_flag,
        help="""(Default: False)""",
    )
    parser.add_argument(
        "--separate_obj_attn",
        default=False,
        type=utils_ibot.bool_flag,
        help="""(Default: False)""",
    )
    parser.add_argument(
        "--min_obj_area",
        default=10,
        type=int,
        help="""Minimum object area in the image.""",
    )
    parser.add_argument(
        "--bb_margin",
        default=0,
        type=int,
        help="""Number of margin patches around the bounding box.""",
    )
    parser.add_argument(
        "--bb_margin_strategy",
        default="random",
        type=str,
        choices=["fixed", "random"],
        help="""Bounding box margin sampling strategy.""",
    )
    parser.add_argument(
        "--window_size",
        default=7,
        type=int,
        help="""Size of window - default 7.
        This config is only valid for Swin Transofmer and is ignoired for vanilla ViT architectures.""",
    )
    parser.add_argument("--obj_pred_offset", default=0.0, type=float, help=""".""")
    parser.add_argument(
        "--out_dim",
        default=8192,
        type=int,
        help="""Dimensionality of
        output for [CLS] token.""",
    )
    parser.add_argument(
        "--patch_out_dim",
        default=8192,
        type=int,
        help="""Dimensionality of
        output for patch tokens.""",
    )
    parser.add_argument(
        "--shared_head",
        default=False,
        type=utils_ibot.bool_flag,
        help="""Whether to share 
        the same head for [CLS] token output and patch tokens output. When set to false, patch_out_dim
        is ignored and enforced to be same with out_dim. (Default: False)""",
    )
    parser.add_argument(
        "--shared_head_teacher",
        default=True,
        type=utils_ibot.bool_flag,
        help="""See above.
        Only works for teacher model. (Defeault: True)""",
    )
    parser.add_argument(
        "--norm_last_layer",
        default=True,
        type=utils_ibot.bool_flag,
        help="""Whether or not to weight normalize the last layer of the head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""",
    )
    parser.add_argument(
        "--momentum_teacher",
        default=0.996,
        type=float,
        help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""",
    )
    parser.add_argument(
        "--norm_in_head",
        default=None,
        help="Whether to use batch normalizations in projection head (Default: None)",
    )
    parser.add_argument(
        "--act_in_head",
        default="gelu",
        help="Whether to use batch normalizations in projection head (Default: gelu)",
    )
    parser.add_argument(
        "--use_masked_im_modeling",
        default=True,
        type=utils_ibot.bool_flag,
        help="Whether to use masked image modeling (mim) in backbone (Default: True)",
    )
    parser.add_argument(
        "--pred_ratio",
        default=0.3,
        type=float,
        nargs="+",
        help="""Ratio of partial prediction.
        If a list of ratio is specified, one of them will be randomly choosed for each patch.""",
    )
    parser.add_argument(
        "--pred_ratio_var",
        default=0,
        type=float,
        nargs="+",
        help="""Variance of partial prediction
        ratio. Length should be indentical to the length of pred_ratio. 0 for disabling. """,
    )
    parser.add_argument(
        "--pred_shape",
        default="block",
        type=str,
        help="""Shape of partial prediction.""",
    )
    parser.add_argument(
        "--pred_start_epoch",
        default=0,
        type=int,
        help="""Start epoch to perform masked
        image prediction. We typically set this to 50 for swin transformer. (Default: 0)""",
    )
    parser.add_argument(
        "--lambda1",
        default=1.0,
        type=float,
        help="""loss weight for dino
        loss over [CLS] tokens (Default: 1.0)""",
    )
    parser.add_argument(
        "--lambda2",
        default=1.0,
        type=float,
        help="""loss weight for beit 
        loss over masked patch tokens (Default: 1.0)""",
    )
    parser.add_argument(
        "--lambda3",
        default=1.0,
        type=float,
        help="""loss weight for beit 
        loss over masked object tokens (Default: 1.0)""",
    )

    # Temperature teacher parameters
    parser.add_argument(
        "--warmup_teacher_temp",
        default=0.04,
        type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""",
    )
    parser.add_argument(
        "--teacher_temp",
        default=0.04,
        type=float,
        help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""",
    )
    parser.add_argument(
        "--warmup_teacher_patch_temp",
        default=0.04,
        type=float,
        help="""See 
        `--warmup_teacher_temp`""",
    )
    parser.add_argument(
        "--teacher_patch_temp",
        default=0.07,
        type=float,
        help=""""See 
        `--teacher_temp`""",
    )
    parser.add_argument(
        "--warmup_teacher_temp_epochs",
        default=30,
        type=int,
        help="Number of warmup epochs for the teacher temperature (Default: 30).",
    )

    # Training/Optimization parameters
    parser.add_argument(
        "--use_fp16",
        type=utils_ibot.bool_flag,
        default=True,
        help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.04,
        help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""",
    )
    parser.add_argument(
        "--weight_decay_end",
        type=float,
        default=0.4,
        help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=3.0,
        help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""",
    )
    parser.add_argument(
        "--batch_size_per_gpu",
        default=128,
        type=int,
        help="Per-GPU batch-size : number of distinct images loaded on one GPU.",
    )
    parser.add_argument(
        "--epochs", default=100, type=int, help="Number of epochs of training."
    )
    parser.add_argument(
        "--freeze_last_layer",
        default=1,
        type=int,
        help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""",
    )
    parser.add_argument(
        "--lr",
        default=0.0005,
        type=float,
        help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""",
    )
    parser.add_argument(
        "--warmup_epochs",
        default=10,
        type=int,
        help="Number of epochs for the linear learning-rate warm up.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""",
    )
    parser.add_argument(
        "--optimizer",
        default="adamw",
        type=str,
        choices=["adamw", "sgd", "lars"],
        help="""Type of optimizer. We recommend using adamw with ViTs.""",
    )
    parser.add_argument(
        "--load_from",
        default=None,
        help="""Path to load checkpoints to resume training.""",
    )
    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        help="""Drop path rate for student network.""",
    )

    # Multi-crop parameters
    parser.add_argument(
        "--global_crops_number",
        type=int,
        default=2,
        help="""Number of global
        views to generate. Default is to use two global crops. """,
    )
    parser.add_argument(
        "--global_crops_scale",
        type=float,
        nargs="+",
        default=(0.25, 1.0),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""",
    )
    parser.add_argument(
        "--local_crops_number",
        type=int,
        default=0,
        help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """,
    )
    parser.add_argument(
        "--local_crops_scale",
        type=float,
        nargs="+",
        default=(0.05, 0.25),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""",
    )

    # Misc
    parser.add_argument(
        "--data_path",
        default="/path/to/imagenet/train/",
        type=str,
        help="Please specify path to the ImageNet training data.",
    )

    parser.add_argument("--boxes_root", default="/path/to/vidor/annotations/", type=str)
    parser.add_argument(
        "--output_dir", default=".", type=str, help="Path to save logs and checkpoints."
    )
    parser.add_argument(
        "--saveckp_freq", default=40, type=int, help="Save checkpoint every x epochs."
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument(
        "--num_workers",
        default=10,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""",
    )
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
        help="Please ignore and do not set this argument.",
    )

    # after "--local_crops_scale"
    parser.add_argument(
        "--frame_per_clip", default=2, type=int, help="frames per training clip"
    )
    parser.add_argument(
        "--pair_sampling",
        default="all",
        choices=["all", "consecutive", "random"],
        help="which frame pairs enter the cross-time loss",
    )
    parser.add_argument(
        "--time_matching",
        default="tg2sg_one2one",
        choices=[
            "tg2sg_one2one",
            "tg2sg_all2all",
            "sg2th_one2one",
            "sg2th_all2all",
            "tg2sg_sg2tg_one2one",
        ],
        help="matching strategy for the cross-time loss",
    )

    parser.add_argument(
        "--obj_time_matching",
        default="tg2sg_one2one",
        choices=[
            "tg2sg_one2one",
        ],
        help="matching strategy for the cross-time object-level loss",
    )

    parser.add_argument(
        "--lambda_timeimg",
        default=0.0,
        type=float,
        help="weight of cross-time CLS loss",
    )
    parser.add_argument(
        "--lambda_timeobj",
        default=0.0,
        type=float,
        help="weight of cross-time OBJ loss",
    )
    parser.add_argument(
        "--neighbor_mim_mask",
        default=False,
        type=utils_ibot.bool_flag,
        help="Whether to use MIM masks for neighbor frames.",
    )
    parser.add_argument(
        "--static_crop",
        default=False,
        type=utils_ibot.bool_flag,
        help="Whether to use static crops for neighbor frames.",
    )
    parser.add_argument(
        "--clever_initial_cropping",
        default=False,
        type=utils_ibot.bool_flag,
        help="Whether to use clever initial cropping for neighbor frames.",
    )

    parser.add_argument(
        "--resize_first",
        default=False,
        type=utils_ibot.bool_flag,
        help="Whether to resize the first frame before augmentations.",
    )
    parser.add_argument(
        "--resize_short_side",
        default=640,
        type=int,
        help="Resize dimension for the first frame before augmentations.",
    )

    # ----------  W&B  ----------
    wandb_group = parser.add_argument_group("wandb")
    wandb_group.add_argument(
        "--wandb",
        default=False,
        type=utils_ibot.bool_flag,
        help="log to Weights & Biases",
    )
    wandb_group.add_argument("--wandb_project", default=None, help="W&B project name")
    wandb_group.add_argument(
        "--wandb_entity", default="agape", help="W&B entity (team / user)"
    )
    wandb_group.add_argument(
        "--wandb_run_name", default=None, help="Optional run name shown in the UI"
    )
    wandb_group.add_argument(
        "--wandb_run_id",
        default=None,
        help="Resume run id (set automatically when you load a checkpoint)",
    )
    # -------------------------------------------

    return parser


def train_ibot(args):
    utils_ibot.init_distributed_mode(args)
    utils_ibot.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils_ibot.get_sha()))
    print(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = DataAugmentationSingleObject(
        args.global_crops_scale,
        args.local_crops_scale,
        args.global_crops_number,
        args.local_crops_number,
        sample_strategy=args.object_sampling_strategy,
        obj_aware_lc_loader=args.obj_aware_lc_loader,
        # num_objects=args.num_object_tokens,
        # patch_obj_assignment_strategy=args.patch_obj_assignment_strategy,
    )

    pred_size = args.patch_size * 8 if "swin" in args.arch else args.patch_size

    need_neighbor = args.lambda_timeimg > 0 or args.lambda_timeobj > 0

    dataset = ODISWTVeniceFrameDataset(
        data_path=args.data_path,
        boxes_root=args.boxes_root,
        transform=transform,
        patch_size=pred_size,
        pred_ratio=args.pred_ratio,
        pred_ratio_var=args.pred_ratio_var,
        pred_aspect_ratio=(0.3, 1 / 0.3),
        pred_shape=args.pred_shape,
        pred_start_epoch=args.pred_start_epoch,
        need_neighbor=need_neighbor,
        num_object_per_view=args.num_object_tokens,
        bb_margin=args.bb_margin,
        bb_margin_strategy=args.bb_margin_strategy,
        patch_obj_assignment_strategy=args.patch_obj_assignment_strategy,
        clever_initial_cropping=args.clever_initial_cropping,
        neighbor_mim_mask=args.neighbor_mim_mask,
        static_crop=args.static_crop,
        resize_first=args.resize_first,
        resize_short_side=args.resize_short_side,
    )

    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is of hierechical features (i.e. swin_tiny, swin_small, swin_base)
    if args.arch in models.__dict__.keys() and "swin" in args.arch:
        student = models.__dict__[args.arch](
            window_size=args.window_size,
            return_all_tokens=True,
            masked_im_modeling=args.use_masked_im_modeling,
            num_object_tokens=args.num_object_tokens,
        )
        teacher = models.__dict__[args.arch](
            window_size=args.window_size,
            drop_path_rate=0.0,
            return_all_tokens=True,
            num_object_tokens=args.num_object_tokens,
        )
        embed_dim = student.num_features
    # if the network is a vision transformer (i.e. vit_tiny, vit_small, vit_base, vit_large)
    elif args.arch in models.__dict__.keys():
        student = models.__dict__[args.arch](
            patch_size=args.patch_size,
            num_object_tokens=args.num_object_tokens,
            single_embed_obj_pos=args.single_embed_obj_pos,
            drop_path_rate=args.drop_path,
            return_all_tokens=True,
            masked_im_modeling=args.use_masked_im_modeling,
            use_different_obj_tokens=args.use_different_obj_tokens,
            separate_obj_attn=args.separate_obj_attn,
        )
        teacher = models.__dict__[args.arch](
            patch_size=args.patch_size,
            num_object_tokens=args.num_object_tokens,
            single_embed_obj_pos=args.single_embed_obj_pos,
            return_all_tokens=True,
            use_different_obj_tokens=args.use_different_obj_tokens,
            separate_obj_attn=args.separate_obj_attn,
        )
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(
        student,
        iBOTHead(
            embed_dim,
            args.out_dim,
            patch_out_dim=args.patch_out_dim,
            num_object_tokens=args.num_object_tokens,
            norm=args.norm_in_head,
            act=args.act_in_head,
            norm_last_layer=args.norm_last_layer,
            shared_head=args.shared_head,
        ),
    )
    teacher = utils.MultiCropWrapper(
        teacher,
        iBOTHead(
            embed_dim,
            args.out_dim,
            patch_out_dim=args.patch_out_dim,
            num_object_tokens=args.num_object_tokens,
            norm=args.norm_in_head,
            act=args.act_in_head,
            shared_head=args.shared_head_teacher,
        ),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils_ibot.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = (
            nn.parallel.DistributedDataParallel(
                teacher, device_ids=[args.gpu], broadcast_buffers=False
            )
            if "swin" in args.arch
            else nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        )
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = (
        nn.parallel.DistributedDataParallel(
            student, device_ids=[args.gpu], broadcast_buffers=False
        )
        if "swin" in args.arch
        else nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    )
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    same_dim = args.shared_head or args.shared_head_teacher
    G = args.global_crops_number  # views-per-frame (global)
    L = args.local_crops_number  # views-per-frame (local)
    T = args.frame_per_clip  # frames per clip

    ibot_loss = iBOTLoss(
        out_dim=args.out_dim,
        patch_out_dim=args.out_dim if same_dim else args.patch_out_dim,
        ngcrops=G,
        nlcrops=L,
        warmup_teacher_temp=args.warmup_teacher_temp,
        teacher_temp=args.teacher_temp,
        warmup_teacher_patch_temp=args.warmup_teacher_patch_temp,
        teacher_patch_temp=args.teacher_patch_temp,
        teacher_obj_temp=args.teacher_temp,
        warmup_teacher_obj_temp=args.warmup_teacher_temp,
        warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
        nepochs=args.epochs,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda_obj=args.lambda3,
        lambda_timeimg=args.lambda_timeimg,
        lambda_timeobj=args.lambda_timeobj,
        frames_per_clip=T,
        pair_sampling=args.pair_sampling,
        time_matching=args.time_matching,
        obj_time_matching=args.obj_time_matching,
        mim_start_epoch=args.pred_start_epoch,
        batch_size_per_gpu=args.batch_size_per_gpu,
    ).cuda()

    if utils_ibot.is_main_process():  # Tensorboard configuration
        local_runs = os.path.join(args.output_dir, "tf_logs")
        writer = SummaryWriter(logdir=local_runs)

    # ============ preparing optimizer ... ============
    params_groups = utils_ibot.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params_groups, lr=0, momentum=0.9
        )  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils_ibot.LARS(
            params_groups
        )  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils_ibot.cosine_scheduler(
        args.lr
        * (args.batch_size_per_gpu * utils_ibot.get_world_size())
        / 256.0,  # linear scaling rule
        args.min_lr,
        args.epochs,
        len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils_ibot.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils_ibot.cosine_scheduler(
        args.momentum_teacher, 1, args.epochs, len(data_loader)
    )

    print(f"Loss, optimizer and schedulers ready.")
    to_restore = {
        "epoch": 0,
        "wandb_run_name": None,
        "wandb_run_id": None,
        "wandb_entity": None,
        "wandb_project": None,
    }

    # ============ optionally resume training ... ============
    if args.load_from:
        utils_ibot.restart_from_checkpoint(
            os.path.join(args.output_dir, args.load_from),
            run_variables=to_restore,
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            ibot_loss=ibot_loss,
        )
        if args.wandb and args.wandb_run_id is None:
            print("Heading W&B run id from checkpoint...")
            args.wandb_run_id = to_restore["wandb_run_id"]
            args.wandb_project = (
                args.wandb_project or to_restore["wandb_project"]
            )  # args is prioritized
            args.wandb_entity = (
                args.wandb_entity or to_restore["wandb_entity"]
            )  # args is prioritized
            args.wandb_run_name = (
                args.wandb_run_name or to_restore["wandb_run_name"]
            )  # args is prioritized
    start_epoch = to_restore["epoch"]
    # set wandb
    if utils_ibot.is_main_process() and args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            id=args.wandb_run_id,
            resume="allow" if args.wandb_run_id else None,
            config=vars(args),
        )

    start_time = time.time()
    print("Starting iBOT training!")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        data_loader.dataset.set_epoch(epoch)

        # ============ training one epoch of iBOT ... ============
        train_stats = train_one_epoch(
            student,
            teacher,
            teacher_without_ddp,
            ibot_loss,
            data_loader,
            optimizer,
            lr_schedule,
            wd_schedule,
            momentum_schedule,
            epoch,
            fp16_scaler,
            args,
        )

        # ============ writing logs ... ============
        save_dict = {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": args,
            "ibot_loss": ibot_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict["fp16_scaler"] = fp16_scaler.state_dict()

            # save the wandb stats so that we can resume the run using run id
        if args.wandb and utils_ibot.is_main_process():
            save_dict.update(
                {
                    "wandb_run_name": wandb.run.name,
                    "wandb_run_id": wandb.run.id,
                    "wandb_entity": args.wandb_entity,
                    "wandb_project": args.wandb_project,
                }
            )
        utils_ibot.save_on_master(
            save_dict, os.path.join(args.output_dir, "checkpoint.pth")
        )
        if args.saveckp_freq and (epoch % args.saveckp_freq == 0) and epoch:
            utils_ibot.save_on_master(
                save_dict, os.path.join(args.output_dir, f"checkpoint{epoch:04}.pth")
            )
        elif epoch == 87:
            utils_ibot.save_on_master(
                save_dict, os.path.join(args.output_dir, f"checkpoint{epoch:04}.pth")
            )
        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }
        if utils_ibot.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
                for k, v in train_stats.items():
                    writer.add_scalar(k, v, epoch)

            # ------------- W&B (per‑epoch) -------------
            if args.wandb and utils_ibot.is_main_process():
                wandb.log(
                    log_stats, step=epoch
                )  # keys: loss, cls, obj, patch, nmi, ari, etc.
            # -------------------------------------------
        if args.wandb and utils_ibot.is_main_process():
            # log the epoch:
            wandb.log({"epoch": epoch}, step=epoch)

    if args.wandb and utils_ibot.is_main_process():
        wandb.finish()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def train_one_epoch(
    student,
    teacher,
    teacher_without_ddp,
    ibot_loss,
    data_loader,
    optimizer,
    lr_schedule,
    wd_schedule,
    momentum_schedule,
    epoch,
    fp16_scaler,
    args,
):
    metric_logger = utils_ibot.MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)

    # common params
    names_q, params_q, names_k, params_k = [], [], [], []
    for name_q, param_q in student.module.named_parameters():
        names_q.append(name_q)
        params_q.append(param_q)
    for name_k, param_k in teacher_without_ddp.named_parameters():
        names_k.append(name_k)
        params_k.append(param_k)
    names_common = list(set(names_q) & set(names_k))
    params_q = [
        param_q for name_q, param_q in zip(names_q, params_q) if name_q in names_common
    ]
    params_k = [
        param_k for name_k, param_k in zip(names_k, params_k) if name_k in names_common
    ]

    fetch_neighbors = args.lambda_timeimg > 0 or args.lambda_timeobj > 0

    pred_labels, real_labels = [], []
    for it, (
        images,
        obj_labels,
        obj_assignments,
        mim_masks,
        nimages,
        obj_neighbor_labels,
        nobj_assignments,
        masks_neigh,
    ) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        obj_assignments = [
            obj.cuda(non_blocking=True)
            for obj in obj_assignments[: args.global_crops_number]
        ]
        images = [im.cuda(non_blocking=True) for im in images]
        mim_masks = [m.cuda(non_blocking=True) for m in mim_masks]
        nimages = [m.cuda(non_blocking=True) for m in nimages]
        nobj_assignments = [
            obj.cuda(non_blocking=True)
            for obj in nobj_assignments[: args.global_crops_number]
        ]
        obj_labels = [label.cuda(non_blocking=True) for label in obj_labels]
        obj_neighbor_labels = [
            label.cuda(non_blocking=True) for label in obj_neighbor_labels
        ]

        # ---------------------------------------------------------------
        # QUICK DEBUG:  Are neighbour views really different?
        if it == 0:  # do it only on the first iter
            with torch.no_grad():
                B = images[0].size(0)  # batch size
                G = args.global_crops_number  # #global views per frame
                L = args.local_crops_number  # #local  views per frame
                V = G + L

                # compare every current view to its corresponding neighbour view
                # (=> same v-index, different frame).  Report perfect matches.
                tot_identical = 0

                for v in range(G):
                    cur = images[v].flatten(1)  # [B, 3*H*W]
                    neig = nimages[v].flatten(1) if len(nimages) > 0 else None
                    if neig is None:
                        print("⚠️  Dataset returned no neighbours")
                        break

                    identical = (cur == neig).all(dim=1)  # [B] bool
                    n_same = identical.sum().item()
                    tot_identical += n_same
                    if n_same > 0:
                        print(
                            f"🚨  view {v:>2}: {n_same}/{B} neighbours are *identical*"
                        )

                if tot_identical == 0:
                    print("✅  All neighbour images differ from current images.")
                else:
                    print(f"❌  {tot_identical}/{B*V} neighbour views are identical!")
        # ---------------------------------------------------------------

        #######################################################
        # images = G+L many tensors (each tensor: [B, 3, 224, 224])
        # nimages = G+L many tensors (each tensor: [B, 3, 224, 224]) (optional, it can be empty)
        # obj_labels = a tensor of shape [B, O]
        # obj_assignments = list of G many tensors (each tensor: [B, O, 14, 14]) (object assignments)
        # mim_masks = list of G many tensors (each tensor: [B, 14, 14]) (MIM masks)
        # LATER: nmim_masks= list of G many tensors (each tensor: [B, 14, 14]) (MIM masks for neighbor)

        G = args.global_crops_number
        L = args.local_crops_number
        Vpf = G + L  # views per *frame*

        images_global = images[:G]  # G global views
        masks_global = mim_masks[:G]  # G MIM masks
        images_local = images[G:]  # L local views

        nimages_global = []
        nimages_local = []
        if fetch_neighbors:
            nimages_global = nimages[:G]  # G global views
            nimages_local = nimages[G:]  # L local views

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # get global views
            teacher_output = teacher(
                images_global,
                obj_attn_mask=obj_assignments,
            )

            if fetch_neighbors:
                # teacher_neighbor_output = teacher(
                #    nimages_global,
                #    obj_attn_mask=nobj_assignments,
                # )
                teacher_neighbor_output = (None, None, None)
            else:
                teacher_neighbor_output = (None, None, None)

            # MIM (only for frame 1)
            # TODO: We should return the masks for the neighbor images as well
            student_output = student(
                images_global, mask=masks_global, obj_attn_mask=obj_assignments
            )

            # get local views
            student.module.backbone.masked_im_modeling = False
            student_local_output = student(images_local)[0:2]
            student_local_cls = (
                student_local_output[0] if student_local_output is not None else None
            )
            student_local_obj = (
                student_local_output[1] if student_local_output is not None else None
            )

            student_neighbor_output = (None, None, None)
            (
                student_neighbor_local_cls,
                student_neighbor_local_obj,
            ) = (None, None)
            if fetch_neighbors:
                # student_neighbor_local_output = student(nimages_local)[0:2]
                # student_neighbor_local_cls = student_neighbor_local_output[0]
                # student_neighbor_local_obj = student_neighbor_local_output[1]

                # neighbor global output is calculated WITHOUT MIM masks
                student_neighbor_output = student(
                    nimages_global, obj_attn_mask=nobj_assignments
                )

            student.module.backbone.masked_im_modeling = args.use_masked_im_modeling

            all_loss = ibot_loss(
                student_output,
                student_neighbor_output,
                teacher_output,
                teacher_neighbor_output,
                student_local_cls,
                student_local_obj,
                student_neighbor_local_cls,
                student_neighbor_local_obj,
                masks_global,
                obj_labels,
                obj_neighbor_labels,
                epoch,
            )
            loss = all_loss.pop("loss")

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # log statistics - [CLS]
        cls_probs1 = teacher_output[0].chunk(args.global_crops_number)
        cls_probs2 = student_output[0].chunk(args.global_crops_number)
        cls_pred1 = utils_ibot.concat_all_gather(cls_probs1[0].max(dim=1)[1])
        cls_pred2 = utils_ibot.concat_all_gather(cls_probs2[1].max(dim=1)[1])
        cls_acc = (cls_pred1 == cls_pred2).sum() / cls_pred1.size(0)
        # log statistics - [OBJ]
        obj_probs1 = teacher_output[1].chunk(args.global_crops_number)  # [V, B, O, D]
        obj_probs2 = student_output[1].chunk(args.global_crops_number)  # [V, B, O, D]
        obj_pred1 = utils_ibot.concat_all_gather(
            obj_probs1[0].view(-1, obj_probs1[0].size(-1)).max(dim=1)[1]
        )  # [B*O]
        obj_pred2 = utils_ibot.concat_all_gather(
            obj_probs2[1].view(-1, obj_probs2[1].size(-1)).max(dim=1)[1]
        )  # [B*O]
        obj_labels1 = utils_ibot.concat_all_gather(
            obj_labels[0].view(-1).to(obj_pred1.device)
        )
        obj_labels2 = utils_ibot.concat_all_gather(
            obj_labels[1].view(-1).to(obj_pred2.device)
        )
        obj_acc = obj_pred1 == obj_pred2
        obj_exist_in_both_views = (obj_labels1 != 0) & (obj_labels2 != 0)
        obj_acc = obj_acc[obj_exist_in_both_views].sum() / obj_exist_in_both_views.sum()
        pred_labels.append(obj_pred1)
        real_labels.append(obj_labels1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils_ibot.clip_gradients(student, args.clip_grad)
            utils_ibot.cancel_gradients_last_layer(
                epoch, student, args.freeze_last_layer
            )
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils_ibot.clip_gradients(student, args.clip_grad)
            utils_ibot.cancel_gradients_last_layer(
                epoch, student, args.freeze_last_layer
            )
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(params_q, params_k):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        for key, value in all_loss.items():
            metric_logger.update(**{key: value.item()})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        metric_logger.update(cls_acc=cls_acc, obj_acc=obj_acc)

        # wandb logging
        if args.wandb and utils_ibot.is_main_process():
            wandb.log(
                {
                    "iter_loss": loss.item(),
                    "cls_loss": all_loss["cls"].item(),
                    "patch_loss": all_loss["patch"].item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "obj_loss": all_loss["obj"].item(),
                    "timeimg_loss": all_loss["timeimg"].item(),
                    "timeobj_loss": all_loss["timeobj"].item(),
                    "cls_acc_step": cls_acc.item(),
                    "obj_acc_step": obj_acc.item(),
                }
            )  # keep the current step open

    pred_labels = torch.cat(pred_labels).cpu().detach().numpy()
    real_labels = torch.cat(real_labels).cpu().detach().numpy()
    nmi, ari, fscore, adjacc = eval_pred(real_labels, pred_labels, calc_acc=False)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("NMI: {}, ARI: {}, F: {}, ACC: {}".format(nmi, ari, fscore, adjacc))
    if args.wandb and utils_ibot.is_main_process():
        wandb.log({"nmi": nmi, "ari": ari, "fscore": fscore, "adjacc": adjacc})
    print("Averaged stats:", metric_logger)
    return_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return_dict.update({"nmi": nmi, "ari": ari, "fscore": fscore, "adjacc": adjacc})
    return return_dict


class iBOTLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        patch_out_dim,
        ngcrops,
        nlcrops,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_patch_temp,
        teacher_patch_temp,
        teacher_obj_temp,
        warmup_teacher_obj_temp,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
        center_momentum2=0.9,
        lambda1=1.0,
        lambda2=1.0,
        lambda_obj=1.0,
        lambda_timeobj=0.0,
        mim_start_epoch=0,
        lambda_timeimg=0.0,  #   cross-time CLS
        frames_per_clip=2,
        pair_sampling="all",  # all | consecutive | random
        time_matching="tg2sg_one2one",
        obj_time_matching="tg2sg_one2one",
        batch_size_per_gpu=64,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.center_momentum2 = center_momentum2
        self.ngcrops = ngcrops
        self.nlcrops = nlcrops
        self.ncrops = ngcrops + nlcrops
        self.register_buffer("center_cls", torch.zeros(1, out_dim))
        self.register_buffer("center_obj", torch.zeros(1, 1, out_dim))
        self.register_buffer("center_patch", torch.zeros(1, 1, patch_out_dim))
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda_obj = lambda_obj
        self.lambda_timeimg = lambda_timeimg
        self.lambda_timeobj = lambda_timeobj
        self.frames = frames_per_clip
        self.pair_sampling = pair_sampling
        self.time_matching = time_matching
        self.obj_time_matching = obj_time_matching
        self.batch_size_per_gpu = batch_size_per_gpu

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(
                    warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
                ),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )
        self.teacher_obj_temp_schedule = (
            np.concatenate(
                (
                    np.linspace(
                        warmup_teacher_obj_temp,
                        teacher_obj_temp,
                        warmup_teacher_temp_epochs,
                    ),
                    np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_obj_temp,
                )
            )
            if mim_start_epoch == 0
            else np.concatenate(
                (
                    np.ones(mim_start_epoch) * warmup_teacher_obj_temp,
                    np.linspace(
                        warmup_teacher_obj_temp,
                        teacher_obj_temp,
                        warmup_teacher_temp_epochs,
                    ),
                    np.ones(nepochs - warmup_teacher_temp_epochs - mim_start_epoch)
                    * teacher_obj_temp,
                )
            )
        )

        self.teacher_patch_temp_schedule = (
            np.concatenate(
                (
                    np.linspace(
                        warmup_teacher_patch_temp,
                        teacher_patch_temp,
                        warmup_teacher_temp_epochs,
                    ),
                    np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_patch_temp,
                )
            )
            if mim_start_epoch == 0
            else np.concatenate(
                (
                    np.ones(mim_start_epoch) * warmup_teacher_patch_temp,
                    np.linspace(
                        warmup_teacher_patch_temp,
                        teacher_patch_temp,
                        warmup_teacher_temp_epochs,
                    ),
                    np.ones(nepochs - warmup_teacher_temp_epochs - mim_start_epoch)
                    * teacher_patch_temp,
                )
            )
        )

    def forward(
        self,
        student_output,
        student_neighbor_output,
        teacher_output,
        teacher_neighbor_output,
        student_local_cls,
        student_local_obj,
        student_neighbor_local_cls,
        student_neighbor_local_obj,
        student_mask,
        obj_labels,
        obj_neighbor_labels,
        epoch,
    ):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_cls, student_obj, student_patch = student_output  # V many tensors
        teacher_cls, teacher_obj, teacher_patch = teacher_output

        # global
        student_neighbor_cls, student_neighbor_obj, student_neighbor_patch = (
            student_neighbor_output
        )
        teacher_neighbor_cls, teacher_neighbor_obj, teacher_neighbor_patch = (
            teacher_neighbor_output
        )

        if student_local_cls is not None:
            student_cls = torch.cat([student_cls, student_local_cls])
        if student_local_obj is not None:
            student_obj = torch.cat([student_obj, student_local_obj])

        # if student_neighbor_local_cls is not None:
        #    student_neighbor_cls = torch.cat(
        #        [student_neighbor_cls, student_neighbor_local_cls]
        #    )
        # if student_neighbor_local_obj is not None:
        #    student_neighbor_obj = torch.cat(
        #        [student_neighbor_obj, student_neighbor_local_obj]
        #    )

        # Shape analysis:
        # V = G + L
        # student_cls:  [B*T*(G+L), D]  (e.g. [256, 8192])
        # student_patch: [B*T*G, H*W, D] (e.g. [64, 196, 8192])
        # teacher_cls:  [B*T*G, D]  (e.g. [64, 8192])
        # teacher_patch: [B*T*G, H*W, D] (e.g. [64, 196, 8192])
        # student_mask: list of T*G tensors, each of shape [B, H', W'] (e.g. [64, 14, 14])  --> this is still a list since we didn't preprocess it before calling this func

        # print shapes:
        B = self.batch_size_per_gpu  # batch size
        G = self.ngcrops  # global views per frame
        L = self.nlcrops  # local  views per frame
        V = G + L  # total views / frame
        P = student_patch.size(1)  # number of patches (e.g. 196): H*W

        student_cls = student_cls / self.student_temp
        student_cls_c = student_cls.chunk(self.ncrops)
        student_obj = student_obj / self.student_temp
        student_obj_c = student_obj.chunk(self.ncrops)
        student_patch = student_patch / self.student_temp
        student_patch_c = student_patch.chunk(self.ngcrops)
        # # [B*G, P, D] -> list of G many [B, P, D]

        # neighbor:
        fetch_neighbors = self.lambda_timeimg > 0 or self.lambda_timeobj > 0
        if fetch_neighbors:
            student_neighbor_obj_c = student_neighbor_obj / self.student_temp
            student_neighbor_obj_c = student_neighbor_obj_c.chunk(self.ngcrops)

        temp = self.teacher_temp_schedule[epoch]
        temp_obj = self.teacher_obj_temp_schedule[epoch]
        temp_patch = self.teacher_patch_temp_schedule[epoch]
        teacher_cls_c = F.softmax((teacher_cls - self.center_cls) / temp, dim=-1)
        teacher_cls_c = teacher_cls_c.detach().chunk(self.ngcrops)
        teacher_obj_c = F.softmax((teacher_obj - self.center_obj) / temp_obj, dim=-1)
        teacher_obj_c = teacher_obj_c.detach().chunk(self.ngcrops)
        teacher_patch_c = F.softmax(
            (teacher_patch - self.center_patch) / temp_patch, dim=-1
        )
        teacher_patch_c = teacher_patch_c.detach().chunk(self.ngcrops)

        # neighbor:
        if fetch_neighbors:
            # teacher_neighbor_cls_c = F.softmax(
            #    (teacher_neighbor_cls - self.center_cls) / temp, dim=-1
            # )
            # teacher_neighbor_cls_c = teacher_neighbor_cls_c.detach().chunk(self.ngcrops)
            # teacher_neighbor_obj_c = F.softmax(
            #    (teacher_neighbor_obj - self.center_obj) / temp_obj, dim=-1
            # )
            # teacher_neighbor_obj_c = teacher_neighbor_obj_c.detach().chunk(self.ngcrops)
            # teacher_neighbor_patch_c = F.softmax(
            #    (teacher_neighbor_patch - self.center_patch) / temp_patch, dim=-1
            # )
            ##teacher_neighbor_patch_c = teacher_neighbor_patch_c.detach().chunk(self.ngcrops)
            pass

        total_loss_cls, n_loss_terms_cls = 0, 0
        total_loss_obj, n_loss_terms_obj = 0, 0
        total_loss_patch, n_loss_terms_patch = 0, 0
        total_loss_timeimg, n_loss_terms_timeimg = 0, 0
        total_loss_timeobj, n_loss_terms_timeobj = 0, 0

        # Global Crops - [CLS], [OBJ], [PATCH]
        for q in range(len(teacher_cls_c)):
            for v in range(len(student_cls_c)):
                if v == q:
                    # Patch-level loss
                    loss_patch = torch.sum(
                        -teacher_patch_c[q] * F.log_softmax(student_patch_c[v], dim=-1),
                        dim=-1,
                    )
                    mask = student_mask[v].flatten(-2, -1)
                    loss_patch = torch.sum(
                        loss_patch * mask.float(), dim=-1
                    ) / mask.sum(dim=-1).clamp(min=1.0)

                    total_loss_patch += loss_patch.mean()
                    n_loss_terms_patch += 1
                else:
                    # CLS-level loss
                    loss_cls = torch.sum(
                        -teacher_cls_c[q] * F.log_softmax(student_cls_c[v], dim=-1),
                        dim=-1,
                    )
                    total_loss_cls += loss_cls.mean()
                    n_loss_terms_cls += 1
                    # Object-level loss
                    obj_assignments_mask = obj_labels[v].clamp(max=1.0) * obj_labels[
                        q
                    ].clamp(
                        max=1.0
                    )  # [V, B, O] -> [B, O] obj labels between 0 - 255, 0: no object, 1-255: object class
                    loss_obj = torch.sum(
                        -teacher_obj_c[q] * F.log_softmax(student_obj_c[v], dim=-1),
                        dim=-1,
                    )
                    loss_obj = torch.sum(
                        loss_obj * obj_assignments_mask.float(), dim=-1
                    ) / obj_assignments_mask.sum(dim=-1).clamp(min=1.0)

                    total_loss_obj += loss_obj.mean()
                    n_loss_terms_obj += 1

        if self.lambda_timeimg > 0:
            # normally for a frame: we compare teacher global views with student global+local views
            # here we have different options:
            # 1. frame 1 teacher global view <--> frame 2 student global view
            # 2. frame 1 teacher global view <--> frame 2 student global+local view
            # 3. frame 1 teacher global view <--> frame 2 student local view ??  meaningless
            # 4. frame 1 teacher + student global view <--> frame 2 student global view
            # 5. frame 1 teacher + student global view <--> frame 2 student global + local view
            # 6. frame 1 teacher + student global view <--> frame 2 student local view ?? meaningless
            # ... it goes like this

            # there are two other options:
            # image-level loss like matching: each view is matched with the other views
            # patch-level loss like matching: each view is matched with the exact same view

            # also:
            # frame 1 teacher global view <--> frame 2 teacher global view

            if self.time_matching == "tg2sg_one2one":
                # reminder: there is no thing such as "teacher local view"
                # teachers are not given the local views

                # frame 1 teacher global view <--> frame 2 student global view
                for q in range(len(teacher_cls_c)):
                    for v in range(len(student_neighbor_cls_c)):
                        if q == v:
                            loss_cls = torch.sum(
                                -teacher_cls_c[q]
                                * F.log_softmax(student_neighbor_cls_c[v], dim=-1),
                                dim=-1,
                            )
                            total_loss_timeimg += loss_cls.mean()
                            n_loss_terms_timeimg += 1
            elif self.time_matching == "tg2sg_all2all":
                # all teacher global views are matched with all student global views
                for q in range(len(teacher_cls_c)):
                    for v in range(len(student_neighbor_cls_c)):
                        if q >= v:
                            loss_cls = torch.sum(
                                -teacher_cls_c[q]
                                * F.log_softmax(student_neighbor_cls_c[v], dim=-1),
                                dim=-1,
                            )
                            total_loss_timeimg += loss_cls.mean()
                            n_loss_terms_timeimg += 1
                        else:
                            pass  # student local view
            elif self.time_matching == "sg2th_one2one":
                # frame 1 student global view <--> frame 2 teacher global view
                for q in range(len(student_cls_c)):
                    for v in range(len(teacher_neighbor_cls_c)):
                        if q == v:
                            # frame 1 student global view <--> frame 2 teacher global view
                            loss_cls = torch.sum(
                                -teacher_neighbor_cls_c[v]
                                * F.log_softmax(student_cls_c[q], dim=-1),
                                dim=-1,
                            )
                            total_loss_timeimg += loss_cls.mean()
                            n_loss_terms_timeimg += 1
            elif self.time_matching == "sg2th_all2all":
                # all frame 1 student global views are matched with all frame 2 teacher global views
                for q in range(len(student_cls_c)):
                    for v in range(len(teacher_neighbor_cls_c)):
                        if q >= v:
                            loss_cls = torch.sum(
                                -teacher_neighbor_cls_c[v]
                                * F.log_softmax(student_cls_c[q], dim=-1),
                                dim=-1,
                            )
                            total_loss_timeimg += loss_cls.mean()
                            n_loss_terms_timeimg += 1
                        else:
                            pass  # student local view
            elif self.time_matching == "tg2sg_sg2tg_one2one":
                # frame 1 teacher global view <--> frame 2 student global view
                # frame 1 student global view <--> frame 2 teacher global view
                for q in range(len(teacher_cls_c)):
                    for v in range(len(student_neighbor_cls_c)):
                        if q == v:
                            loss_cls = torch.sum(
                                -teacher_cls_c[q]
                                * F.log_softmax(student_neighbor_cls_c[v], dim=-1),
                                dim=-1,
                            )
                            total_loss_timeimg += loss_cls.mean()
                            n_loss_terms_timeimg += 1

                            loss_cls = torch.sum(
                                -student_cls_c[v]
                                * F.log_softmax(teacher_neighbor_cls_c[q], dim=-1),
                                dim=-1,
                            )
                            total_loss_timeimg += loss_cls.mean()
                            n_loss_terms_timeimg += 1

            else:  # TODO
                raise NotImplementedError

        if self.lambda_timeobj > 0:
            # TODO: Implement this

            if self.obj_time_matching == "tg2sg_one2one":
                # frame 1 teacher global view <--> frame 2 student global view
                for q in range(len(teacher_obj_c)):
                    for v in range(len(student_neighbor_obj_c)):
                        if q == v:
                            # obj assignment mask is created using
                            # obj labels for the current frame
                            # and obj labels for the neighbor frame
                            obj_assignments_mask = obj_neighbor_labels[v].clamp(
                                max=1.0
                            ) * obj_labels[q].clamp(max=1.0)
                            loss_obj = torch.sum(
                                -teacher_obj_c[q]
                                * F.log_softmax(student_neighbor_obj_c[v], dim=-1),
                                dim=-1,
                            )
                            loss_obj = torch.sum(
                                loss_obj * obj_assignments_mask.float(), dim=-1
                            ) / obj_assignments_mask.sum(dim=-1).clamp(min=1.0)
                            total_loss_timeobj += loss_obj.mean()
                            n_loss_terms_timeobj += 1

        total_loss_cls = total_loss_cls / n_loss_terms_cls
        total_loss_patch = total_loss_patch / n_loss_terms_patch
        total_loss_obj = total_loss_obj / n_loss_terms_obj
        total_loss_timeimg = (
            total_loss_timeimg / n_loss_terms_timeimg
            if n_loss_terms_timeimg > 0
            else torch.tensor(0.0, device=teacher_cls.device)
        )
        total_loss_timeobj = (
            total_loss_timeobj / n_loss_terms_timeobj
            if n_loss_terms_timeobj > 0
            else torch.tensor(0.0, device=teacher_cls.device)
        )

        total_loss = dict(
            cls=total_loss_cls * self.lambda1,
            patch=total_loss_patch * self.lambda2,
            obj=total_loss_obj * self.lambda_obj,
            timeimg=total_loss_timeimg * self.lambda_timeimg,
            timeobj=total_loss_timeobj * self.lambda_timeobj,
            loss=total_loss_cls * self.lambda1
            + total_loss_patch * self.lambda2
            + total_loss_timeimg * self.lambda_timeimg
            + total_loss_timeobj * self.lambda_timeobj
            + total_loss_obj * self.lambda_obj,
        )

        self.update_center(teacher_cls, teacher_obj, teacher_patch)
        return total_loss
        # -------------------------------------------------------------

    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_obj, teacher_patch):
        """
        Update center used for teacher output.
        teacher_cls: [V*B, D]
        teacher_obj: [V*B, O, D]
        teacher_patch: [V*B, P, D]
        """
        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        dist.all_reduce(cls_center)
        cls_center = cls_center / (len(teacher_cls) * dist.get_world_size())
        self.center_cls = self.center_cls * self.center_momentum + cls_center * (
            1 - self.center_momentum
        )

        # active_obj = teacher_labels != 0  # [V*B, O]
        obj_center = torch.sum(teacher_obj, dim=0, keepdim=True)
        dist.all_reduce(obj_center)
        obj_center = obj_center / (len(teacher_obj) * dist.get_world_size())
        self.center_obj = self.center_obj * self.center_momentum + obj_center * (
            1 - self.center_momentum
        )

        patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
        dist.all_reduce(patch_center)
        patch_center = patch_center / (len(teacher_patch) * dist.get_world_size())
        self.center_patch = self.center_patch * self.center_momentum2 + patch_center * (
            1 - self.center_momentum2
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("iBOT", parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_ibot(args)
