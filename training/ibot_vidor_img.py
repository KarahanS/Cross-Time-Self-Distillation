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
import models_ibot
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
from models_ibot.ibot_head import iBOTHead
from loader import ImageFolderMask
from evaluation.unsupervised.unsup_cls import eval_pred
from loader_vidor import DataAugmentationiBOT
import wandb


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
        "--window_size",
        default=7,
        type=int,
        help="""Size of window - default 7.
        This config is only valid for Swin Transofmer and is ignoired for vanilla ViT architectures.""",
    )
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
        default=(0.14, 1.0),
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
        default=(0.05, 0.4),
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
    parser.add_argument(
        "--dataset",
        default="VidORiBOTFrameDataset",
        choices=["ImageFolderMask", "VidORiBOTFrameDataset"],
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

    # turn off everything object–related by default
    parser.add_argument(
        "--num_object_tokens",
        default=0,
        type=int,
        help="0 → no [OBJ] tokens, object loss disabled",
    )
    parser.add_argument(
        "--lambda_timeimg",
        default=0.0,
        type=float,
        help="weight of cross-time CLS loss",
    )
    parser.add_argument(
        "--neigh_mim",  # whether we make masks for the neighbour
        default="random",  #  random | shared | off
        choices=["random", "shared", "off"],
    )
    # the neighbour frame gets its own fresh mask
    # shared → the neighbour re‑uses the anchor‑frame mask
    # off → revert to the behaviour you have now (CLS only

    parser.add_argument(
        "--lambda_timepatch", default=0.0, type=float  # cross-time patch loss
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
    transform = DataAugmentationiBOT(
        args.global_crops_scale,
        args.local_crops_scale,
        args.global_crops_number,
        args.local_crops_number,
    )
    pred_size = args.patch_size * 8 if "swin" in args.arch else args.patch_size

    if args.dataset == "ImageFolderMask":
        dataset = ImageFolderMask(
            args.data_path,
            transform=transform,
            patch_size=pred_size,
            pred_ratio=args.pred_ratio,
            pred_ratio_var=args.pred_ratio_var,
            pred_aspect_ratio=(0.3, 1 / 0.3),
            pred_shape=args.pred_shape,
            pred_start_epoch=args.pred_start_epoch,
        )

    elif args.dataset == "VidORiBOTFrameDataset":
        from loader_vidor import VidORiBOTFrameDataset

        need_neighbor = args.lambda_timeimg > 0
        neigh_create_mask = args.neigh_mim == "random"
        # if it is off or shared, we don't have to create different masks for the neighbour

        dataset = VidORiBOTFrameDataset(
            data_path=args.data_path,
            boxes_root=args.boxes_root,
            transform=transform,
            patch_size=pred_size,
            pred_ratio=args.pred_ratio,
            pred_ratio_var=args.pred_ratio_var,
            pred_aspect_ratio=(0.3, 1 / 0.3),
            pred_shape=args.pred_shape,
            pred_start_epoch=args.pred_start_epoch,
            need_neighbour=need_neighbor,
            neigh_mim=neigh_create_mask,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

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
    if args.arch in models_ibot.__dict__.keys() and "swin" in args.arch:
        student = models_ibot.__dict__[args.arch](
            window_size=args.window_size,
            return_all_tokens=True,
            masked_im_modeling=args.use_masked_im_modeling,
            num_object_tokens=0,
        )
        teacher = models_ibot.__dict__[args.arch](
            window_size=args.window_size,
            drop_path_rate=0.0,
            return_all_tokens=True,
            num_object_tokens=0,
        )
        embed_dim = student.num_features
    # if the network is a vision transformer (i.e. vit_tiny, vit_small, vit_base, vit_large)
    elif args.arch in models_ibot.__dict__.keys():
        student = models_ibot.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path,
            return_all_tokens=True,
            masked_im_modeling=args.use_masked_im_modeling,
        )
        teacher = models_ibot.__dict__[args.arch](
            patch_size=args.patch_size,
            return_all_tokens=True,
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
    student = utils_ibot.MultiCropWrapper(
        student,
        iBOTHead(
            embed_dim,
            args.out_dim,
            patch_out_dim=args.patch_out_dim,
            norm=args.norm_in_head,
            act=args.act_in_head,
            norm_last_layer=args.norm_last_layer,
            shared_head=args.shared_head,
        ),
    )
    teacher = utils_ibot.MultiCropWrapper(
        teacher,
        iBOTHead(
            embed_dim,
            args.out_dim,
            patch_out_dim=args.patch_out_dim,
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
        warmup_teacher_temp2=args.warmup_teacher_patch_temp,
        teacher_temp2=args.teacher_patch_temp,
        warmup_teacher_temp_epochs=args.warmup_teacher_temp_epochs,
        nepochs=args.epochs,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda_timeimg=args.lambda_timeimg,
        lambda_timepatch=args.lambda_timepatch,
        frames_per_clip=T,
        pair_sampling=args.pair_sampling,
        time_matching=args.time_matching,
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
            print(to_restore["wandb_project"])
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

            print("Wandb info retrieved.")
            print(args.wandb_project)
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

    pred_labels, real_labels = [], []
    for it, (images, labels, mim_masks, nimages, nmim_masks) in enumerate(
        metric_logger.log_every(data_loader, 10, header)
    ):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        images = [im.cuda(non_blocking=True) for im in images]
        mim_masks = [m.cuda(non_blocking=True) for m in mim_masks]
        nimages = [m.cuda(non_blocking=True) for m in nimages]
        nmim_masks = (
            [m.cuda(non_blocking=True) for m in nmim_masks] if nmim_masks else []
        )

        #######################################################
        # images = G+L many tensors (each tensor: [B, 3, 224, 224])
        # nimages = G+L many tensors (each tensor: [B, 3, 224, 224]) (optional, it can be empty)
        # labels = a tensor of shape [B] (it is actually dummy for vidor)
        # mim_masks = list of G many tensors (each tensor: [B, 14, 14]) (MIM masks)
        # LATER: nmim_masks= list of G many tensors (each tensor: [B, 14, 14]) (MIM masks for neighbor)

        G = args.global_crops_number
        L = args.local_crops_number
        Vpf = G + L  # views per *frame*

        images_global = images[:G]  # G global views
        masks_global = mim_masks[:G]  # G MIM masks
        images_local = images[G:]  # L local views

        if args.neigh_mim == "shared":
            neigh_masks_global = mim_masks[:G]  # re‑use anchor masks
        elif args.neigh_mim == "random":
            neigh_masks_global = nmim_masks[:G]  # fresh masks
        else:  # 'off' – keep behaviour you had
            neigh_masks_global = None

        nimages_global = []
        nimages_local = []
        if args.lambda_timeimg > 0:
            nimages_global = nimages[:G]  # G global views
            nimages_local = nimages[G:]  # L local views

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # get global views

            # teacher_output_all = teacher(images_global + nimages_global)
            teacher_output_all = teacher(images_global)

            #  teacher_output = tuple (cls, obj, patch) each has shape:
            # teacher_cls.shape = [B x (G + G if neighbor), 8192]
            # TODO: student_obj.shape = [B x (G + G if neighbor), O, 8192]
            # teacher_patch.shape = [B x (G + G if neighbor), 196, 8192]

            B = args.batch_size_per_gpu  # batch size
            teacher_cls = teacher_output_all[0][: B * G]
            teacher_patch = teacher_output_all[1][
                : B * G
            ]  # TODO: change idx to 2 when obj comes

            teacher_neighbor_cls = None
            teacher_neighbor_patch = None

            if args.lambda_timeimg > 0:
                # teacher_neighbor_cls = teacher_output_all[0][B * G :]
                # teacher_neighbor_patch = teacher_output_all[1][
                #    B * G :
                # ]  # TODO: change idx to 2 when obj comes

                pass  # for now, we do not process teacher for neighbor frames

            # MIM
            student_output = student(images_global, mask=masks_global)
            student_neighbor_global = None

            if args.lambda_timeimg > 0 and neigh_masks_global is not None:
                student_neighbor_global = student(
                    nimages_global, mask=neigh_masks_global
                )

            # get local views
            student.module.backbone.masked_im_modeling = False
            student_local_cls = student(images_local)[0] if len(images_local) else None

            if args.lambda_timeimg > 0 and neigh_masks_global is None:
                student_neighbor_global = student(nimages_global)  # without mask

            student.module.backbone.masked_im_modeling = args.use_masked_im_modeling

            all_loss = ibot_loss(
                student_output,
                teacher_cls,
                teacher_patch,
                student_neighbor_global,
                student_local_cls,
                masks_global,
                neigh_masks_global,
                epoch,
            )
            loss = all_loss.pop("loss")

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        probs1 = teacher_cls.chunk(args.global_crops_number)
        probs2 = student_output[0].chunk(args.global_crops_number)
        pred1 = utils_ibot.concat_all_gather(probs1[0].max(dim=1)[1])
        pred2 = utils_ibot.concat_all_gather(probs2[1].max(dim=1)[1])
        acc = (pred1 == pred2).sum() / pred1.size(0)
        pred_labels.append(pred1)
        real_labels.append(utils_ibot.concat_all_gather(labels.to(pred1.device)))

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
        metric_logger.update(acc=acc)

        # wandb logging
        if args.wandb and utils_ibot.is_main_process():
            wandb.log(
                {
                    "iter_loss": loss.item(),
                    "cls_loss": all_loss["cls"].item(),
                    "patch_loss": all_loss["patch"].item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "timeimg_loss": all_loss["timeimg"].item(),
                    "cls_acc_step": acc.item(),
                    "timepatch_loss": all_loss["timepatch"].item(),
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
        warmup_teacher_temp2,
        teacher_temp2,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
        center_momentum2=0.9,
        lambda1=1.0,
        lambda2=1.0,
        mim_start_epoch=0,
        lambda_timeimg=0.0,  #   cross-time CLS
        lambda_timepatch=0.0,  # cross-time patch
        frames_per_clip=2,
        pair_sampling="all",  # all | consecutive | random
        time_matching="tg2sg_one2one",
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
        self.register_buffer("center_patch", torch.zeros(1, 1, patch_out_dim))
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda_timeimg = lambda_timeimg
        self.lambda_timepatch = lambda_timepatch
        self.frames = frames_per_clip
        self.pair_sampling = pair_sampling
        self.time_matching = time_matching
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
        self.teacher_temp2_schedule = (
            np.concatenate(
                (
                    np.linspace(
                        warmup_teacher_temp2, teacher_temp2, warmup_teacher_temp_epochs
                    ),
                    np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp2,
                )
            )
            if mim_start_epoch == 0
            else np.concatenate(
                (
                    np.ones(mim_start_epoch) * warmup_teacher_temp2,
                    np.linspace(
                        warmup_teacher_temp2, teacher_temp2, warmup_teacher_temp_epochs
                    ),
                    np.ones(nepochs - warmup_teacher_temp_epochs - mim_start_epoch)
                    * teacher_temp2,
                )
            )
        )

    def forward(
        self,
        student_output,
        teacher_cls,
        teacher_patch,
        student_neighbor_global,
        student_local_cls,
        masks_global,
        neigh_masks_global,
        epoch,
    ):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_cls, student_patch = student_output
        if student_local_cls is not None:
            student_cls = torch.cat([student_cls, student_local_cls])

        if student_neighbor_global is not None:
            neigh_student_cls, neigh_student_patch = (
                student_neighbor_global  # both are masked
            )
            neigh_student_patch = neigh_student_patch / self.student_temp
            neigh_student_patch_c = neigh_student_patch.chunk(
                self.ngcrops
            )  # list[G × B × P × D]
        else:
            neigh_student_patch_c = None

        student_neighbor_global_cls = (
            student_neighbor_global[0] if student_neighbor_global is not None else None
        )

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
        student_cls_c = student_cls.chunk(
            self.ncrops
        )  # [B*V, D] -> list of V many [B, D]
        student_patch = student_patch / self.student_temp
        student_patch_c = student_patch.chunk(
            self.ngcrops
        )  # [B*G, P, D] -> list of G many [B, P, D]

        temp = self.teacher_temp_schedule[epoch]
        temp2 = self.teacher_temp2_schedule[epoch]
        teacher_cls_c = F.softmax((teacher_cls - self.center_cls) / temp, dim=-1)
        teacher_cls_c = teacher_cls_c.detach().chunk(self.ngcrops)
        teacher_patch_c = F.softmax((teacher_patch - self.center_patch) / temp2, dim=-1)
        teacher_patch_c = teacher_patch_c.detach().chunk(self.ngcrops)

        if self.lambda_timeimg > 0:
            # TODO: Have different temperatures for these guys? ODIS doesn't have, though.
            student_neighbor_global_cls_c = (
                student_neighbor_global_cls / self.student_temp
            )
            student_neighbor_global_cls_c = student_neighbor_global_cls_c.chunk(
                self.ngcrops
            )

            # student_neighbor_local_cls_c = (
            #    student_neighbor_local_cls / self.student_temp
            # )
            # student_neighbor_local_cls_c = student_neighbor_local_cls_c.chunk(
            #    self.nlcrops
            # )

            # teacher_neighbor_cls_c = F.softmax(
            #    (teacher_neighbor_cls - self.center_cls) / temp, dim=-1
            # )
            # teacher_neighbor_cls_c = teacher_neighbor_cls_c.detach().chunk(self.ngcrops)
            # teacher_neighbor_patch_c = F.softmax(
            #    (teacher_neighbor_patch - self.center_patch) / temp2, dim=-1
            # )
            # teacher_neighbor_patch_c = teacher_neighbor_patch_c.detach().chunk(
            #    self.ngcrops
            # )

        total_loss_cls, n_loss_terms_cls = 0, 0
        total_loss_patch, n_loss_terms_patch = 0, 0

        # Global Crops - [CLS], [OBJ], [PATCH]
        for q in range(len(teacher_cls_c)):
            for v in range(len(student_cls_c)):
                if v == q:
                    # Patch-level loss
                    loss_patch = torch.sum(
                        -teacher_patch_c[q] * F.log_softmax(student_patch_c[v], dim=-1),
                        dim=-1,
                    )
                    mask = masks_global[v].flatten(-2, -1)
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

        # ------------------------------------------------------------------
        # cross‑time masked‑patch loss:  teacher(unmasked, t)  vs student(masked, t+1)
        # ------------------------------------------------------------------
        total_loss_timepatch, n_loss_terms_timepatch = 0, 0
        if self.lambda_timepatch > 0 and neigh_student_patch_c is not None:
            for q in range(len(teacher_patch_c)):  # teacher views
                for v in range(len(neigh_student_patch_c)):  # neighbour views
                    if q == v:  # one‑to‑one
                        loss_patch = torch.sum(
                            -teacher_patch_c[q]
                            * F.log_softmax(neigh_student_patch_c[v], dim=-1),
                            dim=-1,
                        )
                        # use the *neighbour* mask when we compute the loss
                        mask = neigh_masks_global[v].flatten(-2, -1).float()
                        loss_patch = torch.sum(loss_patch * mask, dim=-1) / mask.sum(
                            dim=-1
                        ).clamp(min=1.0)
                        total_loss_timepatch += loss_patch.mean()
                        n_loss_terms_timepatch += 1

        if n_loss_terms_timepatch:
            total_loss_timepatch = total_loss_timepatch / n_loss_terms_timepatch
        else:
            total_loss_timepatch = torch.tensor(0.0, device=teacher_cls.device)

        total_loss_timeimg = 0
        n_loss_terms_timeimg = 0
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
                    for v in range(len(student_neighbor_global_cls_c)):
                        if q == v:
                            loss_cls = torch.sum(
                                -teacher_cls_c[q]
                                * F.log_softmax(
                                    student_neighbor_global_cls_c[v], dim=-1
                                ),
                                dim=-1,
                            )
                            total_loss_timeimg += loss_cls.mean()
                            n_loss_terms_timeimg += 1
            elif self.time_matching == "tg2sg_all2all":
                for q in range(len(teacher_cls_c)):
                    for v in range(len(student_neighbor_global_cls_c)):
                        loss_cls = torch.sum(
                            -teacher_cls_c[q]
                            * F.log_softmax(student_neighbor_global_cls_c[v], dim=-1),
                            dim=-1,
                        )
                        total_loss_timeimg += loss_cls.mean()
                        n_loss_terms_timeimg += 1
            else:  # TODO
                raise NotImplementedError

        total_loss_cls = total_loss_cls / n_loss_terms_cls
        total_loss_patch = total_loss_patch / n_loss_terms_patch
        total_loss_timeimg = (
            total_loss_timeimg / n_loss_terms_timeimg
            if n_loss_terms_timeimg > 0
            else torch.tensor(0.0, device=teacher_cls.device)
        )

        total_loss = dict(
            cls=total_loss_cls * self.lambda1,
            patch=total_loss_patch * self.lambda2,
            timeimg=total_loss_timeimg * self.lambda_timeimg,
            timepatch=total_loss_timepatch * self.lambda_timepatch,
            loss=total_loss_cls * self.lambda1
            + total_loss_patch * self.lambda2
            + total_loss_timeimg * self.lambda_timeimg
            + total_loss_timepatch * self.lambda_timepatch,
        )

        self.update_center(teacher_cls, teacher_patch)
        return total_loss
        # -------------------------------------------------------------

    @torch.no_grad()
    def update_center(self, teacher_cls, teacher_patch):
        # Should I update the center using teacher_neighbor_cls as well?
        # chatgpt answer:
        """
        If you call update_center(teacher_cls, teacher_patch) (anchor only)
        You reproduce vanilla iBOT behaviour; the extra views affect the loss but not the
        running centre. That keeps the centre less “blurred” and is what many papers do.
        """

        cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
        dist.all_reduce(cls_center)
        cls_center = cls_center / (len(teacher_cls) * dist.get_world_size())
        self.center_cls = self.center_cls * self.center_momentum + cls_center * (
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
