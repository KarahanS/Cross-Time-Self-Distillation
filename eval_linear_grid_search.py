# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Copy-paste from DINO library:
https://github.com/facebookresearch/dino
"""

import os
import argparse
import json
import copy
import torch
import torch.backends.cudnn as cudnn
from loader_imagenet_mine import ImagenetBBV2LP
import utils
import models

from pathlib import Path
from torch import nn
from torchvision import transforms as pth_transforms
from torchvision.transforms import v2 as T
from loader import ImageFolder


def infer_num_obj_tokens(ckpt_path, checkpoint_key="teacher", default=1):
    """
    Return the number of object tokens stored in the checkpoint.
    Falls back to `default` if the field cannot be found.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # ---------- 1. Prefer the saved training args ----------
    if "args" in ckpt:
        args_blob = ckpt["args"]
        if isinstance(args_blob, dict) and "num_object_tokens" in args_blob:
            return int(args_blob["num_object_tokens"])
        if hasattr(args_blob, "num_object_tokens"):
            return int(args_blob.num_object_tokens)

    # ---------- 2. Otherwise infer from a weight tensor ----------
    state_dict = ckpt.get(checkpoint_key, ckpt)  # works for DDP or plain
    for k, v in state_dict.items():
        if k.endswith("object_tokens") and v.ndim == 2:
            return int(v.shape[0])

    print(
        f"[infer_num_obj_tokens] Could not find info in {ckpt_path}. "
        f"Using default={default}."
    )
    return default


def eval_linear(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )
    cudnn.benchmark = True

    # fix the seed for reproducibility
    utils.fix_random_seeds(args.seed)

    # ============ preparing data ... ============
    train_spatial_transform = T.Compose(
        [
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
        ]
    )
    train_transform_rest = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    val_spatial_transform = T.Compose(
        [
            T.Resize(256, interpolation=3),
            T.CenterCrop(224),
        ]
    )
    val_transform_rest = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    traindir = os.path.join(args.data_path, "train")
    valdir = os.path.join(args.data_path, "val")

    requested_num_obj_tokens = args.num_object_tokens
    args.num_object_tokens = infer_num_obj_tokens(
        args.pretrained_weights, args.checkpoint_key, default=args.num_object_tokens
    )
    print(f"→ Detected num_object_tokens = {args.num_object_tokens}")
    assert (
        requested_num_obj_tokens == args.num_object_tokens
    ), "Mismatch between checkpoint and requested num_object_tokens!"

    dataset_train = ImagenetBBV2LP(
        traindir,
        boxes_root=args.boxes_root,
        spatial_transform=train_spatial_transform,
        transform_rest=train_transform_rest,
        num_object_per_view=requested_num_obj_tokens,
        train=True,
    )
    dataset_val = ImagenetBBV2LP(
        valdir,
        boxes_root=args.boxes_root,
        spatial_transform=val_spatial_transform,
        transform_rest=val_transform_rest,
        num_object_per_view=requested_num_obj_tokens,
        train=False,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(
        f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs."
    )

    # ============ building network ... ============

    model = models.__dict__[args.arch](
        patch_size=args.patch_size,
        num_classes=0,
        num_object_tokens=args.num_object_tokens,
        use_mean_pooling=False,
    )
    embed_dim = model.embed_dim
    model.cuda()
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    # load weights to evaluate
    utils.load_pretrained_weights(
        model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size
    )

    linear_classifiers, optim_param_groups = setup_linear_classifiers(
        embed_dim,
        args.n_last_blocks_list,
        args.learning_rates,
        args.batch_size_per_gpu,
        obj_pools=args.obj_pool,
        num_object_tokens=args.num_object_tokens,
        num_classes=1000,
    )
    linear_classifiers = nn.parallel.DistributedDataParallel(
        linear_classifiers, device_ids=[args.gpu]
    )

    optimizer = torch.optim.SGD(
        optim_param_groups,
        momentum=0.9,
        weight_decay=0,  # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=0
    )

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.0}
    if args.load_from:
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, args.load_from),
            run_variables=to_restore,
            state_dict=linear_classifiers,
            optimizer=optimizer,
            scheduler=scheduler,
        )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]
    max_n_last_blocks = max(args.n_last_blocks_list)

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        model.eval()
        linear_classifiers.train()

        train_stats = train(
            model, linear_classifiers, optimizer, train_loader, epoch, max_n_last_blocks
        )
        scheduler.step()

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            model.eval()
            linear_classifiers.eval()
            test_stats = validate_network(
                val_loader, model, linear_classifiers, max_n_last_blocks
            )
            # print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            results_dict = {}
            max_accuracy = 0
            best_classifier = ""
            for k, v in test_stats.items():
                if k.startswith("acc1_") and v > max_accuracy:
                    max_accuracy = v
                    best_classifier = k

            results_dict["best_classifier"] = {
                "name": best_classifier,
                "accuracy": max_accuracy,
            }
            print(f"Best classifier: {results_dict['best_classifier']}")

            log_stats = {
                **{k: v for k, v in log_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
            }

            if utils.is_main_process() and (max_accuracy >= best_acc):
                # always only save best checkpoint till now
                with (Path(args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                with (Path(args.output_dir) / "best_classifier.txt").open("a") as f:
                    f.write(f"epoch: {epoch}\n")
                    for k, v in results_dict.items():
                        f.write(json.dumps({k: v}) + "\n")
                    f.write("\n")

                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": linear_classifiers.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_acc": max_accuracy,
                }
                torch.save(
                    save_dict,
                    os.path.join(
                        args.output_dir,
                        "checkpoint_{}_linear.pth".format(args.checkpoint_key),
                    ),
                )

            best_acc = max(best_acc, max_accuracy)
            print(f"Max accuracy so far: {best_acc:.2f}%")

    print(
        "Training of the supervised linear classifier on frozen features completed.\n"
        "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc)
    )


def train(model, linear_classifiers, optimizer, loader, epoch, n):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    for inp, target, obj_attn_mask in metric_logger.log_every(loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        obj_attn_mask = obj_attn_mask.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            intermediate_output = model.get_intermediate_layers(
                inp, n, obj_attn_mask=obj_attn_mask
            )

        outputs = linear_classifiers(intermediate_output)

        # compute cross entropy loss
        losses = {
            f"loss_{k}": nn.CrossEntropyLoss()(v, target) for k, v in outputs.items()
        }
        loss = sum(losses.values())

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, linear_classifiers, n):
    linear_classifiers.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    for inp, target, obj_attn_mask in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        obj_attn_mask = obj_attn_mask.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            intermediate_output = model.get_intermediate_layers(
                inp, n, obj_attn_mask=obj_attn_mask
            )

        output = linear_classifiers(intermediate_output)
        losses = {
            f"loss_{k}": nn.CrossEntropyLoss()(v, target) for k, v in output.items()
        }
        loss = sum(losses.values())

        # if linear_classifiers.module.num_labels >= 5:
        #     acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        # else:
        metric_logger.update(loss=loss.item())
        batch_size = inp.shape[0]
        for k, v in output.items():
            (acc1,) = utils.accuracy(v, target, topk=(1,))
            metric_logger.meters[f"acc1_{k}"].update(acc1.item(), n=batch_size)
            # metric_logger.meters[f'acc5_{k}'].update(acc5.item(), n=batch_size)

        # if linear_classifiers.module.num_labels >= 5:
        #     metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # if linear_classifiers.module.num_labels >= 5:
    #     print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #       .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    # else:
    # print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
    #     .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# class LinearClassifier(nn.Module):
#     """Linear layer to train on top of frozen features"""
#     def __init__(self, dim, num_labels=1000):
#         super(LinearClassifier, self).__init__()
#         self.num_labels = num_labels
#         self.linear = nn.Linear(dim, num_labels)
#         self.linear.weight.data.normal_(mean=0.0, std=0.01)
#         self.linear.bias.data.zero_()

#     def forward(self, x):
#         # flatten
#         x = x.view(x.size(0), -1)


#         # linear layer
#         return self.linear(x)
def pool_obj_tokens(tokens_3d, mode, attn_weights=None):
    """
    tokens_3d : Tensor[B, O, D]   – object tokens **only**
    mode      : 'avg' | 'max' | 'cat' | 'attn'
    Returns   : Tensor[B, D] or [B, O*D] for 'cat'
    """
    if mode == "avg":
        return tokens_3d.mean(1)
    if mode == "max":
        return tokens_3d.max(1).values
    if mode == "cat":
        return tokens_3d.flatten(1)  # (B, O*D)
    if mode == "attn":
        assert attn_weights is not None
        # attn_weights: (B, O) already normalised → weighted sum
        return torch.einsum("bo,bod->bd", attn_weights, tokens_3d)
    raise ValueError(f"unknown mode {mode}")


def create_linear_input(
    x_tokens_list, use_n_blocks, use_avgpool, obj_pool="obj-0", num_object_tokens=1
):
    intermediate_output = x_tokens_list[-use_n_blocks:]  # last n blocks
    reps = []

    if obj_pool == "obj-attn":
        # NOTE: If it doesn't work, switch to how we did in KNN eval
        attn = x_tokens_list[-1]._attention.mean(1)  # (B, 1+O+H*W, 1+O+H*w)
        obj_w = attn[:, 0, 1 : 1 + num_object_tokens]
        obj_w = obj_w / obj_w.sum(1, keepdim=True)

    for blk in intermediate_output:
        if obj_pool == "cls":
            reps.append(blk[:, 0])
        else:
            obj_tok = blk[:, 1 : 1 + num_object_tokens]  # (B, O, D)
            if obj_pool == "obj-0":
                reps.append(obj_tok[:, 0])
            elif obj_pool == "obj-avg":
                reps.append(pool_obj_tokens(obj_tok, "avg"))
            elif obj_pool == "obj-max":
                reps.append(pool_obj_tokens(obj_tok, "max"))
            elif obj_pool == "obj-cat":
                reps.append(pool_obj_tokens(obj_tok, "cat"))
            elif obj_pool == "obj-attn":
                reps.append(pool_obj_tokens(obj_tok, "attn", obj_w))
            else:
                raise ValueError(obj_pool)

    out = torch.cat(reps, dim=-1)

    # optional global patch average pooling
    if use_avgpool:
        patch_avg = torch.mean(
            intermediate_output[-1][:, 1 + num_object_tokens :], dim=1
        )
        out = torch.cat((out, patch_avg), dim=-1)

    return out.float()


def scale_lr(learning_rates, batch_size):
    return learning_rates * (batch_size * utils.get_world_size()) / 256.0


def setup_linear_classifiers(
    embed_dim,
    n_last_blocks_list,
    learning_rates,
    batch_size,
    obj_pools=["cls", "obj-avg"],  # default list
    num_object_tokens=1,
    num_classes=1000,
):
    linear_classifiers_dict = nn.ModuleDict()
    optim_param_groups = []
    for n in n_last_blocks_list:
        for avgpool in [False, True]:
            for obj_pool in obj_pools:  # whatever set you want
                for _lr in learning_rates:
                    lr = scale_lr(_lr, batch_size)
                    dim = (
                        embed_dim * (n + int(avgpool))
                        if obj_pool != "obj-cat"
                        else embed_dim * (n * num_object_tokens)
                    )
                    linear_classifier = LinearClassifier(
                        dim,
                        use_n_blocks=n,
                        use_avgpool=avgpool,
                        obj_pool=obj_pool,
                        num_object_tokens=num_object_tokens,
                        num_classes=num_classes,
                    )
                    linear_classifier = linear_classifier.cuda()
                    linear_classifiers_dict[
                        f"classifier_{n}_blocks_avgpool_{avgpool}_lr_{_lr:.5f}".replace(
                            ".", "_"
                        )
                    ] = linear_classifier
                    optim_param_groups.append(
                        {"params": linear_classifier.parameters(), "lr": lr}
                    )

    linear_classifiers = AllClassifiers(linear_classifiers_dict)

    return linear_classifiers, optim_param_groups


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(
        self,
        dim,
        use_n_blocks,
        use_avgpool,
        obj_pool,
        num_object_tokens,
        num_classes=1000,
    ):
        super().__init__()
        self.out_dim = dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes
        self.obj_pool = obj_pool
        self.num_object_tokens = num_object_tokens
        self.linear = nn.Linear(dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x_tokens_list):
        x = create_linear_input(
            x_tokens_list,
            self.use_n_blocks,
            self.use_avgpool,
            self.obj_pool,
            self.num_object_tokens,
        )
        return self.linear(x)


class AllClassifiers(nn.Module):
    def __init__(self, classifiers_dict):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict()
        self.classifiers_dict.update(classifiers_dict)

    def forward(self, inputs):
        return {k: v.forward(inputs) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Evaluation with linear classification on ImageNet"
    )
    parser.add_argument(
        "--n_last_blocks_list",
        default=[1, 4],
        nargs="+",
        type=int,
        help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base/Large.""",
    )
    parser.add_argument(
        "--avgpool_patchtokens",
        default=0,
        choices=[0, 1, 2],
        type=int,
        help="""Whether or not to use global average pooled features or the [CLS] token.
        We typically set this to 1 for BEiT and 0 for models with [CLS] token (e.g., DINO).
        we set this to 2 for base/large-size models with [CLS] token when doing linear classification.""",
    )
    parser.add_argument(
        "--arch",
        default="vit_small",
        type=str,
        choices=[
            "vit_tiny",
            "vit_small",
            "vit_base",
            "vit_large",
            "swin_tiny",
            "swin_small",
            "swin_base",
            "swin_large",
            "resnet50",
            "resnet101",
            "dalle_encoder",
        ],
        help="Architecture.",
    )
    parser.add_argument(
        "--patch_size", default=16, type=int, help="Patch resolution of the model."
    )
    parser.add_argument(
        "--window_size", default=7, type=int, help="Window size of the model."
    )
    parser.add_argument(
        "--pretrained_weights",
        default="",
        type=str,
        help="""Path to pretrained 
        weights to evaluate. Set to `download` to automatically load the pretrained DINO from url.
        Otherwise the model is randomly initialized""",
    )
    parser.add_argument(
        "--checkpoint_key",
        default="teacher",
        type=str,
        help='Key to use in the checkpoint (example: "teacher")',
    )
    parser.add_argument(
        "--epochs", default=100, type=int, help="Number of epochs of training."
    )
    parser.add_argument(
        "--learning_rates",
        default=[
            1e-5,
            2e-5,
            5e-5,
            1e-4,
            2e-4,
            5e-4,
            1e-3,
            2e-3,
            5e-3,
            1e-2,
            2e-2,
            5e-2,
            0.1,
            0.2,
            0.5,
        ],
        nargs="+",
        type=float,
        help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""",
    )
    parser.add_argument(
        "--batch_size_per_gpu", default=128, type=int, help="Per-GPU batch-size"
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
    parser.add_argument(
        "--data_path",
        default="/path/to/imagenet/",
        type=str,
        help="Please specify path to the ImageNet data.",
    )
    parser.add_argument(
        "--boxes_root",
        default="/path/to/imagenet/bboxes",
        type=str,
        help="Please specify path to the ImageNet bounding boxes.",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--num_workers",
        default=10,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser.add_argument(
        "--val_freq", default=1, type=int, help="Epoch frequency for validation."
    )
    parser.add_argument(
        "--output_dir", default=".", help="Path to save logs and checkpoints"
    )
    parser.add_argument(
        "--num_labels",
        default=1000,
        type=int,
        help="Number of labels for linear classifier",
    )
    parser.add_argument(
        "--load_from", default=None, help="Path to load checkpoints to resume training"
    )
    parser.add_argument("--num_object_tokens", default=1, type=int)
    parser.add_argument(
        "--obj_pool",
        default=["cls"],  # default list
        nargs="+",  # ← accept 1 … N values
        choices=["cls", "obj-0", "obj-avg", "obj-max", "obj-cat", "obj-attn"],
        help="Space-separated list of pooling modes used for linear probes.",
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for checkpoint_key in args.checkpoint_key.split(","):
        print("Starting evaluating {}.".format(checkpoint_key))
        args_copy = copy.deepcopy(args)
        args_copy.checkpoint_key = checkpoint_key
        eval_linear(args_copy)
