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
import copy
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from loader_imagenet_mine import ImagenetBB
import utils
import models
import models_ibot

from torch import nn
from torchvision import transforms as pth_transforms
from loader import ImageFolderInstance


# ➋ helper to pool / reshape object tokens -----------------------------------
def pool_obj_tokens(tokens_3d, mode, attn_weights=None):
    """
    tokens_3d : Tensor[B, O, D] – object tokens ONLY
    mode      : 'avg' | 'max' | 'cat' | 'attn'
    attn_weights: Tensor[B, O] – normalised weights per object (optional)

    Returns: Tensor[B, D]  (or [B, O*D] for 'cat')
    """
    if mode == "avg":
        return tokens_3d.mean(1)

    if mode == "max":
        return tokens_3d.max(1).values

    if mode == "cat":  # flatten to (B, O*D)
        return tokens_3d.reshape(tokens_3d.size(0), -1)

    if mode == "attn":
        assert attn_weights is not None, "Need attention weights for obj-attn"
        # attn_weights already normalised so just @ gives weighted sum
        return torch.einsum("bo,bod->bd", attn_weights, tokens_3d)

    raise ValueError(f"Unknown mode {mode}")


def eval_knn(args):
    print("git:\n  {}\n".format(utils.get_sha()))
    print(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )
    cudnn.benchmark = True

    if args.load_features:
        try:
            print("loading features...")
            train_features = torch.load(
                os.path.join(args.load_features, "trainfeat.pth")
            )
            test_features = torch.load(os.path.join(args.load_features, "testfeat.pth"))
            train_labels = torch.load(
                os.path.join(args.load_features, "trainlabels.pth")
            )
            test_labels = torch.load(os.path.join(args.load_features, "testlabels.pth"))
        except:
            train_features, test_features, train_labels, test_labels = (
                extract_feature_pipeline(args)
            )
    else:
        # need to extract features !
        train_features, test_features, train_labels, test_labels = (
            extract_feature_pipeline(args)
        )

    if utils.get_rank() == 0:
        if args.use_cuda:
            train_features = train_features.cuda()
            test_features = test_features.cuda()
            train_labels = train_labels.cuda()
            test_labels = test_labels.cuda()

        print("Features are ready!\nStart the k-NN classification.")
        for k in args.nb_knn:
            top1, top5 = knn_classifier(
                train_features,
                train_labels,
                test_features,
                test_labels,
                k,
                args.temperature,
                args.use_cuda,
            )
            print(f"{k}-NN classifier result: Top1: {top1}%, Top5: {top5}%")
    dist.barrier()


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


def extract_feature_pipeline(args):
    requested_num_obj_tokens = args.num_object_tokens
    args.num_object_tokens = infer_num_obj_tokens(
        args.pretrained_weights, args.checkpoint_key, default=args.num_object_tokens
    )
    print(f"→ Detected num_object_tokens = {args.num_object_tokens}")
    assert (
        requested_num_obj_tokens == args.num_object_tokens
    ), "Mismatch between checkpoint and requested num_object_tokens!"

    # ============ preparing data ... ============
    transform = pth_transforms.Compose(
        [
            pth_transforms.Resize(256, interpolation=3),
            pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    bb_transform = pth_transforms.Compose(
        [
            pth_transforms.Resize(256, interpolation=3),
            pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
        ]
    )
    traindir = os.path.join(args.data_path, "train")
    valdir = os.path.join(args.data_path, "val")

    dataset_train = ImagenetBB(
        traindir,
        boxes_root=args.boxes_root,
        transform=transform,
        bb_transform=bb_transform,
        train=True,
        which_boxes=args.which_boxes,
        num_object_tokens=args.num_object_tokens,
    )

    dataset_val = ImagenetBB(
        valdir,
        boxes_root=args.boxes_root,
        transform=transform,
        bb_transform=bb_transform,
        train=False,
        which_boxes=args.which_boxes,
        num_object_tokens=args.num_object_tokens,
    )
    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(
        f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs."
    )

    # load the checkpoint and read content
    checkpoint = torch.load(args.pretrained_weights, map_location="cpu")
    # get the model shape
    model = checkpoint["student"]
    # ============ building network ... ============
    if "swin" in args.arch:
        args.patch_size = 4
        model = models.__dict__[args.arch](
            window_size=args.window_size,
            patch_size=args.patch_size,
            num_classes=0,
            num_object_tokens=args.num_object_tokens,
        )
    else:
        model = models.__dict__[args.arch](
            patch_size=args.patch_size,
            num_classes=0,
            use_mean_pooling=args.avgpool_patchtokens == 1,
            num_object_tokens=args.num_object_tokens,
        )
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    model.cuda()
    utils.load_pretrained_weights(
        model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size
    )
    model.eval()

    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features, train_labels = extract_features(
        model,
        data_loader_train,
        args.n_last_blocks,
        args.avgpool_patchtokens,
        args.use_cuda,
        args.knn_token,
        args.use_bounding_boxes,
        num_object_tokens=args.num_object_tokens,
    )
    print("Extracting features for val set...")
    test_features, test_labels = extract_features(
        model,
        data_loader_val,
        args.n_last_blocks,
        args.avgpool_patchtokens,
        args.use_cuda,
        args.knn_token,
        args.use_bounding_boxes,
        num_object_tokens=args.num_object_tokens,
    )

    if utils.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

    # save features and labels
    if args.dump_features and dist.get_rank() == 0:
        print("Dumping features ...")
        torch.save(
            train_features.cpu(), os.path.join(args.dump_features, "trainfeat.pth")
        )
        torch.save(
            test_features.cpu(), os.path.join(args.dump_features, "testfeat.pth")
        )
        torch.save(
            train_labels.cpu(), os.path.join(args.dump_features, "trainlabels.pth")
        )
        torch.save(
            test_labels.cpu(), os.path.join(args.dump_features, "testlabels.pth")
        )
    return train_features, test_features, train_labels, test_labels

def patch_avg(blk, num_obj):
    """blk: [B, 1+O+P, D] → averaged patch token [B, D]"""
    return blk[:, 1 + num_obj : ].mean(1)

@torch.no_grad()
def extract_features(
    model,
    data_loader,
    n,
    avgpool,
    use_cuda=True,
    knn_token="cls",
    use_bounding_boxes=True,
    multiscale=False,
    num_object_tokens=1,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    labels = None
    for samples, obj_attn_masks, labs, index in metric_logger.log_every(
        data_loader, 10
    ):
        

        # obj_attention_masks = [B, O, 1+O+H*W]
        samples = samples.cuda(non_blocking=True)
        if obj_attn_masks is not None and obj_attn_masks.numel():
            obj_attn_masks = obj_attn_masks.cuda(non_blocking=True)
        else:            # either None *or* an empty tensor with 0 rows
            obj_attn_masks = None
        labs = labs.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)

        if knn_token == "obj-per":
            labs = labs.repeat_interleave(num_object_tokens)
            index = index.repeat_interleave(num_object_tokens)

        def forward_single(samples):
            if use_bounding_boxes and obj_attn_masks is not None:
                # use bounding boxes to compute features
                
                intermediate_output = model.get_intermediate_layers(
                    samples, n, obj_attn_mask=obj_attn_masks
                )
            else:
                intermediate_output = model.get_intermediate_layers(samples, n)

            reps = []
            
            # we apply bbox for obj token but not for cls token if use_bounding_boxes is set to False
            if (knn_token in ["fuse-cat", "fuse-sum", "fuse-obj-patch", "fuse-obj-cls-patch"]):
                # use bbox for obj, dont use it for cls
                inter_bbox = model.get_intermediate_layers(
                    samples, n, obj_attn_mask=obj_attn_masks
                )
                inter_nomask = model.get_intermediate_layers(
                    samples, n, obj_attn_mask=None
                )


            if knn_token == "obj-attn":
                # NOTE: We use the same attention weights for all blocks (i.e. the last self-attention)
                attn = model.get_last_selfattention(
                    samples,
                    obj_attn_mask=obj_attn_masks if use_bounding_boxes else None,
                )  # [B, H, 1+O+H*W, 1+O+H*W]
                attn = attn.mean(1)  # mean over heads: [B, 1+O+H*W, 1+O+H*W]

                # the attention weights that the CLS token (query index 0) puts on each of the O object tokens (key indices 1..O)
                obj_weights = attn[:, 0, 1 : 1 + num_object_tokens]
                obj_weights = obj_weights / obj_weights.sum(dim=1, keepdim=True)

            if knn_token in ["fuse-cat", "fuse-sum", "fuse-obj-patch", "fuse-obj-cls-patch"]:
                if use_bounding_boxes and obj_attn_masks is not None: # use bboxes for all
                    inter_nomask = inter_bbox
                else:
                    pass 
                
                for blk_out, blk_out_obj in zip(
                    inter_nomask, inter_bbox
                ):
                    if knn_token == "fuse-cat":
                        cls_vec   = blk_out[:, 0]                              # [B,D]
                        obj_vec   = blk_out_obj[:, 1 : 1 + num_object_tokens].mean(1)
                        fused     = torch.cat([cls_vec, obj_vec], dim=-1)
                        
                    elif knn_token == "fuse-sum":
                        cls_vec   = blk_out[:, 0]                              # [B,D]
                        obj_vec   = blk_out_obj[:, 1 : 1 + num_object_tokens].mean(1)
                        fused     = cls_vec + obj_vec

                    elif knn_token == "fuse-obj-patch":
                        obj_vec = blk_out_obj[:, 1 : 1 + num_object_tokens].mean(1)  # [B, D]
                        patch_vec = patch_avg(blk_out, num_object_tokens)
                        fused = torch.cat([obj_vec, patch_vec], dim=-1)
                    
                    elif knn_token == "fuse-obj-cls-patch":
                        obj_vec = blk_out_obj[:, 1 : 1 + num_object_tokens].mean(1)
                        patch_vec = patch_avg(blk_out, num_object_tokens)
                        cls_vec = blk_out[:, 0]
                        fused = torch.cat([cls_vec, obj_vec, patch_vec], dim=-1)
                        
                    
                    reps.append(fused)  # [B, D + num_object_tokens * D]
            
            else:
                for blk_out in intermediate_output:
                    if knn_token == "cls":
                        reps.append(blk_out[:, 0])
                    elif knn_token.startswith("obj"):
                        obj_tok = blk_out[:, 1 : 1 + num_object_tokens]
                        if knn_token == "obj-0":
                            assert torch.allclose(blk_out[:, 1], obj_tok[:, 0], rtol=1e-5, atol=1e-8), \
                                "blk_out[:,1] and obj_tok[:,0] are not close enough!"
                            reps.append(obj_tok[:, 0])  # take the first object token
                        elif knn_token == "obj-avg":
                            reps.append(pool_obj_tokens(obj_tok, "avg"))
                        elif knn_token == "obj-max":
                            reps.append(pool_obj_tokens(obj_tok, "max"))
                        elif knn_token == "obj-cat":
                            reps.append(pool_obj_tokens(obj_tok, "cat"))
                        elif knn_token == "obj-attn":
                            reps.append(
                                pool_obj_tokens(obj_tok, "attn", attn_weights=obj_weights)
                            )
                        elif knn_token == "obj-per":
                            reps.append(obj_tok)  # [B, O, D]
                    else:
                        raise ValueError(f"Invalid knn_token {knn_token}")

            if knn_token == "obj-per":
                # [B, O, D] * n -> cat along O
                reps = [t.reshape(t.size(0) * t.size(1), -1) for t in reps]
            else:
                reps = [t.reshape(t.size(0), -1) for t in reps]
            return torch.cat(reps, dim=-1).clone()

        if multiscale:
            v = None
            for s in [1, 1 / 2 ** (1 / 2), 1 / 2]:  # we use 3 different scales
                if s == 1:
                    inp = samples.clone()
                else:
                    inp = nn.functional.interpolate(
                        samples, scale_factor=s, mode="bilinear", align_corners=False
                    )
                feats = forward_single(inp)
                if v is None:
                    v = feats
                else:
                    v += feats
            v /= 3
            v /= v.norm()
            feats = v
        else:
            feats = forward_single(samples)

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            multiplier = num_object_tokens if knn_token == "obj-per" else 1

            features = torch.zeros(
                len(data_loader.dataset) * multiplier, feats.shape[-1]
            ).to(feats.dtype)
            labels = torch.zeros(len(data_loader.dataset) * multiplier).to(labs.dtype)
            if use_cuda:
                features = features.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")
            print(f"Storing labels into tensor of shape {labels.shape}")

        # get indexes from all processes
        y_all = torch.empty(
            dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device
        )
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        labels_all = torch.empty(
            dist.get_world_size(),
            labs.size(0),
            dtype=labs.dtype,
            device=labs.device,
        )
        label_l = list(labels_all.unbind(0))
        label_all_reduce = torch.distributed.all_gather(label_l, labs, async_op=True)
        label_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
                labels.index_copy_(0, index_all, torch.cat(label_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
                labels.index_copy_(0, index_all.cpu(), torch.cat(label_l).cpu())
    return features, labels


@torch.no_grad()
def knn_classifier(
    train_features,
    train_labels,
    test_features,
    test_labels,
    k,
    T,
    use_cuda=True,
    num_classes=1000,
):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes)
    if use_cuda:
        retrieval_one_hot = retrieval_one_hot.cuda()
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = test_features[idx : min((idx + imgs_per_chunk), num_test_images), :]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, 5).sum().item()
        total += targets.size(0)

    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    return top1, top5


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluation with weighted k-NN on ImageNet")
    parser.add_argument(
        "--n_last_blocks",
        default=1,
        type=int,
        help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=1` all the time for k-NN evaluation.""",
    )
    parser.add_argument(
        "--avgpool_patchtokens",
        default=0,
        choices=[0, 1, 2],
        type=int,
        help="""Whether or not to use global average pooled features or the [CLS] token.
        We typically set this to 1 for BEiT and 0 for models with [CLS] token (e.g., DINO).
        we set this to 2 for base-size models with [CLS] token when doing linear classification.""",
    )
    parser.add_argument(
        "--batch_size_per_gpu", default=128, type=int, help="Per-GPU batch-size"
    )
    parser.add_argument(
        "--num_object_tokens",
        default=1,
        type=int,
        help="Number of object tokens per view.",
    )
    parser.add_argument(
        "--nb_knn",
        default=[5, 8, 10, 20, 100, 200],
        nargs="+",
        type=int,
        help="Number of NN to use. 20 is usually working the best.",
    )
    parser.add_argument(
        "--temperature",
        default=0.07,
        type=float,
        help="Temperature used in the voting coefficient",
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
        "--knn_token",
        default="cls",
        type=str,
        choices=[
            "cls",
            "obj-0",
            "obj-avg",
            "obj-max",
            "obj-cat",
            "obj-per",
            "obj-attn",
            "fuse-cat",        #  (early-fusion: CLS‖OBJ)
            "fuse-sum",        #  (w·CLS + (1-w)·OBJ)
            "fuse-obj-patch",
            "fuse-obj-cls-patch",
        ],
        help="How to turn object tokens into a single image representation.",
    )
    parser.add_argument(
        "--use_bounding_boxes",
        default=True,
        type=utils.bool_flag,
        help="Use bounding boxes for informed attention to compute features.",
    )
    # TODO: Update the ImageNetBB to accept which_boxes argument
    parser.add_argument(
        "--which_boxes",
        default="ground_truth",
        type=str,
        choices=["ground_truth", "grounding_dino"],
        help="The bounding boxes to use.",
    )
    parser.add_argument(
        "--use_cuda",
        default=True,
        type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM",
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
        "--checkpoint_key",
        default="teacher",
        type=str,
        help='Key to use in the checkpoint (example: "teacher")',
    )
    parser.add_argument(
        "--dump_features",
        default=None,
        help="Path where to save computed features, empty for no saving",
    )
    parser.add_argument(
        "--load_features",
        default=".",
        help="""If the features have
        already been computed, where to find them.""",
    )
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

    args = parser.parse_args()
    
    if args.num_object_tokens == 0:
        # 1. You cannot ask for an object-token based representation
        if args.knn_token != "cls":
            raise ValueError(
                "num_object_tokens=0 ⇒ only --knn_token cls is valid "
                f"(got {args.knn_token!r})"
            )


    utils.init_distributed_mode(args)
    for checkpoint_key in args.checkpoint_key.split(","):
        print("Starting evaluating {}.".format(checkpoint_key))
        args_copy = copy.deepcopy(args)
        args_copy.checkpoint_key = checkpoint_key
        eval_knn(args_copy)
