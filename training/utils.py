# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""

import os
import sys
import time
import math
import json
import random
import datetime
import subprocess
import numpy as np
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from collections import defaultdict, deque
from pathlib import Path
from torch import nn
from PIL import ImageFilter, ImageOps, Image, ImageDraw

import einops
from einops import rearrange, repeat, reduce
from torch.nn import functional as F


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class PermutePatch(object):
    """
    Apply Patch permutation to the PIL image.
    """

    def __init__(self, psz):
        self.psz = psz

    def __call__(self, img):
        imgs = []
        imgwidth, imgheight = img.size
        for i in range(0, imgheight, self.psz):
            for j in range(0, imgwidth, self.psz):
                box = (j, i, j + self.psz, i + self.psz)
                imgs.append(img.crop(box))
        random.shuffle(imgs)
        new_img = Image.new("RGB", (imgwidth, imgheight))
        k = 0
        for i in range(0, imgheight, self.psz):
            for j in range(0, imgwidth, self.psz):
                new_img.paste(imgs[k], (j, i))
                k += 1
        return new_img


class HideAndSeek(object):
    """
    Apply Patch permutation to the PIL image.
    """

    def __init__(self, ratio, psz):
        self.ratio = ratio
        self.psz = psz

    def __call__(self, img):
        imgwidth, imgheight = img.size
        numw, numh = imgwidth // self.psz, imgheight // self.psz
        mask_num = int(numw * numh * self.ratio)
        mask_patch = np.random.choice(np.arange(numw * numh), mask_num, replace=False)
        mask_w, mask_h = mask_patch % numh, mask_patch // numh
        # img.save('test1.png')
        draw = ImageDraw.Draw(img)
        for mw, mh in zip(mask_w, mask_h):
            draw.rectangle(
                (
                    mw * self.psz,
                    mh * self.psz,
                    (mw + 1) * self.psz,
                    (mh + 1) * self.psz,
                ),
                fill="black",
            )
        # img.save('test2.png')
        return img


def load_pretrained_weights(
    model, pretrained_weights, checkpoint_key, model_name, patch_size
):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(
            "Pretrained weights found at {} and loaded with msg: {}".format(
                pretrained_weights, msg
            )
        )
        return
    elif pretrained_weights == "download":
        url = None
        if model_name == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif model_name == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif model_name == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        if url is not None:
            print(
                "Since no pretrained weights are provided, we load the pretrained weights from {}.".format(
                    url
                )
            )
            state_dict = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/dino/" + url
            )
            model.load_state_dict(state_dict, strict=True)
            return
    print(
        "There is no reference weights available for this model => We use random weights."
    )


def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(
                    "=> loaded '{}' from checkpoint '{}' with msg {}".format(
                        key, ckp_path, msg
                    )
                )
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print(
                        "=> failed to load '{}' from checkpoint: '{}'".format(
                            key, ckp_path
                        )
                    )
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def cosine_scheduler(
    base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0
):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.6f}")
        data_time = SmoothedValue(fmt="{avg:.6f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.6f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    # launched with submitit on a slurm cluster
    # elif 'SLURM_PROCID' in os.environ:
    #     args.rank = int(os.environ['SLURM_PROCID'])
    #     args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print("Will run the code on one GPU.")
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
    else:
        print("Does not support training without GPU.")
        sys.exit(1)

    if sys.platform == "win32":
        backend = "gloo"
    else:
        backend = "nccl"

    dist.init_process_group(
        backend=backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class LARS(torch.optim.Optimizer):
    """
    Almost copy-paste from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """

    def __init__(
        self,
        params,
        lr=0,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim != 1:
                    dp = dp.add(p, alpha=g["weight_decay"])

                if p.ndim != 1:
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


def create_ds_config(args):
    args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * get_world_size(),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "bias_correction": True,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                },
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 7,
                "loss_scale_window": 128,
            },
        }

        writer.write(json.dumps(ds_config, indent=2))


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, backbone, head=None):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        if head is None:
            self.head = nn.Identity()
        else:
            self.head = head

    def forward(
        self,
        x,
        mask=None,
        obj_attn_mask=None,
        return_backbone_feat=False,
        head_only: bool = False,
        **kwargs,
    ):
        if head_only:
            assert not isinstance(
                x, list
            ), "head_only should be used only with a single tensor input"
            return self.head(x)

        # convert to list
        if not isinstance(x, list):
            x = [x]
            mask = [mask] if mask is not None else None
            obj_attn_mask = [obj_attn_mask] if obj_attn_mask is not None else None
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1],
            0,
        )
        start_idx = 0
        for end_idx in idx_crops:
            inp_x = torch.cat(x[start_idx:end_idx])

            if mask is not None:
                inp_m = torch.cat(mask[start_idx:end_idx])
                kwargs.update(dict(mask=inp_m))
            if obj_attn_mask is not None:
                inp_o = torch.cat(obj_attn_mask[start_idx:end_idx])
                kwargs.update(dict(obj_attn_mask=inp_o))

            _out = self.backbone(inp_x, **kwargs)
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        output_ = self.head(output)
        if return_backbone_feat:
            return output, output_
        return output_


class MultiCropWrapperWStats(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, backbone, head=None):
        super(MultiCropWrapperWStats, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        if head is None:
            self.head = nn.Identity()
        else:
            self.head = head

    def forward(
        self,
        x,
        mask=None,
        obj_attn_mask=None,
        teacher_obj_assignments=None,
        return_backbone_feat=False,
        **kwargs,
    ):
        # convert to list
        if not isinstance(x, list):
            x = [x]
            mask = [mask] if mask is not None else None
            obj_attn_mask = [obj_attn_mask] if obj_attn_mask is not None else None
            teacher_obj_assignments = (
                [teacher_obj_assignments]
                if teacher_obj_assignments is not None
                else None
            )
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1],
            0,
        )
        start_idx = 0
        for end_idx in idx_crops:
            inp_x = torch.cat(x[start_idx:end_idx])

            if mask is not None:
                inp_m = torch.cat(mask[start_idx:end_idx])
                kwargs.update(dict(mask=inp_m))
            if obj_attn_mask is not None:
                inp_o = torch.cat(obj_attn_mask[start_idx:end_idx])
                kwargs.update(dict(obj_attn_mask=inp_o))
            if teacher_obj_assignments is not None:
                # inp_t = torch.cat(teacher_obj_assignments[start_idx: end_idx])
                kwargs.update(dict(teacher_obj_assignments=teacher_obj_assignments))

            _out, _obj_assign_counts, _attn_occur_mask, _obj_attn = self.backbone(
                inp_x, **kwargs
            )
            if start_idx == 0:
                output = _out
                obj_assign_counts = _obj_assign_counts
                attn_occur_mask = _attn_occur_mask
                obj_attn = _obj_attn
            else:
                output = torch.cat((output, _out))
                obj_assign_counts = torch.cat((obj_assign_counts, _obj_assign_counts))
                attn_occur_mask = torch.cat((attn_occur_mask, _attn_occur_mask))
                obj_attn = torch.cat((obj_attn, _obj_attn))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        output_ = self.head(output)
        if return_backbone_feat:
            return output, output_
        return output_, obj_assign_counts, attn_occur_mask, obj_attn


class MultiCropWrapperWObjAssign(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """

    def __init__(self, backbone, head=None):
        super(MultiCropWrapperWObjAssign, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        if head is None:
            self.head = nn.Identity()
        else:
            self.head = head

    def forward(
        self,
        x,
        mask=None,
        obj_attn_mask=None,
        teacher_obj_assignments=None,
        return_backbone_feat=False,
        **kwargs,
    ):
        # convert to list
        if not isinstance(x, list):
            x = [x]
            mask = [mask] if mask is not None else None
            obj_attn_mask = [obj_attn_mask] if obj_attn_mask is not None else None
            teacher_obj_assignments = (
                [teacher_obj_assignments]
                if teacher_obj_assignments is not None
                else None
            )
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1],
            0,
        )
        start_idx = 0
        for end_idx in idx_crops:
            inp_x = torch.cat(x[start_idx:end_idx])

            if mask is not None:
                inp_m = torch.cat(mask[start_idx:end_idx])
                kwargs.update(dict(mask=inp_m))
            if obj_attn_mask is not None:
                inp_o = torch.cat(obj_attn_mask[start_idx:end_idx])
                kwargs.update(dict(obj_attn_mask=inp_o))
            if teacher_obj_assignments is not None:
                # inp_t = torch.cat(teacher_obj_assignments[start_idx: end_idx])
                kwargs.update(dict(teacher_obj_assignments=teacher_obj_assignments))

            _out, _obj_attn = self.backbone(inp_x, **kwargs)
            if start_idx == 0:
                output = _out
                obj_attn = _obj_attn
            else:
                output = torch.cat((output, _out))
                obj_attn = torch.cat((obj_attn, _obj_attn))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        output_ = self.head(output)
        if return_backbone_feat:
            return output, output_
        return output_, obj_attn


def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{"params": regularized}, {"params": not_regularized, "weight_decay": 0.0}]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class PCA:
    """
    Class to  compute and apply PCA.
    """

    def __init__(self, dim=256, whit=0.5):
        self.dim = dim
        self.whit = whit
        self.mean = None

    def train_pca(self, cov):
        """
        Takes a covariance matrix (np.ndarray) as input.
        """
        d, v = np.linalg.eigh(cov)
        eps = d.max() * 1e-5
        n_0 = (d < eps).sum()
        if n_0 > 0:
            d[d < eps] = eps

        # total energy
        totenergy = d.sum()

        # sort eigenvectors with eigenvalues order
        idx = np.argsort(d)[::-1][: self.dim]
        d = d[idx]
        v = v[:, idx]

        print("keeping %.2f %% of the energy" % (d.sum() / totenergy * 100.0))

        # for the whitening
        d = np.diag(1.0 / d**self.whit)

        # principal components
        self.dvt = np.dot(d, v.T)

    def apply(self, x):
        # input is from numpy
        if isinstance(x, np.ndarray):
            if self.mean is not None:
                x -= self.mean
            return np.dot(self.dvt, x.T).T

        # input is from torch and is on GPU
        if x.is_cuda:
            if self.mean is not None:
                x -= torch.cuda.FloatTensor(self.mean)
            return torch.mm(
                torch.cuda.FloatTensor(self.dvt), x.transpose(0, 1)
            ).transpose(0, 1)

        # input if from torch, on CPU
        if self.mean is not None:
            x -= torch.FloatTensor(self.mean)
        return torch.mm(torch.FloatTensor(self.dvt), x.transpose(0, 1)).transpose(0, 1)


def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.
    Arguments
    ---------
    ranks : zerro-based ranks of positive images
    nres  : number of positive images
    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1.0 / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.0
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.0

    return ap


def compute_map(ranks, gnd, kappas=[]):
    """
    Computes the mAP for a given set of returned results.
         Usage:
           map = compute_map (ranks, gnd)
                 computes mean average precsion (map) only
           map, aps, pr, prs = compute_map (ranks, gnd, kappas)
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query
         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.0
    nq = len(gnd)  # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]["ok"])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float("nan")
            prs[i, :] = float("nan")
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]["junk"])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgndj)]

        k = 0
        ij = 0
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while ip < len(pos):
                while ij < len(junk) and pos[ip] > junk[ij]:
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1  # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j])
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, aps, pr, prs


def plot_obj_assignments(
    view1, view2, obj_attn_mask_teacher, obj_attn_mask_student, epoch, output_dir
):
    out_dir = os.path.join(output_dir, "object_assignment_figures", f"epoch{epoch}")
    os.makedirs(out_dir, exist_ok=True)
    # move tensors to cpu for visualization
    view1, view2 = view1.cpu(), view2.cpu()
    obj_attn_mask_teacher = obj_attn_mask_teacher.cpu()
    obj_attn_mask_student = obj_attn_mask_student.cpu()

    invTrans = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
            ),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
        ]
    )
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(invTrans(view1).permute(1, 2, 0))
    plt.axis("off")
    plt.title("Image 1, View 1")

    #  obj_attn_mask_teacher: B x O x N
    num_object_tokens = obj_attn_mask_teacher.size(1)
    obj_attn_mask_teacher = obj_attn_mask_teacher.float().chunk(2)
    v1a = obj_attn_mask_teacher[0][0]
    v1a = v1a.argmax(dim=0).reshape(14, 14).numpy()
    plt.subplot(1, 4, 2)
    ax = plt.gca()
    ax.matshow(v1a, vmin=0, vmax=num_object_tokens - 1)
    # plt.colorbar()
    plt.axis("off")
    ax.set_title("Teacher OA, View 1")
    for i in range(14):
        for j in range(14):
            c = v1a[j, i]
            plt.gca().text(i, j, str(c), va="center", ha="center")

    plt.subplot(1, 4, 3)
    plt.imshow(invTrans(view2).permute(1, 2, 0))
    plt.axis("off")
    plt.title("Image 1, View 2")

    plt.subplot(1, 4, 4)
    v1b = obj_attn_mask_teacher[1][0]
    v1b = v1b.argmax(dim=0).reshape(14, 14).numpy()
    ax = plt.gca()
    ax.matshow(v1b, vmin=0, vmax=num_object_tokens - 1)
    # plt.colorbar()
    plt.axis("off")
    ax.set_title("Teacher OA, View 2")
    for i in range(14):
        for j in range(14):
            c = v1b[j, i]
            plt.gca().text(i, j, str(c), va="center", ha="center")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "object_assignment_t_v1_t_v2.png"))
    plt.close()

    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(invTrans(view1).permute(1, 2, 0))
    plt.axis("off")
    plt.title("Image 1, View 1")

    plt.subplot(1, 4, 2)
    #  obj_attn_mask_teacher: B x O x N
    v1a = obj_attn_mask_teacher[0][0]
    v1a = v1a.argmax(dim=0).reshape(14, 14).cpu().numpy()

    ax = plt.gca()
    ax.matshow(v1a, vmin=0, vmax=num_object_tokens - 1)
    # plt.colorbar()
    plt.axis("off")
    ax.set_title("Teacher OA, View 1")
    for i in range(14):
        for j in range(14):
            c = v1a[j, i]
            plt.gca().text(i, j, str(c), va="center", ha="center")

    plt.subplot(1, 4, 3)
    plt.imshow(invTrans(view1).permute(1, 2, 0))
    plt.axis("off")
    plt.title("Image 1, View 1")

    plt.subplot(1, 4, 4)
    obj_attn_mask_student = obj_attn_mask_student.float().chunk(2)
    v1b = obj_attn_mask_student[0][0]
    v1b = v1b.argmax(dim=0).reshape(14, 14).cpu().numpy()
    ax = plt.gca()
    ax.matshow(v1b, vmin=0, vmax=num_object_tokens - 1)
    # plt.colorbar()
    plt.axis("off")
    ax.set_title("Student OA, View 1")
    for i in range(14):
        for j in range(14):
            c = v1b[j, i]
            plt.gca().text(i, j, str(c), va="center", ha="center")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "object_assignment_t_v1_s_v1.png"))
    plt.close()

    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.imshow(invTrans(view1).permute(1, 2, 0))
    plt.axis("off")
    plt.title("Image 1, View 1")

    plt.subplot(1, 4, 2)
    #  obj_attn_mask_teacher: B x O x N
    v1a = obj_attn_mask_teacher[0][0]
    v1a = v1a.argmax(dim=0).reshape(14, 14).cpu().numpy()

    ax = plt.gca()
    ax.matshow(v1a, vmin=0, vmax=num_object_tokens - 1)
    # plt.colorbar()
    plt.axis("off")
    ax.set_title("Teacher OA, View 1")
    for i in range(14):
        for j in range(14):
            c = v1a[j, i]
            plt.gca().text(i, j, str(c), va="center", ha="center")

    plt.subplot(1, 4, 3)
    plt.imshow(invTrans(view2).permute(1, 2, 0))
    plt.axis("off")
    plt.title("Image 1, View 2")

    plt.subplot(1, 4, 4)
    v1b = obj_attn_mask_student[1][0]
    v1b = v1b.argmax(dim=0).reshape(14, 14).cpu().numpy()
    ax = plt.gca()
    ax.matshow(v1b, vmin=0, vmax=num_object_tokens - 1)
    # plt.colorbar()
    plt.axis("off")
    ax.set_title("Student OA, View 2")
    for i in range(14):
        for j in range(14):
            c = v1b[j, i]
            plt.gca().text(i, j, str(c), va="center", ha="center")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "object_assignment_t_v1_s_v2.png"))
    plt.close()


class Clustering:
    def __init__(
        self,
        args,
        n_tokens,
        n_heads,
        sinkhorn_lambda,
        sinkhorn_iterations=3,
        pos_alpha=(0.2, 0.2),
    ):
        self.patch_size = args.patch_size
        self.n_tokens = n_tokens
        self.pos_alpha = np.linspace(pos_alpha[1], pos_alpha[0], args.epochs)
        self.sinkhorn_lambda = sinkhorn_lambda
        self.sinkhorn_iterations = sinkhorn_iterations
        self.n_heads = n_heads
        self.args = args

    @torch.no_grad()
    def sinkhorn(self, M, r, c, lambda_, iterations):
        P = torch.exp(-lambda_ * M).float()
        P /= reduce(P, "b n k -> b 1 1", reduction="sum")

        # Iterate over the sinkhorn algorithm
        for _ in range(iterations):
            u = reduce(P, "b n k -> b n 1", reduction="sum")
            P *= r / u
            u = reduce(P, "b n k -> b 1 k", reduction="sum")
            P *= c / u
        P = torch.nan_to_num(P, nan=1e-8)
        return P, torch.sum(P * M, dim=[1, 2])

    def compute_assignments(
        self,
        tokens,
        positions,
        k,
        pos_alpha,
        sinkhorn_lambda,
        sinkhorn_iterations,
        use_hard_assignment=True,
    ):
        # Normalize the tokens
        tokens = F.normalize(tokens, dim=-1)

        # Get the dimensions
        b, n, d = tokens.shape

        # Compute the random distribution
        r_uni = torch.ones([b, n, 1], device=self.args.gpu) / n
        r = r_uni
        c = torch.ones([b, 1, k], device=self.args.gpu) / k
        p = r_uni.squeeze()
        index = p.multinomial(num_samples=k, replacement=False)
        index = rearrange(index, "b k -> (b k)")
        index = torch.eye(n, device=index.device)[index].to(tokens.device)
        index = rearrange(index, "(b k) n -> b k n", b=b)

        # Set the initial centroids
        centroids = torch.einsum("b n d, b k n -> b k d", tokens, index)

        assignment = index.permute(0, 2, 1)

        for _ in range(self.args.n_iter):
            # Compute the semantic similarity
            sem_similarity = torch.einsum("b n d, b k d -> b n k", tokens, centroids)

            # Compute the distance matrix
            pos_similarity = torch.sqrt(
                torch.sum(
                    (positions[:, None, :, :] - positions[:, :, None, :]) ** 2, dim=-1
                )
            )
            pos_similarity = torch.einsum(
                "B N n, B n k -> B N n k", pos_similarity, assignment
            )

            tmp = torch.ones_like(pos_similarity)
            tmp[pos_similarity == 0.0] = 0.0
            tmp = tmp.sum(dim=2, keepdim=True)

            pos_similarity[torch.logical_and(pos_similarity == 0.0, tmp != 0.0)] = (
                1e5  # If column is not zero, replace all 0 values with high value
            )
            pos_similarity = einops.reduce(
                pos_similarity, "B N n k -> B N k", reduction="min"
            )

            # If cost is 0, replace with average cost
            avg_cost = pos_similarity.mean(dim=[1, 2], keepdim=True)
            avg_cost = repeat(avg_cost, "b 1 1 -> b n k", k=k, n=n)
            pos_similarity[pos_similarity == 0.0] = avg_cost[pos_similarity == 0.0]
            pos_similarity /= pos_similarity.amax(dim=(-1, -2))[:, None, None]

            # Get the cost
            M = -sem_similarity + pos_alpha * pos_similarity
            M = (M - M.min()) / (M.max() - M.min())

            # Compute the transportation plan and the distance
            assignment, cost = self.sinkhorn(
                M=M, r=r, c=c, lambda_=sinkhorn_lambda, iterations=sinkhorn_iterations
            )

            # Compute the hard assignments
            hard_assignment = torch.max(assignment, dim=-1, keepdim=True).values
            hard_assignment = repeat(hard_assignment, "b n 1 -> b n k", k=k)
            hard_assignment = (assignment == hard_assignment).float()

            if use_hard_assignment:
                assignment = hard_assignment

            # Update c
            if self.args.update_c:
                c = hard_assignment.sum(dim=1, keepdim=True) + 1e-2
                c /= c.sum(dim=-1, keepdim=True)

            # Update the centroids
            centroids = torch.einsum("b n d, b n k -> b k d", tokens, assignment)
            centroids = F.normalize(centroids, dim=-1)

        # Normalize column-wise and view-wise
        assignment = rearrange(assignment, "b (m n) k -> m b n k", m=2)
        assignment_v1, assignment_v2 = assignment.unbind()

        # Normalize hard assignment
        # If a cluster is not present in two views, the normalization will divide by 0
        # If that happens, we just replace the 0 by 1
        # Later on, the centroids originating from that cluster will be discarded anyways
        tmpv1 = assignment_v1.sum(dim=-2, keepdim=True)
        tmpv2 = assignment_v2.sum(dim=-2, keepdim=True)
        tmpv1[tmpv1 == 0.0] = 1.0
        tmpv2[tmpv2 == 0.0] = 1.0

        assignment_v1 = assignment_v1 / tmpv1
        assignment_v2 = assignment_v2 / tmpv2
        assignment = torch.cat([assignment_v1, assignment_v2], dim=1)
        return assignment, cost, index

    def compute_student_centroids(self, assignments, tokens, valid_centroids):
        # Reshape the tokens
        tokens = rearrange(tokens, "(m b) n d -> m b n d", m=2)

        # Compute the centroids
        centroids = torch.einsum("m b n d, m b h n k -> m b h k d", tokens, assignments)
        centroids = rearrange(centroids, "m b h k d -> m (b h k) d")

        # Split the centroids view-wise
        centroids_v1, centroids_v2 = centroids.unbind()
        centroids_v1, centroids_v2 = (
            centroids_v1[valid_centroids],
            centroids_v2[valid_centroids],
        )
        return centroids_v1, centroids_v2

    def compute_teacher_centroids(
        self, last_tokens, positions, epoch, use_hard_assignment=True
    ):
        with torch.autocast(device_type="cuda", enabled=False):
            tokens_to_compute_assignment = rearrange(
                last_tokens, "(m b h) n d -> (b h) (m n) d", m=2, h=self.n_heads
            )
            last_tokens = rearrange(last_tokens, "(m b) n d -> m b n d", m=2)

            # tokens_to_compute_assignment and last_tokens are the same thing, only with different shapes

            # Compute area of views
            tmp = positions[:, :, :, [0, -1]]
            diff = torch.abs(tmp[:, :, :, 0] - tmp[:, :, :, 1])
            area = torch.prod(diff, dim=-1)

            # Patchify positional encodings
            positions = rearrange(positions, "m b d n -> b (m n) d")
            positions = repeat(positions, "b n d -> (b h) n d", h=self.n_heads)

            # Compute the assignments
            assignments, _, _ = self.compute_assignments(
                tokens_to_compute_assignment,
                positions,
                self.n_tokens,
                self.pos_alpha[epoch],
                self.sinkhorn_lambda,
                self.sinkhorn_iterations,
                use_hard_assignment,
            )

            # ============================ Split the cluster view-wise ===========================================
            # Each token belongs to a single cluster
            hard_assignments = torch.max(assignments, dim=-1, keepdim=True).values
            hard_assignments = repeat(
                hard_assignments, "b n 1 -> b n k", k=assignments.shape[-1]
            )
            hard_assignments = (assignments == hard_assignments).float()
            hard_assignments = rearrange(hard_assignments, "b (m n) k -> m b n k", m=2)
            assignments = rearrange(
                assignments, "(b h) (m n) k -> m b h n k", m=2, h=self.n_heads
            )

            # Compute the centroids of each view and normalize the assignments
            centroids = torch.einsum(
                "m b n d, m b h n k -> m b h k d", last_tokens, assignments
            )
            centroids = rearrange(centroids, "m b h k d -> m (b h k) d")
            centroids_v1, centroids_v2 = centroids.unbind()

            # Discard a cluster if it's empty in either view
            hard_assignments_v1, hard_assignments_v2 = rearrange(
                hard_assignments, "m b n k -> m (b k) n"
            ).unbind()
            valid_centroids = torch.logical_and(
                (hard_assignments_v1.sum(dim=-1) > 0),
                (hard_assignments_v2.sum(dim=-1) > 0),
            )
            centroids_v1, centroids_v2 = (
                centroids_v1[valid_centroids],
                centroids_v2[valid_centroids],
            )

            # Correct the number of centroids per view
            centroids_per_view = torch.tensor_split(
                valid_centroids, tokens_to_compute_assignment.shape[0]
            )
            centroids_per_view = torch.stack([cpv.sum() for cpv in centroids_per_view])

            # Count the average number of regions
            region_count = centroids_per_view.float().mean().item()
            return (
                (centroids_v1, centroids_v2),
                valid_centroids,
                assignments,
                region_count,
                centroids_per_view,
            )
