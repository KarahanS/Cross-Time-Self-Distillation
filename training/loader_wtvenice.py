# wt_venice_clip_dataset.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict, Any, Sequence, Optional, Iterable, Set
import json, math, random, subprocess, glob, os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision import tv_tensors
from tqdm import tqdm
from torchvision import datasets, transforms
import utils_ibot


# ---------------------------------------------------------------------------
# OPTIONAL: helpers reused from your VidOR loader – Patchify & mask generator
# ---------------------------------------------------------------------------


class Patchify(torch.nn.Module):
    def __init__(self, patch_size=16):
        super().__init__()
        self.p = patch_size
        self.unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        bs, o, h, w = x.shape
        x = self.unfold(x)
        a = x.view(bs, o, self.p, self.p, -1).permute(0, 4, 1, 2, 3)
        return a  # (B, L, O, p, p)


class DataAugmentationiBOT(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        global_crops_number,
        local_crops_number,
    ):
        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale

        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.global_crops_number = global_crops_number
        # transformation for the first global crop
        self.global_transfo1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, scale=global_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                utils_ibot.GaussianBlur(1.0),
                normalize,
            ]
        )
        # transformation for the rest of global crops
        self.global_transfo2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, scale=global_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                utils_ibot.GaussianBlur(0.1),
                utils_ibot.Solarization(0.2),
                normalize,
            ]
        )
        # transformation for the local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    96, scale=local_crops_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                utils_ibot.GaussianBlur(p=0.5),
                normalize,
            ]
        )

    def __call__(self, image, disable_local: bool = False) -> List[torch.Tensor]:
        crops = []
        crops.append(self.global_transfo1(image))
        for _ in range(self.global_crops_number - 1):
            crops.append(self.global_transfo2(image))

        if not disable_local:
            for _ in range(self.local_crops_number):
                crops.append(self.local_transfo(image))
        return crops


# ------------------------- WT-Venice Frame Dataset --------------------------
class WTVeniceFrameDataset(Dataset):
    """
    Each directory under `clips_root/` is a clip containing exactly two JPGs.
    A *single* 300×300 crop is sampled per clip and applied to both frames.

    Returns (default, `need_neigh=False`)
        views, 0, mim_masks, []            # like your VidOR loader

    Returns (`need_neigh=True`)
        views, 0, mim_masks, neigh_views   # neighbour uses SAME crop box
    """

    def __init__(
        self,
        clips_root: str | Path,
        transform,  # your DataAugmentationiBOT instance
        pred_ratio,
        pred_ratio_var,
        pred_start_epoch=0,
        pred_aspect_ratio: Tuple[float, float] = (0.3, 3.0),
        bb_margin=1,
        bb_margin_strategy="fixed",
        pred_shape="block",
        crop_size: int = 300,  # aggressive cropping as in the DoRA paper
        need_neighbor: bool = False,
        patch_size: int = 16,
        neigh_mim: bool = False,
        static_crop: bool = False,
        resize_first: bool = True,
        resize_short_side: int = 640,
    ):
        super().__init__()
        self.root = Path(clips_root)
        self.transform = transform
        self.crop_size = crop_size
        self.need_neighbor = need_neighbor
        self.neigh_mim = neigh_mim
        self.patch_size = patch_size
        self.patchify = Patchify(patch_size)
        self.static_crop = static_crop
        self.resize_first = resize_first
        self.resize_short_side = resize_short_side

        # ---------- gather all clips ---------------------------------------
        self.clips: List[Tuple[Path, List[Path]]] = []
        for clip_dir in sorted(self.root.iterdir()):
            images = sorted(p for p in clip_dir.glob("*.jpg"))
            if len(images) >= 1:  # tolerate 1-frame clips
                self.clips.append((clip_dir, images))
        # build a flat list of (clip_id, frame_idx) so every frame is indexable
        self.frames: List[Tuple[int, int]] = [
            (cid, k)
            for cid, (_, imgs) in enumerate(self.clips)
            for k in range(len(imgs))
        ]
        print(
            f"[WTVenice] {len(self.frames)} individual frames, "
            f"{len(self.clips)} clips found under {self.root}"
        )

        self.pred_ratio = (
            pred_ratio[0]
            if isinstance(pred_ratio, list) and len(pred_ratio) == 1
            else pred_ratio
        )
        self.pred_ratio_var = (
            pred_ratio_var[0]
            if isinstance(pred_ratio_var, list) and len(pred_ratio_var) == 1
            else pred_ratio_var
        )
        if isinstance(self.pred_ratio, list) and not isinstance(
            self.pred_ratio_var, list
        ):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(math.log, pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch
        self.epoch = 0

    # -----------------------------------------------------------------------
    def __len__(self):
        return len(self.frames)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def get_pred_ratio(self) -> float:
        if self.epoch < self.pred_start_epoch:
            return 0
        if isinstance(self.pred_ratio, list):
            choices = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                choices.append(random.uniform(prm - prv, prm + prv) if prv > 0 else prm)
            return random.choice(choices)
        else:
            return (
                random.uniform(
                    self.pred_ratio - self.pred_ratio_var,
                    self.pred_ratio + self.pred_ratio_var,
                )
                if self.pred_ratio_var > 0
                else self.pred_ratio
            )

    def _sample_box(self, W: int, H: int, clip_id: int) -> Tuple[int, int, int, int]:
        """
        Return (left, top, right, bottom) of a 300×300 crop.
        If static_crop=True the choice is deterministic w.r.t. clip_id.
        """
        rng = random.Random(clip_id) if self.static_crop else random
        left = rng.randint(0, W - self.crop_size)
        top = rng.randint(0, H - self.crop_size)
        return (left, top, left + self.crop_size, top + self.crop_size)

    def _make_mask(self, H: int, W: int) -> np.ndarray:
        high = int(round(self.get_pred_ratio() * H * W))
        if self.pred_shape == "block":
            # following BEiT (https://arxiv.org/abs/2106.08254), see at
            # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
            mask = np.zeros((H, W), dtype=bool)
            mask_count = 0
            while mask_count < high:
                max_mask_patches = high - mask_count

                delta = 0
                for attempt in range(10):
                    low = (min(H, W) // 3) ** 2
                    target_area = random.uniform(low, max_mask_patches)
                    aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                    h = int(round(math.sqrt(target_area * aspect_ratio)))
                    w = int(round(math.sqrt(target_area / aspect_ratio)))
                    if w < W and h < H:
                        top = random.randint(0, H - h)
                        left = random.randint(0, W - w)

                        num_masked = mask[top : top + h, left : left + w].sum()
                        if 0 < h * w - num_masked <= max_mask_patches:
                            for i in range(top, top + h):
                                for j in range(left, left + w):
                                    if mask[i, j] == 0:
                                        mask[i, j] = 1
                                        delta += 1

                    if delta > 0:
                        break

                if delta == 0:
                    break
                else:
                    mask_count += delta

        elif self.pred_shape == "rand":
            mask = np.hstack(
                [
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]
            ).astype(bool)
            np.random.shuffle(mask)
            mask = mask.reshape(H, W)

        else:
            # no implementation
            assert False
        return mask

    def __getitem__(self, idx):
        cid, fid = self.frames[idx]
        clip_dir, imgs = self.clips[cid]
        img = Image.open(imgs[fid]).convert("RGB")

        # crop the image
        W, H = img.size
        if W < self.crop_size or H < self.crop_size:
            raise RuntimeError(f"Frame too small for {self.crop_size}×{self.crop_size}")

        if self.resize_first:
            # resize the image to a minimum size before cropping
            if W < self.resize_short_side or H < self.resize_short_side:
                scale = self.resize_short_side / min(W, H)
                new_size = (int(W * scale), int(H * scale))
                img = img.resize(new_size, Image.BICUBIC)
                W, H = img.size
        else:
            # aggressive cropping as in the DoRA paper
            box = self._sample_box(W, H, cid)
            img = img.crop(box)

        views = self.transform(img)
        G = self.transform.global_crops_number

        mim_masks = []
        for v in range(G):
            H_img, W_img = views[v].shape[-2:]
            H_p, W_p = H_img // self.patch_size, W_img // self.patch_size
            mim_mask = self._make_mask(H_p, W_p)
            mim_masks.append(torch.as_tensor(mim_mask, dtype=torch.bool))

        nviews, nmim = [], []
        if self.need_neighbor and len(imgs) > 1:
            # choose the “other” frame in a two-image clip
            other_fid = 1 if fid == 0 else 0  # 2-frame clip
            other_img = Image.open(imgs[other_fid]).convert("RGB")

            if self.resize_first:
                # resize the image to a minimum size before cropping
                if W < self.resize_short_side or H < self.resize_short_side:
                    scale = self.resize_short_side / min(W, H)
                    new_size = (int(W * scale), int(H * scale))
                    other_img = other_img.resize(new_size, Image.BICUBIC)
                    W, H = other_img.size
            else:
                other_img = other_img.crop(box)  # crop the same box
            nviews = self.transform(other_img, disable_local=True)
            if self.neigh_mim:
                for v in range(G):
                    # mimic the same mask as for the original view
                    H_img, W_img = nviews[v].shape[-2:]
                    H_p, W_p = H_img // self.patch_size, W_img // self.patch_size
                    mim_mask = self._make_mask(H_p, W_p)
                    nmim.append(torch.as_tensor(mim_mask, dtype=torch.bool))

        return views, 0, mim_masks, nviews, nmim  # 0 - dummy label
