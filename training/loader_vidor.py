###############################################################################
#  Clip-index-based VidOR → iBOT/ODIS dataset
#
#  • Every line of  the index file is
#        {"video": "1025/5159741010.mp4", "frames": [0,1]}      # example
#    The path is *relative* to   frames_root/.
#  • The class guarantees:
#        – same #items and same (video, frame-ids) order every epoch
#        – every clip already has ≥1 object track that survives all frames
#          ⇒ no more recursive “resample” calls (prior version)
#
#  Returned dict keys, tensor shapes, and the collate function are unchanged,
#  so the rest of your training script stays as-is.
###############################################################################
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

# -------------  helpers reused from the old loader --------------------------
from contextlib import contextmanager


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


@contextmanager
def rng_seed_everywhere(seed: int):
    state_py = random.getstate()
    state_np = np.random.get_state()
    state_torch = torch.random.get_rng_state()
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(state_py)
        np.random.set_state(state_np)
        torch.random.set_rng_state(state_torch)


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


def _build_frame_index(
    trajs: List[List[Dict[str, Any]]],
) -> List[Dict[int, Tuple[int, int, int, int]]]:
    return [
        {obj["tid"]: tuple(obj["bbox"].values()) for obj in frame} for frame in trajs
    ]


def _get_fps(meta) -> int:
    raw = meta.get("fps", 30)
    return max(int(round(float(raw))), 1)


# ----------------------------------------------------------------------------


class VidORiBOTFrameDataset(torch.utils.data.Dataset):
    """
    Return ONE anchor frame. If time-image loss is enabled, also return
    ONE neighbour frame (prev or next) from the same clip.

    Output (per __getitem__):
        {
            "anchor_views"   : list[Tensor]        # len = G+L
            "neigh_views"    : list[Tensor] | None # len = G   or  None
            "anchor_labels"  : int                 # class id (for k-NN)
        }
    """

    def __init__(
        self,
        data_path: str,
        boxes_root: str,
        transform,  # DataAugmentationODISPos
        clip_len: int = 2,
        patch_size: int = 16,
        num_object_per_view: int = 1,
        pred_ratio=0.6,
        pred_ratio_var=0.2,
        pred_aspect_ratio: Tuple[float, float] = (0.3, 3.0),
        pred_shape: str = "block",
        pred_start_epoch: int = 0,
        portion: Optional[float] = 1.0,
        need_neighbour: bool = True,  # if False, no neighbour frame is returned
        neigh_mim: bool = False,
    ):
        super().__init__()
        self.need_neigh = need_neighbour  # toggled once at start-up
        self.patch_size = patch_size

        # ------------------------------------------------------------
        # Load the clip index exactly the same way your current loader
        # does, but FLATTEN it so each frame is one record.
        # ------------------------------------------------------------
        frames_root = Path(data_path) / "images"
        index_file = glob.glob(str(Path(data_path) / "*.jsonl"))[0]

        self.frames_root = Path(frames_root)
        self.ann_root = Path(boxes_root)
        self.index_file = Path(index_file)

        self.transform = transform
        self.clip_len = clip_len
        self.patch_size = patch_size
        self.num_object_per_view = num_object_per_view
        self.patchify = Patchify(patch_size)
        self.portion = portion
        self.neigh_mim = neigh_mim

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

        self.frames = []  # [(video_dir, frame_path, frame_idx_in_clip)]
        self.clip_dict = {}  # video_dir -> [frame_paths] (sorted)

        with open(index_file) as f:
            for line in f:
                rec = json.loads(line)
                video = rec["video"][:-4]  # strip ".mp4"
                vpath = Path(data_path) / "images" / video
                frms = [f"frame_{i:04d}.jpg" for i in rec["frames"]]

                self.clip_dict[str(vpath)] = frms
                for k, fr in enumerate(frms):
                    self.frames.append((vpath, fr, k))

        print(f"[VidORFrameDataset] {len(self.frames)} individual frames loaded")

        if self.portion < 1.0:
            total = len(self.frames)
            total_cnt = int(round(total * self.portion))
            self.frames = random.sample(self.frames, total_cnt)
            print(
                f"[VidORFrameDataset] Using {len(self.frames)} out of {total} videos "
                f"(portion={self.portion})"
            )

    # ------------------------------------------------------------
    def __len__(self):
        return len(self.frames)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _get_pred_ratio(self) -> float:
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

    def _make_mask(self, H: int, W: int) -> np.ndarray:
        high = int(round(self._get_pred_ratio() * H * W))
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

    # ------------------------------------------------------------
    def __getitem__(self, idx):
        vpath, fname, k = self.frames[idx]  # single frame
        full = vpath / fname
        img = Image.open(full).convert("RGB")

        clip = self.clip_dict[str(vpath)]  # in case

        G = self.transform.global_crops_number
        L = self.transform.local_crops_number

        views = []

        H, W = img.size[1], img.size[0]
        if isinstance(self.transform, DataAugmentationiBOT):
            views = self.transform(img)
        else:
            pass

        mim_masks = []
        for v in range(G):  # MIM only for global crops
            # in some of the loaders, they are calculated for local crops as well
            # it is just waste of time because they won't be used anyways.
            H_img, W_img = views[v].shape[-2:]
            H_p, W_p = H_img // self.patch_size, W_img // self.patch_size
            mim_mask = self._make_mask(H_p, W_p)
            mim_masks.append(torch.as_tensor(mim_mask, dtype=torch.bool))

        nviews = []
        nmim_masks = []
        if self.need_neigh:
            nfname = clip[k - 1] if k > 0 else clip[k + 1]  # prev or next
            nfull = vpath / nfname
            nframe = Image.open(nfull).convert("RGB")
            nviews = self.transform(nframe, disable_local=True)

            if self.neigh_mim:
                for v in range(G):
                    H_img, W_img = nviews[v].shape[-2:]
                    H_p, W_p = H_img // self.patch_size, W_img // self.patch_size
                    mim_mask = self._make_mask(H_p, W_p)
                    nmim_masks.append(torch.as_tensor(mim_mask, dtype=torch.bool))

        if self.neigh_mim:
            return views, 0, mim_masks, nviews, nmim_masks  # dummy labels ‘0’
        else:
            return views, 0, mim_masks, nviews, []


class VidORPrebuiltClipDataset(Dataset):
    """
    Dataset that *replays* the clips described in a JSONL index.

    Parameters
    ----------
    index_file   : path to vidor_clip_index.jsonl (output of the builder)
    frames_root  : root that contains  <folder>/<video_id>/frame_XXXX.jpg
    ann_roots    : list of roots that contain the original VidOR json files
                   (training & validation)
    transform    : DataAugmentationODISPos  (same as before)
    clip_len     : how many frames each clip *must* contain (sanity check)
    """

    def __init__(
        self,
        data_path: str,
        boxes_root: str,
        transform,  # DataAugmentationODISPos
        clip_len: int = 2,
        patch_size: int = 16,
        num_object_per_view: int = 1,
        pred_ratio=0.6,
        pred_ratio_var=0.2,
        pred_aspect_ratio: Tuple[float, float] = (0.3, 3.0),
        pred_shape: str = "block",
        pred_start_epoch: int = 0,
        portion: Optional[float] = 1.0,
        image_mode: bool = False,  # ← for baseline
    ):
        super().__init__()

        frames_root = Path(data_path) / "images"

        index_file = glob.glob(str(Path(data_path) / "*.jsonl"))[0]

        self.frames_root = Path(frames_root)
        self.ann_root = Path(boxes_root)
        self.index_file = Path(index_file)

        self.transform = transform
        self.clip_len = 1 if image_mode else clip_len
        self.patch_size = patch_size
        self.num_object_per_view = num_object_per_view
        self.patchify = Patchify(patch_size)
        self.portion = portion

        self.frame_id = {}  # (str(vpath), fname) -> int id
        self.rev_id = []  # id -> (vpath, fname)
        for idx, (vpath, fr, k) in enumerate(self.frames):
            self.frame_id[(str(vpath), fr)] = idx
            self.rev_id.append((str(vpath), fr))

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

        self.clips: List[Tuple[str, List[int]]] = []  # (video, frames)
        with open(index_file) as f:
            for line in f:
                rec = json.loads(line)
                video = rec["video"][:-4]
                frames = rec["frames"]
                ann_json = video + ".json"
                # convert to absolute paths
                ann = self.ann_root / ann_json
                video = Path(data_path) / "images" / video  # remove .mp4
                frames = [f"frame_{fi:04d}.jpg" for fi in frames]
                frames = [video / fi for fi in frames]

                meta = json.load(open(ann))
                bboxes = _build_frame_index(meta["trajectories"])

                local_bboxes = {}

                # find only the bboxes for our frames
                # bboxes is a list of dicts, one per frame
                # frame 0001 should correspond to bboxes[0]
                # NOTE: In LUMI, we have zero-based indexing for frames
                # NOTE: In local, it doesn't matter

                for frame in frames:
                    frame_idx = int(frame.stem.split("_")[-1])
                    # frame_idx -= 1  # convert to 0-based index
                    if frame_idx >= len(bboxes):
                        assert False, (
                            f"Frame {frame} index {frame_idx} is out of bounds "
                            f"(max index {len(bboxes) - 1})"
                        )
                    local_bboxes[frame.stem] = bboxes[frame_idx]
                if image_mode:
                    # if image_mode, we need to convert the frames to a single clip
                    for fr in frames:
                        stem = fr.stem
                        self.clips.append((video, [fr], {stem: local_bboxes[stem]}))
                else:
                    self.clips.append((video, frames, local_bboxes))

        print(
            f"Loaded {len(self.clips)} clip entries from {index_file}, resulting in {len(self.clips) * clip_len} number of images in total"
        )

        # apply portion:
        if self.portion < 1.0:
            total = len(self.clips)
            total_cnt = int(round(total * self.portion))
            self.clips = random.sample(self.clips, total_cnt)
            print(
                f"Using {len(self.clips)} out of {total} videos "
                f"(portion={self.portion})"
            )

    # ..................................................................
    def __len__(self):
        return len(self.clips)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def _get_pred_ratio(self) -> float:
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

    def _mim_mask(self, H: int, W: int) -> np.ndarray:
        high = self._get_pred_ratio() * H * W
        if self.pred_shape == "block":
            mask = np.zeros((H, W), dtype=bool)
            filled = 0
            while filled < high:
                max_fill = high - filled
                delta = 0
                for _ in range(10):
                    low = (min(H, W) // 3) ** 2
                    area = random.uniform(low, max_fill)
                    aspect = math.exp(random.uniform(*self.log_aspect_ratio))
                    h = int(round(math.sqrt(area * aspect)))
                    w = int(round(math.sqrt(area / aspect)))
                    if h < H and w < W:
                        top = random.randint(0, H - h)
                        left = random.randint(0, W - w)
                        overlap = mask[top : top + h, left : left + w].sum()
                        if 0 < h * w - overlap <= max_fill:
                            mask[top : top + h, left : left + w] = True
                            delta = h * w - overlap
                    if delta > 0:
                        break
                if delta == 0:
                    break
                filled += delta
            return mask
        else:  # random
            flat = np.hstack(
                [np.zeros(int(H * W - high), bool), np.ones(int(high), bool)]
            )
            np.random.shuffle(flat)
            return flat.reshape(H, W)

    def _boxes_to_patchmask(
        self, boxes: torch.Tensor, H_img: int, W_img: int
    ) -> torch.Tensor:
        H_p, W_p = H_img // self.patch_size, W_img // self.patch_size
        if boxes.numel() == 0:
            return torch.zeros(self.num_object_per_view, H_p, W_p, dtype=torch.bool)

        assert boxes.size(0) <= self.num_object_per_view
        dense = torch.zeros((1, boxes.size(0), H_img, W_img), dtype=torch.float32)
        for j, (x1, y1, x2, y2) in enumerate(boxes.int()):
            dense[0, j, y1 : y2 + 1, x1 : x2 + 1] = 1.0
        occ = self.patchify(dense).squeeze(0).flatten(2).any(-1).permute(1, 0)
        patchmask = occ.view(boxes.size(0), H_p, W_p)
        if boxes.size(0) < self.num_object_per_view:
            pad = torch.zeros(
                self.num_object_per_view - boxes.size(0), H_p, W_p, dtype=torch.bool
            )
            patchmask = torch.cat([patchmask, pad], 0)
        return patchmask

    # ..................................................................
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        vid_path, frames, local_bboxes = self.clips[idx]

        imgs = [Image.open(fi).convert("RGB") for fi in frames]
        bboxes = [local_bboxes[fi.stem] for fi in frames]

        # all clips in index are guaranteed valid, but we double-check
        common_tids = set.intersection(*[set(fr.keys()) for fr in bboxes])

        assert (
            len(common_tids) > 0
        ), f"Clip {vid_path.stem} {frames}: no common object (index corrupted)"

        if (
            len(common_tids) > self.num_object_per_view and self.num_object_per_view > 0
        ):  # if we want to sample objects
            # area-weighted sampling on 1st frame
            areas = torch.tensor(
                [
                    (bboxes[0][tid][2] - bboxes[0][tid][0])
                    * (bboxes[0][tid][3] - bboxes[0][tid][1])
                    for tid in common_tids
                ],
                dtype=torch.float,
            )
            probs = areas / areas.sum()
            sel = torch.multinomial(probs, self.num_object_per_view, False)
            common_tids = [list(common_tids)[i] for i in sel.tolist()]
        else:
            common_tids = list(common_tids)
        tracked = common_tids

        # “boxes”  as tensor per frame, ordered as tracked
        boxes = [
            torch.tensor([fr[tid] for tid in tracked], dtype=torch.float32)
            for fr in bboxes
        ]

        # ------------- augmentation (per-frame or consistent) -------------
        G = self.transform.global_crops_number
        L = self.transform.local_crops_number
        V_per_frame = G + L

        aug_results = []
        for img, bx in zip(imgs, boxes):
            H, W = img.size[1], img.size[0]
            bx_t = tv_tensors.BoundingBoxes(bx, format="XYXY", canvas_size=(H, W))

            # len(crops) = len(pos_list) = G + L
            # len(gc_boxes) = G
            # crops[0] = [B, 3, 224, 224]
            # gc_boxes[0] = [B, O, 4]
            # pos_list[0] = [B, 2, 224, 224]

            from loader import DataAugmentationODIS, DataAugmentationODISPos

            if isinstance(self.transform, DataAugmentationiBOT):
                crops = self.transform(img)
                gc_boxes = [torch.zeros(0, 4)] * G
                pos_list = [torch.zeros(2, 224, 224)] * (G + L)

                aug_results.append(
                    (crops, gc_boxes, pos_list)
                )  # crops, global boxes, positions

            elif isinstance(self.transform, DataAugmentationODIS):
                crops, gc_boxes = self.transform(img, bx_t)
                pos_list = [torch.zeros(2, 224, 224)] * (G + L)

                aug_results.append((crops, gc_boxes, pos_list))
            elif isinstance(self.transform, DataAugmentationODISPos):
                crops, gc_boxes, pos_list = self.transform(img, bx_t)
                aug_results.append((crops, gc_boxes, pos_list))
            else:
                raise ValueError(f"Unsupported transform type: {type(self.transform)}")

        images, crop_boxes, obj_masks, mim_masks, crops_pos, obj_present = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for (crops, gc_boxes, pos_list), bx in zip(aug_results, boxes):
            assert len(crops) == V_per_frame and len(gc_boxes) == G
            for v in range(V_per_frame):
                crop_tensor = crops[v]
                pos_tensor = pos_list[v]

                H_img, W_img = crop_tensor.shape[1:]
                H_p, W_p = H_img // self.patch_size, W_img // self.patch_size

                present = torch.zeros(self.num_object_per_view, dtype=torch.bool)
                if v < G:
                    bxs_view = gc_boxes[v]
                    Nv = min(bxs_view.size(0), self.num_object_per_view)
                    present[:Nv] = (bxs_view[:Nv, 2] > bxs_view[:Nv, 0]) & (
                        bxs_view[:Nv, 3] > bxs_view[:Nv, 1]
                    )
                else:
                    bxs_view = torch.zeros((0, 4), dtype=torch.float32)

                # object-attention mask
                if v < G and bxs_view.numel() > 0:
                    obj_patch = self._boxes_to_patchmask(
                        bxs_view, H_img, W_img
                    ).flatten(1)
                    token_part = torch.cat(
                        [
                            torch.ones(self.num_object_per_view, 1, dtype=torch.bool),
                            torch.eye(self.num_object_per_view, dtype=torch.bool),
                        ],
                        1,
                    )
                    attn_mask = torch.cat([token_part, obj_patch], 1)
                else:  # local view or global without boxes
                    attn_mask = torch.ones(
                        (1, 1 + self.num_object_per_view + H_p * W_p), dtype=torch.bool
                    )

                mim = self._mim_mask(H_p, W_p)

                images.append(crop_tensor)
                crop_boxes.append(bxs_view)
                obj_masks.append(attn_mask)
                mim_masks.append(mim)
                crops_pos.append(pos_tensor)
                obj_present.append(present)

        return dict(
            video=vid_path,
            frames=[fi.stem for fi in frames],  # remove .jpg
            images=images,
            boxes=crop_boxes,
            obj_masks=obj_masks,
            mim_masks=mim_masks,
            crops_pos=crops_pos,
            obj_present=obj_present,
        )


# ----------------------------------------------------------------------------
# collate function
# ----------------------------------------------------------------------------
def collate_ibot_video(batch):
    V = len(batch[0]["images"])
    images, obj_masks, mim_masks, crop_pos, obj_present = (
        [[] for _ in range(V)] for _ in range(5)
    )
    for sample in batch:
        for v in range(V):
            images[v].append(sample["images"][v])
            obj_masks[v].append(sample["obj_masks"][v])
            mim_masks[v].append(
                torch.as_tensor(sample["mim_masks"][v], dtype=torch.bool)
            )
            crop_pos[v].append(sample["crops_pos"][v])
            obj_present[v].append(sample["obj_present"][v])

    images = [torch.stack(lst, 0) for lst in images]  # [B, 3, 224, 224]
    obj_masks = [torch.stack(lst, 0) for lst in obj_masks]  # [B, O, 1+O+H*W]
    mim_masks = [torch.stack(lst, 0) for lst in mim_masks]  # [B, 14, 14]
    crop_pos = [torch.stack(lst, 0) for lst in crop_pos]  # [B, 2, 224, 224]
    obj_present = [torch.stack(lst, 0) for lst in obj_present]  # [B, O]
    img_labels = torch.zeros(len(batch), dtype=torch.long)  # dummy

    return images, img_labels, obj_masks, mim_masks, crop_pos, obj_present


# --------------------------------------------------------------------
#  vidor_image_mask.py
# --------------------------------------------------------------------
import math, random, numpy as np, torch
from typing import List, Tuple


class VidORImageMask(torch.utils.data.Dataset):
    # This dataset can be used by iBOT for single-frame runs.
    """
    Makes VidOR behave like ImageFolderMask for single-frame iBOT runs.
    Every sample   →  ( views, label, masks )
    * label is always 0 (unsupervised)
    * masks is a list[BoolTensor] – one per global crop
    """

    def __init__(
        self,
        data_path: str,
        boxes_root: str,
        transform,
        patch_size: int,
        pred_ratio,
        pred_ratio_var,
        pred_aspect_ratio=(0.3, 1 / 0.3),
        pred_shape="block",
        pred_start_epoch=0,
    ):

        # --- the “real” dataset ---------------------------------------
        self.inner = VidORPrebuiltClipDataset(
            data_path=data_path,
            boxes_root=boxes_root,
            transform=transform,
            clip_len=1,  # ignored in image_mode
            image_mode=True,  # <- single frame
            patch_size=patch_size,
            num_object_per_view=1,
        )

        self.psz = patch_size
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
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch
        self.epoch = 0

    # ---------- delegations -------------------------------------------
    def __len__(self):
        return len(self.inner)

    def set_epoch(self, e: int):
        self.epoch = e
        self.inner.set_epoch(e)

    # ---------- helper copied from ImageFolderMask --------------------
    def get_pred_ratio(self):
        if hasattr(self, "epoch") and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = (
                random.uniform(
                    self.pred_ratio - self.pred_ratio_var,
                    self.pred_ratio + self.pred_ratio_var,
                )
                if self.pred_ratio_var > 0
                else self.pred_ratio
            )

        return pred_ratio

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

    # ---------- main entry --------------------------------------------
    def __getitem__(
        self, idx: int
    ) -> Tuple[List[torch.Tensor], int, List[torch.Tensor]]:
        sample = self.inner[idx]
        views: List[torch.Tensor] = sample["images"]  # len = G+L
        masks = []

        for v in range(self.inner.transform.global_crops_number):
            H_img, W_img = views[v].shape[-2:]
            H_p, W_p = H_img // self.psz, W_img // self.psz
            m_np = self._make_mask(H_p, W_p)  # (H_p , W_p)
            masks.append(torch.as_tensor(m_np, dtype=torch.bool))

        return views, 0, masks  # dummy labels ‘0’


class VidORImageFolder(Dataset):
    """
    A drop-in replacement for torchvision.datasets.ImageFolder that reads the
    **frame index JSONL** produced by the VidOR clip-builder and treats *every
    single frame* as one training sample.

    •   `transform` must be the usual DINO multi-crop transform
        (DataAugmentationDINO).
    •   The label is always 0 – DINO never uses it, it just has to exist.
    """

    def __init__(
        self,
        data_path: str,  # same as --data_path in DINO
        transform,  # DataAugmentationDINO(...)
        portion: float = 1.0,  # optional subsampling
    ):
        super().__init__()
        self.transform = transform

        # ------------------------------------------------------------------
        # 1) locate <data_path>/images/  and  *.jsonl (frame index file)
        # ------------------------------------------------------------------
        frames_root = Path(data_path) / "images"
        index_files = list(Path(data_path).glob("*.jsonl"))
        if not index_files:
            raise RuntimeError(f"No *.jsonl index file found in {data_path}")
        index_file = index_files[0]

        # ------------------------------------------------------------------
        # 2) build the flat list  self.samples  containing every frame path
        # ------------------------------------------------------------------
        samples: List[Path] = []
        with open(index_file) as f:
            for line in f:
                rec = json.loads(line)
                # "video": "1025/5159741010.mp4", "frames": [0, 1, 8, ...]
                vid_rel = rec["video"][:-4]  # drop .mp4
                for fid in rec["frames"]:
                    frame_name = f"frame_{fid:04d}.jpg"
                    samples.append(frames_root / vid_rel / frame_name)

        if portion < 1.0:
            k = int(round(len(samples) * portion))
            samples = random.sample(samples, k)

        self.samples = samples
        print(
            f"[VidORImageFolder] {len(self.samples):,} frames loaded "
            f"from {index_file}"
        )

    # ----------------------------------------------------------------------
    # torch Dataset API
    # ----------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        crops = self.transform(img)  # list[Tensor] – exactly what DINO needs
        return crops, 0  # dummy label
