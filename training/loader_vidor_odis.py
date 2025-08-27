import os
import random
import math
import json
from PIL import Image
import numpy as np
from requests import patch
import torch
from torchvision.datasets import ImageFolder
import utils
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.transforms import functional as TF
from torchvision import tv_tensors
from scipy.ndimage import label as label_instances

# pycocotools
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils  # already imported
import glob
from pathlib import Path


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    Compute the bounding boxes around the provided masks.

    Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        masks (Tensor[N, H, W]): masks to transform where N is the number of masks
            and (H, W) are the spatial dimensions.

    Returns:
        Tensor[N, 4]: bounding boxes
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

    n = masks.shape[0]

    bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)

    for index, mask in enumerate(masks):
        y, x = torch.where(mask != 0)

        # check if x and y have non-zero size:
        if x.numel() == 0 or y.numel() == 0:
            continue

        bounding_boxes[index, 0] = torch.min(x)
        bounding_boxes[index, 1] = torch.min(y)
        bounding_boxes[index, 2] = torch.max(x)
        bounding_boxes[index, 3] = torch.max(y)

    return bounding_boxes


class DataAugmentationSingleObject(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        global_crops_number,
        local_crops_number,
        sample_strategy="random_area",
        obj_min_area_per=0.04,
        obj_aware_lc_loader=True,
        **kwargs,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.global_crops_number = global_crops_number
        self.sample_strategy = sample_strategy
        self.obj_min_area_per = obj_min_area_per
        self.obj_aware_lc_loader = obj_aware_lc_loader
        #
        color_jitter = transforms.Compose(
            [
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

        # transformation for the first global crop
        self.global_transfo1_rest = transforms.Compose(
            [
                color_jitter,
                utils.GaussianBlur(1.0),
                normalize,
            ]
        )
        # transformation for the rest of global crops
        self.global_transfo2_rest = transforms.Compose(
            [
                color_jitter,
                utils.GaussianBlur(0.1),
                utils.Solarization(0.2),
                normalize,
            ]
        )
        # transformation for the local crops
        self.local_crops_number = local_crops_number
        self.local_transfo_rest = transforms.Compose(
            [
                color_jitter,
                utils.GaussianBlur(p=0.5),
                normalize,
            ]
        )

    def crop_and_flip(
        self,
        image,
        target,
        size,
        scale=(0.08, 1.0),  # default scale values from torchvision
        ratio=(3.0 / 4.0, 4.0 / 3.0),  # default ratio values from torchvision
    ):
        # Random Resized crop
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=scale, ratio=ratio
        )
        image = TF.resized_crop(image, i, j, h, w, size, interpolation=Image.BICUBIC)
        target = TF.resized_crop(
            target, i, j, h, w, size, interpolation=Image.NEAREST
        )  # which interpolation to use for targets?

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            target = TF.hflip(target)
        return image, target

    def __call__(self, image, target, disable_local=False):
        label_offset = 1000
        gc_crops, gc_targets = [], []
        lc_crops, lc_targets = [], []
        # if image is gray scale, convert to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        for _ in range(10):
            crop_gc1, target_gc1 = self.crop_and_flip(
                image, target, (224, 224), scale=self.global_crops_scale
            )
            target_gc1 = TF.to_tensor(target_gc1)
            target_gc1 = target_gc1 * 255
            # no need for further manipulation in VidOR
            gc1_labels, gc1_counts = torch.unique(target_gc1, return_counts=True)
            area = 224 * 224
            min_area = self.obj_min_area_per * area
            # print("gc1_labels: ", gc1_labels)
            # print("gc1_counts: ", gc1_counts)
            # remove unlabeled
            gc1_counts = gc1_counts[gc1_labels != 0]
            gc1_labels = gc1_labels[gc1_labels != 0]
            # remove small objects
            gc1_labels = gc1_labels[gc1_counts > min_area]
            gc1_counts = gc1_counts[gc1_counts > min_area]

            if len(gc1_labels) > 0:
                break

        # print("gc1_labels: ", gc1_labels)
        if len(gc1_labels) > 0:
            if self.sample_strategy == "random":
                idx = torch.randperm(len(gc1_labels))[0]
            elif self.sample_strategy == "random_area":
                idx = torch.argsort(gc1_counts, descending=True)[0]
            else:
                raise ValueError(f"Unknown sample strategy: {self.sample_strategy}")

            gc1_label = gc1_labels[idx]

            gc_crops.append(self.global_transfo1_rest(crop_gc1))
            gc_targets.append(
                torch.where(
                    target_gc1 == gc1_label,
                    torch.ones_like(target_gc1) * gc1_label,
                    torch.zeros_like(target_gc1),
                )
            )
            gc2_added = False
            for i in range(20):
                crop_, target_ = self.crop_and_flip(
                    image, target, (224, 224), scale=self.global_crops_scale
                )
                target_ = TF.to_tensor(target_)
                target_ = target_ * 255
                # no need for further manipulation in VidOR
                crop_labels, _ = torch.unique(target_, return_counts=True)
                if gc1_label in crop_labels:
                    gc_crops.append(self.global_transfo2_rest(crop_))
                    gc_targets.append(
                        torch.where(
                            target_ == gc1_label,
                            torch.ones_like(target_) * gc1_label,
                            torch.zeros_like(target_),
                        )
                    )
                    gc2_added = True
                    # print(f"GC2 FOUND at iteration {i}!!")
                    break
            if not gc2_added:
                # print("GC2 NOT FOUND!!")
                crop_, target_ = self.crop_and_flip(
                    image, target, (224, 224), scale=self.global_crops_scale
                )
                gc_crops.append(self.global_transfo2_rest(crop_))
                gc_targets.append(torch.zeros_like(TF.to_tensor(target_)))

            if self.local_crops_number > 0 and not disable_local:
                if self.obj_aware_lc_loader:
                    num_lc_added = 0
                    for i in range(10 * self.local_crops_number):
                        crop_, target_ = self.crop_and_flip(
                            image, target, (96, 96), scale=self.local_crops_scale
                        )
                        target_ = TF.to_tensor(target_)
                        target_ = target_ * 255
                        # no need for further manipulation in VidOR
                        crop_labels, _ = torch.unique(target_, return_counts=True)
                        if gc1_label in crop_labels:
                            lc_crops.append(self.local_transfo_rest(crop_))
                            # lc_targets.append(torch.where(target_ == gc1_label, torch.ones_like(target_) * gc1_label, torch.zeros_like(target_)))
                            lc_targets.append(torch.ones_like(target_) * gc1_label)
                            num_lc_added += 1
                        if num_lc_added == self.local_crops_number:
                            break

                    # print(f"NUMBER OF LOCAL CROPS ADDED: {num_lc_added}, at iteration: {i}", )
                    if num_lc_added < self.local_crops_number:
                        for _ in range(self.local_crops_number - num_lc_added):
                            crop_, target_ = self.crop_and_flip(
                                image, target, (96, 96), scale=self.local_crops_scale
                            )
                            target_ = TF.to_tensor(target_)
                            target_ = target_ * 255
                            # no need for further manipulation in VidOR
                            crop_labels, _ = torch.unique(target_, return_counts=True)
                            lc_crops.append(self.local_transfo_rest(crop_))
                            # lc_targets.append(torch.where(target_ == gc1_label, torch.ones_like(target_) * gc1_label, torch.zeros_like(target_)))
                            lc_targets.append(torch.ones_like(target_) * gc1_label)
                else:
                    for _ in range(self.local_crops_number):
                        crop_, target_ = self.crop_and_flip(
                            image, target, (96, 96), scale=self.local_crops_scale
                        )
                        target_ = TF.to_tensor(target_)
                        target_ = target_ * 255
                        # no need for further manipulation in VidOR
                        crop_labels, _ = torch.unique(target_, return_counts=True)
                        lc_crops.append(self.local_transfo_rest(crop_))
                        lc_targets.append(torch.ones_like(target_) * gc1_label)

        else:
            crop_, target_ = self.crop_and_flip(
                image, target, (224, 224), scale=self.global_crops_scale
            )
            gc_crops.append(self.global_transfo1_rest(crop_))
            gc_targets.append(torch.zeros_like(TF.to_tensor(target_)))
            #
            for _ in range(self.global_crops_number - 1):
                crop_, target_ = self.crop_and_flip(
                    image, target, (224, 224), scale=self.global_crops_scale
                )
                gc_crops.append(self.global_transfo2_rest(crop_))
                gc_targets.append(torch.zeros_like(TF.to_tensor(target_)))

            if not disable_local:
                for _ in range(self.local_crops_number):
                    crop_, target_ = self.crop_and_flip(
                        image, target, (96, 96), scale=self.local_crops_scale
                    )
                    lc_crops.append(self.local_transfo_rest(crop_))
                    lc_targets.append(torch.zeros_like(TF.to_tensor(target_)))

        assert len(gc_crops) == len(gc_targets)
        assert len(lc_crops) == len(lc_targets)
        # assert (
        #    len(gc_crops) * self.local_crops_number
        #    == len(lc_crops) * self.global_crops_number
        # )
        # assert (
        #    len(gc_targets) * self.local_crops_number
        #    == len(lc_targets) * self.global_crops_number
        # )

        return gc_crops, lc_crops, gc_targets, lc_targets


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


class VidORiBOTFrameDatasetODIS(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        boxes_root,
        transform,
        patch_size,
        num_object_per_view,
        pred_ratio,
        pred_ratio_var,
        pred_aspect_ratio,
        pred_start_epoch=0,
        bb_margin=1,
        bb_margin_strategy="fixed",
        pred_shape="block",
        train=True,
        object_sampling_strategy="random_area",
        min_obj_area=0,
        divide_to_instances=False,
        portion=1.0,
        need_neighbour: bool = False,
        clip_len=2,
        neighbor_masks_disabled: bool = True,
        instance_level: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.need_neigh = need_neighbour  # toggled once at start-up
        self.patch_size = patch_size
        self.divide_to_instances = divide_to_instances
        self.bb_margin = bb_margin
        self.bb_margin_strategy = bb_margin_strategy
        self.min_obj_area = min_obj_area
        self.object_sampling_strategy = object_sampling_strategy
        self.neighbor_masks_disabled = neighbor_masks_disabled
        self.instance_level = instance_level

        if self.instance_level:
            print(
                "[VidORiBOTFrameDatasetODIS] Instance-level mode enabled. "
                "Using trajectory IDs as labels."
            )

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

        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # Build   clip_dict / frames   *and* a universal   cat2id   table
        # ------------------------------------------------------------------
        self.clip_dict = {}  # video_dir  ->  [frame_XXXX.jpg, ...]
        self.frames = []  # flat list of (video_dir, frame_name, k)

        self.cat2id = {}  # category → int   (0 = background)
        next_id = 1

        seen_anns = set()  # avoid opening the same file twice

        with open(index_file) as f:
            for line in f:
                rec = json.loads(line)
                video = rec["video"][:-4]  # "0102/xxx.mp4" → "0102/xxx"
                vpath = Path(data_path) / "images" / video
                frms = [f"frame_{i:04d}.jpg" for i in rec["frames"]]

                # ①  record clip & individual frames
                self.clip_dict[str(vpath)] = frms
                for k, fr in enumerate(frms):
                    self.frames.append((vpath, fr, k))

                # ②  read its annotation json *once* and collect categories
                ann_json = self.ann_root / vpath.parent.name / (vpath.name + ".json")
                if ann_json in seen_anns:
                    continue  # already processed
                seen_anns.add(ann_json)

                meta = json.load(open(ann_json, "r"))
                for sobj in meta.get("subject/objects", []):

                    # (a) category‑mode → label = string category
                    # (b) instance‑mode → label = trajectory id  (int)
                    label = sobj["tid"] if self.instance_level else sobj["category"]
                    if label not in self.cat2id:
                        self.cat2id[label] = next_id
                        next_id += 1

                print(self.cat2id)

        print(f"[VidORFrameDatasetODIS] {len(self.frames)} individual frames loaded")
        print(f"[VidORFrameDatasetODIS] {len(self.clip_dict)} video clips loaded")
        print(f"[VidORFrameDatasetODIS] {len(self.cat2id)} categories found")

        if self.portion < 1.0:
            total = len(self.frames)
            total_cnt = int(round(total * self.portion))
            self.frames = random.sample(self.frames, total_cnt)
            print(
                f"[VidORFrameDatasetODIS] Using {len(self.frames)} out of {total} videos "
                f"(portion={self.portion})"
            )

    # ------------------------------------------------------------
    # helper: do *everything* for ONE frame and return tidy lists
    # ------------------------------------------------------------
    def _process_frame(
        self, img: Image.Image, bbox_dict: dict, lc_disabled: bool, masks_disabled=False
    ):
        H_img, W_img = img.size[1], img.size[0]
        target_np = np.zeros((H_img, W_img), dtype=np.uint8)  # background = 0

        for cat_name, boxes in bbox_dict.items():
            cid = self.cat2id.get(cat_name, 0)  # 0 if truly unseen

            for x1, y1, x2, y2 in boxes:
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                target_np[y1 : y2 + 1, x1 : x2 + 1] = cid  # union over instances

        G = self.transform.global_crops_number
        L = self.transform.local_crops_number

        if isinstance(self.transform, DataAugmentationSingleObject):
            tgt_img = Image.fromarray(target_np.astype(np.uint8))
            gc_crops, lc_crops, gc_targets, lc_targets = self.transform(
                img, tgt_img, disable_local=lc_disabled
            )
        else:
            raise NotImplementedError(
                f"Data augmentation {type(self.transform)} is not implemented."
            )

        masks, gc_labels, lc_labels = [], [], []
        gc_obj_assignments, lc_obj_assignments = [], []

        for i in range(len(gc_crops)):

            H, W = (
                gc_crops[i].shape[1] // self.patch_size,
                gc_crops[i].shape[2] // self.patch_size,
            )

            labels = torch.unique(gc_targets[i].flatten())
            labels = labels[labels != 0]

            if len(labels) == 1:
                dense_obj_masks = gc_targets[i] == labels.item()  # (H_img, W_img)
                dense_obj_masks = self.patchify(
                    dense_obj_masks[None, :, :, :].float()
                ).squeeze(0)
                # a -> ( B no.of patches c p p )
                dense_obj_masks = dense_obj_masks.transpose(0, 1).flatten(2).any(dim=-1)
                dense_obj_masks = dense_obj_masks.view(1, H, W)
                structure = np.ones(
                    (3, 3), dtype=int
                )  # this defines the connection filter
                instance_target, ncomponents = label_instances(
                    dense_obj_masks[0], structure
                )
                if self.divide_to_instances:
                    instance_ids = np.random.permutation(ncomponents)[:1] + 1
                else:
                    instance_ids = np.arange(ncomponents) + 1
                instance_mask = instance_target == instance_ids[:, None, None]
                instance_boxes = masks_to_boxes(torch.from_numpy(instance_mask)).int()
                dense_obj_masks = torch.zeros(1, H, W).to(bool)
                for ii, _ in enumerate(instance_ids):
                    x1, y1, x2, y2 = instance_boxes[ii]

                    if self.bb_margin_strategy == "fixed":
                        y1_margin = y2_margin = x1_margin = x2_margin = self.bb_margin
                    elif self.bb_margin_strategy == "random":
                        y1_margin = random.randint(0, self.bb_margin)
                        y2_margin = random.randint(0, self.bb_margin)
                        x1_margin = random.randint(0, self.bb_margin)
                        x2_margin = random.randint(0, self.bb_margin)
                    else:
                        raise NotImplementedError(
                            f"bb_margin_strategy {self.bb_margin_strategy} not implemented"
                        )

                    dense_obj_masks[
                        0,
                        max(0, y1 - y1_margin) : y2 + 1 + y2_margin,
                        max(0, x1 - x1_margin) : x2 + 1 + x2_margin,
                    ] = 1

                dense_obj_masks = dense_obj_masks.flatten(1)  # (O, H*W)
                dense_obj_masks = torch.cat(
                    [torch.ones(1, 1).to(bool), torch.eye(1).to(bool), dense_obj_masks],
                    dim=1,
                )
                gc_obj_assignments.append(dense_obj_masks)
                gc_labels.append(labels)
            else:
                gc_obj_assignments.append(
                    torch.cat(
                        [torch.ones(1, 2).to(bool), torch.zeros(1, H * W).to(bool)],
                        dim=1,
                    )
                )
                gc_labels.append(torch.zeros(1))

            if not masks_disabled:
                pred_ratio = self.get_pred_ratio()
                high = pred_ratio * H * W
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
                            aspect_ratio = math.exp(
                                random.uniform(*self.log_aspect_ratio)
                            )
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

                masks.append(mask)
                # patch_obj_assignments.append(np.zeros((self.num_object_per_view, H, W)).astype(bool))

        if not lc_disabled:
            for i in range(len(lc_crops)):
                labels = torch.unique(lc_targets[i])
                # print("Labels LEN: ", len(labels))
                labels = labels[labels != 0]  # remove unlabeled
                if len(labels) == 1:
                    lc_labels.append(labels)
                else:
                    lc_labels.append(torch.zeros(1))

                lc_obj_assignments.append(
                    torch.cat(
                        [torch.ones(1, 2).to(bool), torch.ones(1, H * W).to(bool)],
                        dim=1,
                    )
                )
                mask = np.hstack(
                    [
                        np.zeros(H * W - int(high)),
                        np.ones(int(high)),
                    ]
                ).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)
                masks.append(mask)

        return (
            gc_crops,
            lc_crops,
            gc_labels,
            lc_labels,
            gc_obj_assignments,
            lc_obj_assignments,
            masks,
        )

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

    @staticmethod
    def _build_frame_index(trajectories, tid2cat=None):
        """
        index[fid][category]  →  list[ [xmin, ymin, xmax, ymax], ... ]
        """

        if trajectories and isinstance(trajectories[0], list):  # VidOR
            index = []
            for frame_boxes in trajectories:  # for each frame
                frm_dict = {}
                for ann in frame_boxes:  # every bbox in frame
                    key = ann["tid"]
                    if tid2cat is not None:
                        key = tid2cat[key]  # use category label

                    bb = ann["bbox"]
                    bb_xyxy = [bb["xmin"], bb["ymin"], bb["xmax"], bb["ymax"]]

                    frm_dict.setdefault(key, []).append(bb_xyxy)  # ← keep ALL boxes
                index.append(frm_dict)  # may be {}
            return index

    # ------------------------------------------------------------
    def __getitem__(self, idx):
        # (1) locate frame & load image
        vpath, fname, k = self.frames[idx]
        fpath = vpath / fname
        img = Image.open(fpath).convert("RGB")

        parent = vpath.parent.name

        # (2) load the *video* annotation json once
        ann_json = self.ann_root / parent / (vpath.name + ".json")
        if not hasattr(self, "_ann_cache"):
            self._ann_cache = {}

        if ann_json not in self._ann_cache:
            meta = json.load(open(ann_json))

            if self.instance_level:
                bboxes = self._build_frame_index(meta["trajectories"], tid2cat=None)
            else:
                tid2cat_local = {
                    o["tid"]: o["category"] for o in meta["subject/objects"]
                }
                bboxes = self._build_frame_index(meta["trajectories"], tid2cat_local)
            self._ann_cache[ann_json] = bboxes
            # save to a txt file

        frame_index = self._ann_cache[ann_json]
        # frame_index: list[dict]  ->  dict[tid, bbox]
        # each dictionary is for one frame, with keys: tid, bbox

        frame_idx = int(fname.split("_")[-1].split(".")[0])
        bbox_dict = frame_index[frame_idx] if frame_idx < len(frame_index) else {}

        (
            gc_crops,
            lc_crops,
            gc_labels,
            lc_labels,
            gc_obj_assignments,
            lc_obj_assignments,
            masks,
        ) = self._process_frame(img, bbox_dict, lc_disabled=False)

        nviews = None
        nobj_assignments = None
        obj_neighbor_labels = None

        if self.need_neigh:
            # get neighbor frame form the same clip
            # same clip, other index (works for any clip_len)
            frmlist = self.clip_dict[str(vpath)]
            neigh_idx_in_clip = (k + 1) % len(frmlist)
            neigh_fname = frmlist[neigh_idx_in_clip]
            neigh_path = vpath / neigh_fname

            neigh_img = Image.open(neigh_path).convert("RGB")

            frame_idx_n = int(neigh_fname.split("_")[-1].split(".")[0])
            bbox_dict_n = self._ann_cache[ann_json][frame_idx_n]

            (
                gc_crops_neigh,
                lc_crops_neigh,
                gc_labels_neigh,
                lc_labels_neigh,
                gc_obj_assignments_neigh,
                lc_obj_assignments_neigh,
                masks_neigh,  # TODO: Should we use MIM masks for neighbors?
            ) = self._process_frame(
                neigh_img,
                bbox_dict_n,
                lc_disabled=True,
                masks_disabled=self.neighbor_masks_disabled,
            )

            nviews = gc_crops_neigh + lc_crops_neigh
            nobj_assignments = gc_obj_assignments_neigh + lc_obj_assignments_neigh
            obj_neighbor_labels = gc_labels_neigh + lc_labels_neigh

        crops = gc_crops + lc_crops
        labels = gc_labels + lc_labels
        obj_assignments = gc_obj_assignments + lc_obj_assignments

        assert len(crops) == len(labels)
        assert len(crops) == len(obj_assignments)
        assert len(crops) == len(masks)

        if not self.need_neigh:
            nviews = []
            obj_neighbor_labels = []
            nobj_assignments = []

        # crops[0] = [3, 224, 224]
        # labels[0] = [1]
        # obj_assignments[0] = [O, 1+O+196]
        # masks[0] = [14, 14]

        """ 
        images,
        obj_labels,
        obj_assignments,
        mim_masks,
        nimages,
        obj_neighbor_labels,
        nobj_assignments,
        """

        return (
            crops,
            labels,
            obj_assignments,
            masks,
            nviews,
            obj_neighbor_labels,
            nobj_assignments,
            masks_neigh,
        )
