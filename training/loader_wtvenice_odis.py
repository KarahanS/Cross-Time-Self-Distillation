# odis_wtvenice_dataloader.py
# -----------------------------------------------------------
#  A drop-in replacement for VidORiBOTFrameDatasetODIS
#  ----------------------------------------------------------
from __future__ import annotations
import json, math, random, glob
from pathlib import Path
from typing import List, Tuple, Dict, Any, Sequence, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

# ──────────────────  Re-use helper classes from VidOR loader  ────────────────
from loader_vidor_odis import (  # adjust import path
    DataAugmentationSingleObject,
    Patchify,
    masks_to_boxes,
    label_instances,
)

# ─────────────────────────────────────────────────────────────────────────────


class ODISWTVeniceFrameDataset(Dataset):
    """
    Object-level counterpart of WTVeniceFrameDataset (image-level).

    Every __getitem__ returns exactly the *same tuple* as
    VidORiBOTFrameDatasetODIS so that your training loop stays unchanged.

        crops, labels, obj_assignments,neighbor_mim_mask masks,
        nviews, obj_neighbor_labels, nobj_assignments, nmim_masks
    """

    # ───────────────────────────── initialisation ───────────────────────────
    def __init__(
        self,
        data_path: str | Path,  # = ".../dataset"
        transform: DataAugmentationSingleObject,  # object-aware augmenter
        patch_size: int,
        num_object_per_view: int,
        pred_ratio,
        pred_ratio_var,
        pred_aspect_ratio,
        pred_start_epoch: int = 0,
        bb_margin: int = 1,
        bb_margin_strategy: str = "fixed",
        pred_shape: str = "block",
        divide_to_instances: bool = False,
        min_obj_area: int = 0,
        need_neighbor: bool = False,
        clip_len: int = 2,  # irrelevant – clips are tiny
        neighbor_mim_mask: bool = False,
        portion: float = 1.0,
        static_crop: bool = False,
        crop_size: int = 300,
        clever_initial_cropping: bool = False,
        resize_first: bool = True,
        resize_short_side: int = 640,
        **kw,
    ):
        super().__init__()
        self.root = Path(data_path)
        self.transform = transform
        self.patch_size = patch_size
        self.patchify = Patchify(patch_size)
        self.num_object_per_view = num_object_per_view
        self.divide_to_instances = divide_to_instances
        self.bb_margin = bb_margin
        self.bb_margin_strategy = bb_margin_strategy
        self.min_obj_area = min_obj_area
        self.need_neigh = need_neighbor
        self.neighbor_mim_mask = neighbor_mim_mask
        self.static_crop = static_crop
        self.crop_size = crop_size
        self.clever_crop = clever_initial_cropping
        self.resize_first = resize_first
        self.resize_short_side = resize_short_side

        # ----------------------------- gather clips -------------------------
        self.clips: List[Tuple[Path, List[Path]]] = []
        for sub in sorted(self.root.iterdir()):
            if sub.name == "anns" or not sub.is_dir():
                continue
            imgs = sorted(p for p in sub.glob("*.jpg"))
            if imgs:
                self.clips.append((sub, imgs))  # self.clips = (sub_dir, [img_paths])

        # flat frame index  (clip_id, frame_idx)
        self.frames: List[Tuple[int, int]] = [
            (cid, k)
            for cid, (_, imgs) in enumerate(self.clips)
            for k in range(len(imgs))
        ]

        # ------------------------------------------------------------------
        # Pre-build an *annotation cache* so clever-crop look-ups are O(1)
        #   key = Path(frame).name  (e.g. "frame_000001.jpg")
        #   val = list[(class_name, x1,y1,x2,y2), ...]     (float coords)
        # ------------------------------------------------------------------
        self.ann_cache: Dict[str, List[Tuple[str, float, float, float, float]]] = {}
        anns_dir = self.root
        if self.clever_crop and anns_dir.exists():
            for ann_path in anns_dir.glob("*.json"):
                meta = json.load(open(ann_path))
                dets = [
                    (d["class_name"], *d["bbox"]) for d in meta.get("detections", [])
                ]
                self.ann_cache[meta["file_name"]] = dets

        self.cat2id: Dict[str, int] = {}
        next_id = 1
        for ann_path in anns_dir.glob("*.json"):
            meta = json.load(open(ann_path))
            for det in meta.get("detections", []):
                cat = det["class_name"]
                if cat not in self.cat2id:
                    self.cat2id[cat] = next_id
                    next_id += 1
        print(
            f"[WT-Venice-ODIS] {len(self.frames)} frames, "
            f"{len(self.clips)} clips, {len(self.cat2id)} categories"
        )

        # --------------------------- misc training knobs --------------------
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
        self.portion = portion

        if self.portion < 1.0:
            total = len(self.frames)
            keep = int(round(total * self.portion))
            self.frames = random.sample(self.frames, keep)
            print(
                f"[WT-Venice-ODIS] Using {keep}/{total} frames (portion={self.portion})"
            )

        # cache for annotation JSONs
        self._ann_cache: Dict[Path, Dict[str, List[List[float]]]] = {}

    # ──────────────────────── utility helpers ───────────────────────────────
    def __len__(self):
        return len(self.frames)

    def set_epoch(self, e: int):
        self.epoch = e

    def _sample_crop_box(
        self, W: int, H: int, clip_id: int
    ) -> Tuple[int, int, int, int]:
        def _rand_box(rng):
            l = rng.randint(0, W - self.crop_size)
            t = rng.randint(0, H - self.crop_size)
            return l, t, l + self.crop_size, t + self.crop_size

        rng = random.Random(clip_id) if self.static_crop else random
        if not self.clever_crop or len(self.current_clip_ann) < 2:
            return _rand_box(rng)

        # clever cropping: find a box with common categories in two frames
        f1, f2 = self.current_clip_ann[0], self.current_clip_ann[1]
        ann1 = self.ann_cache.get(f1, [])
        ann2 = self.ann_cache.get(f2, [])
        if not ann1 or not ann2:
            return _rand_box(rng)

        last_box = None
        for _ in range(20):
            box = _rand_box(rng)
            last_box = box
            l, t, r, b = box

            def cats_in_crop(dets):
                cats = set()
                for cname, x1, y1, x2, y2 in dets:
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2  # bbox centre
                    if l <= cx <= r and t <= cy <= b:
                        cats.add(cname)
                return cats

            common = cats_in_crop(ann1) & cats_in_crop(ann2)
            if common:
                return box  # success

        return last_box  # give up – use last try

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

    # ------------------------- load & unionise boxes ------------------------
    def _load_ann(self, frame_name: str) -> Dict[str, List[List[float]]]:
        """
        Returns  {class_name: [[x1,y1,x2,y2]]}.  One unionised box per category.
        """
        anns_dir = self.root
        ann_path = anns_dir / f"{Path(frame_name).stem}_pred.json"

        if ann_path not in self._ann_cache:
            if not ann_path.exists():
                self._ann_cache[ann_path] = {}  # no detections
            else:
                meta = json.load(open(ann_path))
                per_cat: Dict[str, List[List[float]]] = {}
                for det in meta.get("detections", []):
                    cat = det["class_name"]
                    bb = det["bbox"]  # [x1,y1,x2,y2] floats
                    per_cat.setdefault(cat, []).append(bb)
                # ------ unionise duplicates --------------------------------
                per_cat = {c: bbs for c, bbs in per_cat.items()}
                self._ann_cache[ann_path] = per_cat
        return self._ann_cache[ann_path]

    # ------------------------------ main worker -----------------------------
    def __getitem__(self, idx: int):
        clip_id, fid = self.frames[idx]
        clip_dir, imgs = self.clips[clip_id]
        self.current_clip_ann = [p.name for p in imgs[:2]]  # at most 2 needed

        fpath = imgs[fid]
        img = Image.open(fpath).convert("RGB")
        W_orig, H_orig = img.size

        if self.resize_first:
            scale = self.resize_short_side / max(W_orig, H_orig)

            target_w = int(round(W_orig * scale))
            target_h = int(round(H_orig * scale))

            img_cropped = img.resize((target_w, target_h), Image.BILINEAR)
            sx, sy = target_w / W_orig, target_h / H_orig
            bbox_dict_raw = self._load_ann(fpath.name)
            bbox_dict = {
                cat: [[x1 * sx, y1 * sy, x2 * sx, y2 * sy] for x1, y1, x2, y2 in boxes]
                for cat, boxes in bbox_dict_raw.items()
            }
            print(f"Processing {fpath.name} ({fid}) with {len(bbox_dict)} categories")
            print("Dict keys:", bbox_dict.keys())

        else:

            # 1) crop -----------------------------------------------------------
            W, H = img.size
            if W < self.crop_size or H < self.crop_size:
                raise RuntimeError(f"{fpath} too small for {self.crop_size}px crop")

            crop_box = self._sample_crop_box(W, H, clip_id)
            img_cropped = img.crop(crop_box)

            # 2) build bbox_dict  {cat_name: [[x1,y1,x2,y2], ...]}
            bbox_dict = self._load_ann(fpath.name)

            print(f"Processing {fpath.name} ({fid}) with {len(bbox_dict)} categories")
            print("Dict keys:", bbox_dict.keys())

        # 3) run the SAME _process_frame implementation you already have
        (gc_crops, lc_crops, gc_labels, lc_labels, gc_assign, lc_assign, masks) = (
            self._process_frame(
                img_cropped,
                bbox_dict,
                lc_disabled=False,
                masks_disabled=False,
            )
        )

        # 4) neighbour (if requested) --------------------------------------
        nviews, nlabels, nassign, nmasks = [], [], [], []
        if self.need_neigh and len(imgs) > 1:
            other_fid = 1 if fid == 0 else 0
            npath = imgs[other_fid]
            nimg = Image.open(npath).convert("RGB")

            if self.resize_first:
                scale = self.resize_short_side / max(W_orig, H_orig)

                target_w = int(round(W_orig * scale))
                target_h = int(round(H_orig * scale))

                nimg = nimg.resize((target_w, target_h), Image.BILINEAR)
                sx, sy = target_w / W_orig, target_h / H_orig
                nbbox_dict_raw = self._load_ann(npath.name)
                nbbox_dict = {
                    cat: [
                        [x1 * sx, y1 * sy, x2 * sx, y2 * sy] for x1, y1, x2, y2 in boxes
                    ]
                    for cat, boxes in nbbox_dict_raw.items()
                }
                print(
                    f"Processing {npath.name} ({other_fid}) with {len(nbbox_dict)} categories"
                )
                print("Dict keys:", nbbox_dict.keys())

            else:
                img_w, img_h = nimg.size
                if img_w < self.crop_size or img_h < self.crop_size:
                    raise RuntimeError(f"{npath} too small for {self.crop_size}px crop")
                nimg = nimg.crop(crop_box)
                nbbox_dict = self._load_ann(npath.name)

            (ngc, nlc, nglbl, nllbl, ngassign, nlassign, nmasks_tmp) = (
                self._process_frame(
                    nimg,
                    nbbox_dict,
                    lc_disabled=True,
                    masks_disabled=not self.neighbor_mim_mask,
                )
            )
            nviews = ngc + nlc
            nlabels = nglbl + nllbl
            nassign = ngassign + nlassign
            nmasks = nmasks_tmp

        # 5) pack & return identical tuple type ----------------------------
        crops = gc_crops + lc_crops
        labels = gc_labels + lc_labels
        assigns = gc_assign + lc_assign

        assert len(crops) == len(labels) == len(assigns) == len(masks)
        return (
            crops,
            labels,
            assigns,
            masks,
            nviews,
            nlabels,
            nassign,
            nmasks,
        )

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

        # print("Categories present:", set(self.cat2id.keys()) & set(bbox_dict.keys()))
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


# ---------------------------------------------------------------------------
# usage example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from loader_vidor_odis import DataAugmentationSingleObject

    aug = DataAugmentationSingleObject(
        global_crops_scale=(0.25, 1.0),
        local_crops_scale=(0.05, 0.25),
        global_crops_number=2,
        local_crops_number=10,
        sample_strategy="random_area",
    )

    ds = ODISWTVeniceFrameDataset(
        data_root="/scratch/project_462000938/wt_venice",
        transform=aug,
        patch_size=16,
        num_object_per_view=1,
        pred_ratio=0.15,
        pred_ratio_var=0.05,
        pred_aspect_ratio=(0.3, 3.3),
        need_neighbor=True,
        static_crop=False,
        crop_size=300,
    )

    print(len(ds))
    sample = ds[0]
    print(
        "Returned tuple sizes:",
        len(sample[0]),
        len(sample[1]),
        len(sample[2]),
        len(sample[3]),
    )
