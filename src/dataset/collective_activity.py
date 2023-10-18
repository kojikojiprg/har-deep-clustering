import os
import sys
from glob import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.append("src")
from dataset.abstract_dataset import AbstractDataset


class CollectiveActivityDataset(AbstractDataset):
    def __init__(self, dataset_dir: str, seq_len: int, resize_ratio: float, stage: str):
        super().__init__(seq_len, resize_ratio)
        self.w = int(720 * resize_ratio)
        self.h = int(480 * resize_ratio)
        self._target_idxs = None
        self._clip_names = None

        self._create_dataset(dataset_dir, stage)

    @property
    def target_idxs(self):
        return self._target_idxs

    @property
    def clip_names(self):
        return self._clip_names

    def _create_dataset(self, dataset_dir, stage):
        clip_dirs = sorted(glob(os.path.join(dataset_dir, "*")))

        annotations, group_classes = self._load_annotations(clip_dirs)
        clip_names = self._split_train_test(group_classes, stage)

        # frame and flow
        frame_sizes = self._load_frames(dataset_dir, clip_names)
        self._load_opticalflows(dataset_dir, clip_names)
        self._transform_frame_flow()

        # bbox
        self._extract_bbox(annotations, clip_names, frame_sizes)
        self._calc_idx_ranges(annotations, clip_names)

    def _load_annotations(self, clip_dirs):
        annotations = {}
        group_classes = {c: [] for c in range(1, 7)}
        for clip_dir in clip_dirs:
            # load from txt file
            ann = np.loadtxt(os.path.join(clip_dir, "annotations.txt"), delimiter="\t")
            mask = ((ann[:, 0].astype(int) - 1) % 10 == 0) & (ann[:, 0].astype(int) > self._seq_len)
            ann = ann[mask]
            n_last_frame = ann[-1, 0]

            # get group class of each clip
            clip_name = clip_dir.split("/")[-1]
            unique, freq = np.unique(ann[:, 5], return_counts=True)
            mode = unique[np.argmax(freq)]

            annotations[clip_name] = {"annotation": ann, "n_last_frame": n_last_frame}
            group_classes[mode].append(clip_name)
        return annotations, group_classes

    def _split_train_test(self, group_classes, stage):
        stage_clip_names = []
        for clip_names in group_classes.values():
            test_length = len(clip_names) // 3
            if stage == "train":
                stage_clip_names += list(clip_names)[:-test_length]
            elif stage == "test":
                stage_clip_names += list(clip_names)[-test_length:]
            elif stage == "validation":
                stage_clip_names += list(clip_names)[-test_length:]
            else:
                raise KeyError

        if stage == "validation":
            stage_clip_names = stage_clip_names[:3]
        self._clip_names = stage_clip_names
        return stage_clip_names

    def _load_frames(self, dataset_dir, clip_names):
        raw_frame_sizes = {}
        for clip_name in tqdm(clip_names, ncols=100, desc="frame"):
            frames = []
            img_paths = sorted(glob(os.path.join(dataset_dir, clip_name, "*.jpg")))
            for i, img_path in enumerate(tqdm(img_paths, ncols=100, leave=False)):
                frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if i == 0:
                    raw_frame_sizes[clip_name] = frame.shape[1::-1]
                frame = cv2.resize(frame, (self.w, self.h))
                frames.append(frame)
            self._frames.append(frames)
            del frames
        return raw_frame_sizes

    def _load_opticalflows(self, dataset_dir, clip_names):
        for clip_name in tqdm(clip_names, ncols=100, desc="flow"):
            flows = np.load(os.path.join(dataset_dir, clip_name, "flow.npy"))
            flows_resized = []
            for flow in tqdm(flows, ncols=100, leave=False):
                flow = cv2.resize(flow, (self.w, self.h))
                flows_resized.append(flow)
            self._flows.append(flows_resized)
            del flows, flows_resized

    def _transform_frame_flow(self):
        for i in tqdm(range(len(self._frames)), ncols=100, desc="transform"):
            self._frames[i] = super().transform_imgs(self._frames[i])
            self._flows[i] = super().transform_imgs(self._flows[i])

    def _extract_bbox(self, annotations, clip_names, raw_frame_sizes):
        max_n_samples = 0

        for clip_name in clip_names:
            frame_size = raw_frame_sizes[clip_name]
            rx = self.w / frame_size[0]
            ry = self.h / frame_size[1]

            ann = annotations[clip_name]["annotation"]
            bboxs_clip = {}
            for n_frame in np.unique(ann[:, 0]):
                mask = np.where(ann[:, 0].astype(int) == n_frame)[0]
                b = ann[mask, 1:5]
                x1 = (b[:, 0] * rx).reshape(-1, 1).astype(int)
                y1 = (b[:, 1] * ry).reshape(-1, 1).astype(int)
                x2 = ((b[:, 0] + b[:, 2]) * rx).reshape(-1, 1).astype(int)
                y2 = ((b[:, 1] + b[:, 3]) * ry).reshape(-1, 1).astype(int)

                x1[x1 < 0] = 0
                y1[y1 < 0] = 0
                x2[self.w < x2] = self.w
                y2[self.h < y2] = self.h

                bboxs = np.concatenate([x1, y1, x2, y2], axis=1)
                target_idx = n_frame - 1
                bboxs_clip[target_idx] = bboxs

                if max_n_samples < len(bboxs):
                    max_n_samples = len(bboxs)

            self._bboxs.append(bboxs_clip)

        self._n_samples_batch = max_n_samples

    def _calc_idx_ranges(self, annotations, clip_names):
        target_idxs = []
        for clip_idx, clip_name in enumerate(clip_names):
            ann = annotations[clip_name]["annotation"]
            for frame_num in ann[:, 0]:
                target_idxs.append((clip_idx, frame_num - 1))

        self._target_idxs = np.array(target_idxs).astype(int)

    def __len__(self):
        return len(self._target_idxs)

    def __getitem__(self, idx):
        clip_idx, target_idx = self._target_idxs[idx]
        frames = self._frames[clip_idx][target_idx - self._seq_len: target_idx]
        frames = frames.transpose(1, 0)
        flows = self._flows[clip_idx][target_idx - self._seq_len : target_idx]
        flows = flows.transpose(1, 0)
        try:
            bboxs = self._bboxs[clip_idx][target_idx]
        except IndexError:
            print(clip_idx, target_idx, len(self._bboxs), len(self._bboxs[clip_idx]))
            raise IndexError
        # append dmy bboxs
        if len(bboxs) < self._n_samples_batch:
            diff_num = self._n_samples_batch - len(bboxs)
            dmy_bboxs = [np.full((4,), np.nan) for _ in range(diff_num)]
            bboxs = np.append(bboxs, dmy_bboxs, axis=0)
        bboxs = torch.Tensor(bboxs)

        return frames, flows, bboxs, idx
