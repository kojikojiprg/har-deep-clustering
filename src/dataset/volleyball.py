import os
import sys
from glob import glob
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.append("src")
from dataset.abstract_dataset import AbstractDataset

VALIDATION_SET = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
TEST_SET = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]
TRAIN_SET = [i for i in range(55) if i not in VALIDATION_SET and i not in TEST_SET]


class VolleyballDataset(AbstractDataset):
    def __init__(self, dataset_dir: str, cfg: SimpleNamespace, stage: str):
        super().__init__(cfg.seq_len, cfg.resize_ratio)
        self.w = int(cfg.img_size.w * cfg.resize_ratio)
        self.h = int(cfg.img_size.h * cfg.resize_ratio)
        self._create_dataset(dataset_dir, stage)

    def _create_dataset(self, dataset_dir, stage):
        if stage == "train":
            video_nums = TRAIN_SET
        elif stage == "validation":
            video_nums = VALIDATION_SET
        elif stage == "test":
            video_nums = TEST_SET
        else:
            raise KeyError
        video_dirs = sorted(glob(os.path.join(dataset_dir, "*/")))
        clip_dirs = []
        for video_dir in video_dirs:
            if int(os.path.basename(os.path.dirname(video_dir))) in video_nums:
                clip_dirs += sorted(glob(os.path.join(video_dir, "*/")))

        # bbox
        annotations = self._load_annotations(video_dirs, clip_dirs)

        # frame and flow
        frame_sizes = self._load_frames(clip_dirs, list(annotations.keys()))
        self._load_opticalflows(clip_dirs)
        self._transform_frame_flow()

        self._extract_bbox(annotations, frame_sizes)

    def _load_annotations(self, video_dirs, clip_dirs):
        ann_txts = {}
        for video_dir in video_dirs:
            # load from txt file
            ann_path = os.path.join(video_dir, "annotations.txt")
            with open(ann_path, "r") as f:
                txt = f.readlines()
                txt = [line.replace("\n", "").split(" ") for line in txt]
            video_num = int(os.path.basename(os.path.dirname(video_dir)))
            ann_txts[video_num] = txt

        annotations = {}
        for clip_dir in clip_dirs:
            video_num, frame_num = clip_dir.split("/")[-3:-1]
            txt = ann_txts[int(video_num)]
            line = list(filter(lambda x: x[0].replace(".jpg", "") == frame_num, txt))[0]
            clip_name = f"{video_num}_{frame_num}"
            annotations[clip_name] = line
        return annotations

    def _load_frames(self, clip_dirs, clip_names):
        raw_frame_sizes = {}
        for i, clip_dir in enumerate(tqdm(clip_dirs, ncols=100, desc="frame")):
            frames = []
            target_frame_num = int(clip_names[i].split("_")[1])
            start_frame_num = target_frame_num - self._seq_len + 1
            img_paths = sorted(glob(os.path.join(clip_dir, "*.jpg")))
            for img_path in img_paths:
                frame_num = int(os.path.basename(img_path).replace(".jpg", ""))
                if start_frame_num <= frame_num and frame_num <= target_frame_num:
                    frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if clip_names[i] not in raw_frame_sizes:
                        raw_frame_sizes[clip_names[i]] = frame.shape[1::-1]
                    frame = cv2.resize(frame, (self.w, self.h))
                    frames.append(frame)
            self._frames.append(frames)
            del frames
        return raw_frame_sizes

    def _load_opticalflows(self, clip_dirs):
        for clip_dir in tqdm(clip_dirs, ncols=100, desc="flow"):
            target_frame_num = 21  # target frame num is 21 in each video
            start_frame_num = target_frame_num - self._seq_len + 1
            flows = np.load(os.path.join(clip_dir, "flow.npy"))
            flows = flows[start_frame_num: target_frame_num + 1]
            flows_resized = []
            for flow in flows:
                flow = cv2.resize(flow, (self.w, self.h))
                flows_resized.append(flow)
            self._flows.append(flows_resized)
            del flows, flows_resized

    def _transform_frame_flow(self):
        for i in tqdm(range(len(self._frames)), ncols=100, desc="transform"):
            self._frames[i] = super().transform_imgs(self._frames[i])
            self._flows[i] = super().transform_imgs(self._flows[i])

    def _extract_bbox(self, annotations, raw_frame_sizes):
        max_n_samples = 0

        for clip_name, line in annotations.items():
            frame_size = raw_frame_sizes[clip_name]
            rx = self.w / frame_size[0]
            ry = self.h / frame_size[1]

            n_persons = int(len(line[1:]) // 5)
            bboxs_clip = []
            for i in range(2, len(line), 5):
                try:
                    x, y, w, h = list(map(int, line[i: i + 4]))
                except ValueError:
                    break  # this line has space in last
                b = np.array([x, y, x + w, y + h], dtype=np.float64)
                b = b * np.array([rx, ry, rx, ry])
                b = b.astype(int)
                b[b < 0] = 0  # check x1, y1
                b[2] = min(b[2], self.w)  # check x2
                b[3] = min(b[3], self.h)  # check y2
                bboxs_clip.append(b)

            if max_n_samples < n_persons:
                max_n_samples = n_persons

            self._bboxs.append(np.array(bboxs_clip))

        self._n_samples_batch = max_n_samples

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        frames = self._frames[idx].transpose(1, 0)
        flows = self._flows[idx].transpose(1, 0)
        bboxs = self._bboxs[idx]
        # append dmy bboxs
        if len(bboxs) < self._n_samples_batch:
            diff_num = self._n_samples_batch - len(bboxs)
            dmy_bboxs = [np.full((4,), np.nan) for _ in range(diff_num)]
            bboxs = np.append(bboxs, dmy_bboxs, axis=0)
        bboxs = torch.Tensor(bboxs)

        return frames, flows, bboxs, idx
