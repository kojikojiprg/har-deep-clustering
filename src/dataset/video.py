import os
import sys
from glob import glob

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.append("src")
from dataset.abstract_dataset import AbstractDataset
from utils import json_handler, video


class VideoDataset(AbstractDataset):
    def __init__(self, dataset_dir: str, seq_len: int, resize_ratio: float, stage: str):
        super().__init__(seq_len, resize_ratio)
        self.w = int(1280 * resize_ratio)
        self.h = int(940 * resize_ratio)
        self._idx_ranges = []
        self._create_dataset(dataset_dir, stage)

    def _create_dataset(self, dataset_dir, stage):
        clip_dirs = sorted(glob(os.path.join(dataset_dir, "*/")))
        clip_paths = sorted(glob(os.path.join(dataset_dir, "*.mp4")))

        # frame and flow
        frame_size, frame_lengths = self._load_frames(clip_paths)
        self._load_opticalflows(clip_dirs)
        self._transform_frame_flow()

        # bbox
        self._load_bboxs(clip_dirs, frame_size)

        self._calc_idx_ranges(frame_lengths)

    def _load_frames(self, clip_paths):
        frame_lengths = []
        for clip_path in tqdm(clip_paths, ncols=100, desc="frame"):
            cap = video.Capture(clip_path)
            frames = []
            for _ in range(cap.frame_count):
                frame = cap.read()[1]
                frame = cv2.resize(frame, (self.w, self.h))
                frames.append(frame)
            frame_lengths.append(len(frames))
            self._frames.append(frames)
            del frames

        # TODO return all frame sizes
        return cap.size, frame_lengths

    def _load_opticalflows(self, clip_dirs):
        for clip_dir in tqdm(clip_dirs, ncols=100, desc="flow"):
            flows = np.load(os.path.join(clip_dir, "bin", "flow.npy"), mmap_mode="r")
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

    def _load_bboxs(self, clip_dirs, frame_size):
        rx = self.w / frame_size[0]
        ry = self.h / frame_size[1]
        max_bboxs_num = 0
        for clip_dir in clip_dirs:
            # load from txt file
            json_path = os.path.join(clip_dir, "json", "pose.json")
            json_data = json_handler.load(json_path)
            bboxs_clip = {}
            for data in json_data:
                frame_num = data["frame"]
                bbox = np.array(data["bbox"]) * np.array([rx, ry, rx, ry])
                if frame_num not in bboxs_clip:
                    bboxs_clip[frame_num] = []
                bboxs_clip[frame_num].append(bbox)

            for bboxs_frame in bboxs_clip.values():
                bboxs_num = len(bboxs_frame)
                if bboxs_num > max_bboxs_num:
                    max_bboxs_num = bboxs_num

            self._bboxs.append(bboxs_clip)

        self._n_samples_batch = max_bboxs_num

    def _calc_idx_ranges(self, frame_lengths):
        start_idx = 0
        for frame_length in frame_lengths:
            end_idx = start_idx + frame_length - self._seq_len
            self._idx_ranges.append([start_idx, end_idx])
            start_idx = end_idx + 1
        self._idx_ranges = np.array(self._idx_ranges)

    def __len__(self):
        return self._idx_ranges[-1, 1] + 1

    def __getitem__(self, idx):
        idx_ranges = self._idx_ranges
        clip_idx = np.where((idx_ranges[:, 0] <= idx) & (idx <= idx_ranges[:, 1]))
        clip_idx = clip_idx[0].item()
        data_idx = idx - idx_ranges[clip_idx, 0]

        frames = self._frames[clip_idx][data_idx : data_idx + self._seq_len]
        frames = frames.transpose(1, 0)
        flows = self._flows[clip_idx][data_idx : data_idx + self._seq_len]
        flows = flows.transpose(1, 0)

        frame_num = data_idx + self._seq_len
        if frame_num in self._bboxs[clip_idx]:
            bboxs = self._bboxs[clip_idx][frame_num]
            bboxs = np.array(bboxs)
            # append dmy bboxs
            if len(bboxs) < self._n_samples_batch:
                diff_num = self._n_samples_batch - len(bboxs)
                dmy_bboxs = [np.full((4,), np.nan) for _ in range(diff_num)]
                bboxs = np.append(bboxs, dmy_bboxs, axis=0)
        else:
            bboxs = np.full((self._n_samples_batch, 4), np.nan)
        bboxs = torch.Tensor(bboxs)

        return frames, flows, bboxs, idx
