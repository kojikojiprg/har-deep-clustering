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
from utils import json_handler, video


class SurgeryDataset(AbstractDataset):
    def __init__(self, dataset_dir: str, cfg: SimpleNamespace, stage: str):
        super().__init__(cfg.seq_len)
        self.w = cfg.img_size.w
        self.h = cfg.img_size.h
        self._norms = []
        self._start_idxs = []
        self._frame_num_period = 1

        self._create_dataset(dataset_dir, stage)

    def _create_dataset(self, dataset_dir, stage):
        clip_dirs = sorted(glob(os.path.join(dataset_dir, "*/")))
        clip_paths = sorted(glob(os.path.join(dataset_dir, "*.mp4")))
        # clip_dirs = clip_dirs[:1]
        # clip_paths = clip_paths[:1]

        # frame and flow
        frame_size, frame_lengths = self._load_frames(clip_paths)
        self._load_opticalflows(clip_dirs)
        self._transform_frame_flow()

        # bbox
        self._load_bboxs(clip_dirs, frame_size)
        self._calc_norm_from_ot(clip_dirs, frame_size)

        self._calc_start_idxs(frame_lengths)

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
            flows = np.load(os.path.join(clip_dir, "flow.npy"), mmap_mode="r")
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
                if (
                    frame_num % self._frame_num_period != 0
                    and frame_num < self._seq_len
                ):
                    continue

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

    def _calc_norm_from_ot(self, clip_dirs, frame_size):
        rx = self.w / frame_size[0]
        ry = self.h / frame_size[1]
        for i, clip_dir in enumerate(clip_dirs):
            ot_coor = np.load(os.path.join(clip_dir, "coor.npy")).astype(np.float32)
            ot_coor *= np.array((rx, ry))
            bboxs_clip = self._bboxs[i]
            norms_clip = {}
            for frame_num, bboxs in bboxs_clip.items():
                bboxs = np.array(bboxs).reshape(-1, 2, 2)
                cbbox = bboxs[:, 0, :] + (bboxs[:, 1, :] - bboxs[:, 0, :]) / 2
                norm = np.linalg.norm(cbbox - ot_coor, axis=1)
                norm /= np.linalg.norm((self.w, self.h))
                norms_clip[frame_num] = norm

            self._norms.append(norms_clip)

    def _calc_start_idxs(self, frame_lengths):
        for clip_idx, frame_length in enumerate(frame_lengths):
            for data_idx in range(
                0, frame_length - self._seq_len + 1, self._frame_num_period
            ):
                frame_num = data_idx + self._seq_len
                if frame_num not in self._bboxs[clip_idx]:
                    continue
                self._start_idxs.append((clip_idx, data_idx))

    def __len__(self):
        return len(self._start_idxs)

    def __getitem__(self, idx):
        clip_idx, data_idx = self._start_idxs[idx]

        frames = self._frames[clip_idx][data_idx : data_idx + self._seq_len]
        frames = frames.transpose(1, 0)
        flows = self._flows[clip_idx][data_idx : data_idx + self._seq_len]
        flows = flows.transpose(1, 0)

        frame_num = data_idx + self._seq_len
        bboxs = np.array(self._bboxs[clip_idx][frame_num])
        norms = np.array(self._norms[clip_idx][frame_num])
        # append dmy bboxs
        if len(bboxs) < self._n_samples_batch:
            diff_num = self._n_samples_batch - len(bboxs)
            dmy_bboxs = [np.full((4,), np.nan) for _ in range(diff_num)]
            bboxs = np.append(bboxs, dmy_bboxs, axis=0)
            norms = np.append(norms, [np.nan for _ in range(diff_num)])
        bboxs = torch.Tensor(bboxs)
        norms = torch.Tensor(norms)

        return frames, flows, bboxs, norms, idx
