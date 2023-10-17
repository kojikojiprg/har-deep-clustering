import os
import sys
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append("src")
from dataset.abstract_dataset import AbstractDataset

VALIDATION_SET = [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51]
TEST_SET = [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47]
TRAIN_SET = [i for i in range(55) if i not in VALIDATION_SET and i not in TEST_SET]


class VolleyballDataset(AbstractDataset):
    def __init__(self, dataset_dir: str, seq_len: int, resize_ratio: float, stage: str):
        super().__init__(seq_len, resize_ratio)
        self.w = int(1280 * resize_ratio)
        self.h = int(720 * resize_ratio)
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
        video_dirs = sorted(glob(os.path.join(dataset_dir, "videos", "*/")))
        clip_dirs = []
        for video_dir in video_dirs:
            if int(os.path.basename(os.path.dirname(video_dir))) in video_nums:
                clip_dirs += sorted(glob(os.path.join(video_dir, "*/")))

        # bbox
        annotations = self._load_annotations(clip_dirs)

        # frame and flow
        frame_sizes = self._load_frames(clip_dirs, list(annotations.keys()))
        self._load_opticalflows(clip_dirs)
        self._transform_frame_flow()

        self._extract_bbox(annotations, frame_sizes)
        self._calc_idx_ranges(annotations)

    def _load_annotations(self, clip_dirs):
        annotations = {}
        for clip_dir in clip_dirs:
            # load from txt file
            ann_dir = clip_dir.replace("videos/", "volleyball-detections/")
            ann_path = os.path.join(ann_dir, "person_detections.txt")
            with open(ann_path, "r") as f:
                ann = f.readlines()
                ann = [line.split("\t") for line in ann]

            video_num, frame_num = clip_dir.split("/")[-3:-1]
            clip_name = f"{video_num}_{frame_num}"
            annotations[clip_name] = ann
        return annotations

    def _load_frames(self, clip_dirs, clip_names):
        raw_frame_sizes = {}
        for i, clip_dir in enumerate(tqdm(clip_dirs, ncols=100, desc="load frame")):
            frames = []
            img_paths = sorted(glob(os.path.join(clip_dir, "*.jpg")))
            for j, img_path in enumerate(tqdm(img_paths, ncols=100, leave=False)):
                frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if j == 0:
                    raw_frame_sizes[clip_names[i]] = frame.shape[1::-1]
                frame = cv2.resize(frame, (self.w, self.h))
                frames.append(frame)
            self._frames.append(frames)
            del frames
        return raw_frame_sizes

    def _load_opticalflows(self, clip_dirs):
        for clip_dir in tqdm(clip_dirs, ncols=100, desc="load opticalflow"):
            flows = np.load(os.path.join(clip_dir, "flow.npy"))
            flows_resized = []
            for flow in tqdm(flows, ncols=100, leave=False):
                flow = cv2.resize(flow, (self.w, self.h))
                flows_resized.append(flow)
            self._flows.append(flows_resized)
            del flows, flows_resized

    def _transform_frame_flow(self):
        for i in range(len(self._frames)):
            self._frames[i] = super().transform_imgs(self._frames[i])
            self._flows[i] = super().transform_imgs(self._flows[i])

    def _extract_bbox(self, annotations, raw_frame_sizes):
        max_n_samples = 0

        for clip_name, ann in annotations.items():
            frame_size = raw_frame_sizes[clip_name]
            rx = self.w / frame_size[0]
            ry = self.h / frame_size[1]

            bboxs_clip = []
            for line in ann:
                n_persons = int(line[1])
                bboxs = []
                for i in range(2, len(line), 6):
                    x, y, w, h = list(map(int, line[i : i + 4]))
                    b = np.array([x, y, x + w, y + h], dtype=np.float64)
                    b = b * np.array([rx, ry, rx, ry])
                    b = b.astype(int)
                    b[b < 0] = 0  # check x1, y1
                    b[2] = min(b[2], self.w)  # check x2
                    b[3] = min(b[3], self.h)  # check y2
                    bboxs.append(b)

                bboxs_clip.append(np.array(bboxs))

                if max_n_samples < n_persons:
                    max_n_samples = n_persons

            self._bboxs.append(bboxs_clip)

        self._n_samples_batch = max_n_samples

    def _calc_idx_ranges(self, clip_dirs):
        idx_ranges = []
        n_start_idx = 0
        for _ in range(len(clip_dirs)):
            n_last_idx = 41 - self._seq_len + n_start_idx  # all of clips have 41 frames
            idx_ranges.append((n_start_idx, n_last_idx))
            n_start_idx = n_last_idx

        self._idx_ranges = np.array(idx_ranges).astype(int)
