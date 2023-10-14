import os
import sys
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append("src")
from dataset.abstract_dataset import AbstractDataset


class CollectiveActivityDataset(AbstractDataset):
    ANNOTATION_COLUMNS = ["n_frame", "x", "y", "w", "h", "class_id", "pose_id"]

    def __init__(self, dataset_dir: str, seq_len: int, resize_ratio: float, stage: str):
        super().__init__(seq_len, resize_ratio)
        self.w = int(720 * resize_ratio)
        self.h = int(480 * resize_ratio)
        self._create_dataset(dataset_dir, stage)

    def _create_dataset(self, dataset_dir, stage):
        video_dirs = sorted(glob(os.path.join(dataset_dir, "*")))

        # bbox
        annotations, group_classes = self._load_annotations(video_dirs)
        video_names = self._split_train_test(group_classes, stage)

        # frame and flow
        frame_sizes = self._load_frames(dataset_dir, video_names)
        self._load_opticalflows(dataset_dir, video_names)
        self._transform_frame_flow()

        self._extract_bbox(annotations, video_names, frame_sizes)
        self._calc_idx_ranges(annotations, video_names)

    def _load_annotations(self, video_dirs):
        annotations = {}
        group_classes = {c: [] for c in range(2, 7)}
        for video_dir in video_dirs:
            # load from txt file
            ann = np.loadtxt(os.path.join(video_dir, "annotations.txt"), delimiter="\t")
            n_last_frame = ann[-1, 0]

            # get group class of each clip
            video_name = video_dir.split("/")[-1]
            unique, freq = np.unique(ann[:, 5], return_counts=True)
            mode = unique[np.argmax(freq)]

            annotations[video_name] = {"annotation": ann, "n_last_frame": n_last_frame}
            group_classes[mode].append(video_name)
        return annotations, group_classes

    def _split_train_test(self, group_classes, stage):
        stage_video_names = []
        for video_names in group_classes.values():
            test_length = len(video_names) // 3
            if stage == "train":
                stage_video_names += list(video_names)[:-test_length]
            else:
                stage_video_names += list(video_names)[-test_length:]
        return stage_video_names

    def _load_frames(self, dataset_dir, video_names):
        raw_frame_sizes = {}
        for video_name in tqdm(video_names, ncols=100, desc="load frame"):
            frames = []
            img_paths = sorted(glob(os.path.join(dataset_dir, video_name, "*.jpg")))
            for i, img_path in enumerate(tqdm(img_paths, ncols=100, leave=False)):
                frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if i == 0:
                    raw_frame_sizes[video_name] = frame.shape[1::-1]
                frame = cv2.resize(frame, (self.w, self.h))
                frames.append(frame)
            self._frames.append(frames)
            del frames
        return raw_frame_sizes

    def _load_opticalflows(self, dataset_dir, video_names):
        for video_name in tqdm(video_names, ncols=100, desc="load opticalflow"):
            flows = np.load(os.path.join(dataset_dir, video_name, "flow.npy"))
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

    def _extract_bbox(self, annotations, video_names, raw_frame_sizes):
        max_n_samples = 0
        n_max_frames = {}

        for video_name in video_names:
            frame_size = raw_frame_sizes[video_name]
            rx = self.w / frame_size[0]
            ry = self.h / frame_size[1]

            ann = annotations[video_name]["annotation"]
            bboxs_video = []
            n_max_frame = int(np.max(ann[:, 0]))
            n_max_frames[video_name] = n_max_frame
            for n_frame in range(1, n_max_frame + 1):
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
                bboxs_video.append(bboxs)

                if max_n_samples < len(bboxs):
                    max_n_samples = len(bboxs)

            self._bboxs.append(bboxs_video)

        self._n_samples_batch = max_n_samples

        return n_max_frames

    def _calc_idx_ranges(self, annotations, video_names):
        idx_ranges = []
        n_start_idx = 0
        for video_name in video_names:
            data = annotations[video_name]
            n_last_idx = data["n_last_frame"] - self._seq_len + n_start_idx + 1
            idx_ranges.append((n_start_idx, n_last_idx))
            n_start_idx = n_last_idx

        self._idx_ranges = np.array(idx_ranges).astype(int)
