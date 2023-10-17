import argparse
import os
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm


def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset_dir",
        type=str,
        help="path of input dataset directory",
    )
    # optional
    parser.add_argument(
        "--dataset_type",
        type=str,
        required=False,
        default=None,
        help="select from 'collective' or 'volleyball or 'video', default by None.",
    )

    return parser.parse_args()


def main():
    args = parser()
    dataset_dir = args.dataset_dir
    dataset_type = args.dataset_type.lower()

    if dataset_type is None:
        if dataset_dir.endswith("/"):
            dataset_type = os.path.basename(os.path.dirname(dataset_dir))
        else:
            dataset_type = os.path.basename(dataset_dir)

    if dataset_type == "collective":
        clip_dirs = sorted(glob(os.path.join(dataset_dir, "*")))
    elif dataset_type == "volleyball":
        video_dirs = sorted(glob(os.path.join(dataset_dir, "*/")))
        clip_dirs = []
        for clip_dir in video_dirs:
            clip_dirs += sorted(glob(os.path.join(clip_dir, "*/")))
    elif dataset_type == "video":
        raise NotImplementedError
    else:
        raise TypeError

    for clip_dir in tqdm(clip_dirs, ncols=100):
        output_path = os.path.join(clip_dir, "flow.npy")
        if os.path.exists(output_path):
            continue
        frame_paths = sorted(glob(os.path.join(clip_dir, "*.jpg")))

        frame = cv2.imread(frame_paths[0])
        pre_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flows = []
        for frame_path in tqdm(frame_paths[1:], ncols=100, leave=False):
            frame = cv2.imread(frame_path)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                pre_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            flows.append(flow)
            pre_gray = gray
        flows = [np.zeros_like(flow)] + flows

        output_path = os.path.join(clip_dir, "flow.npy")
        np.save(output_path, np.array(flows))
        tqdm.write(f"saved {output_path}")

        del flows


if __name__ == "__main__":
    main()
