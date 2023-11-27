import argparse
import os
import sys
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

sys.path.append("src")
from utils import video


def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "dataset_dir",
        type=str,
        help="path of input dataset directory",
    )
    # optional
    parser.add_argument(
        "-dt",
        "--dataset_type",
        type=str,
        required=False,
        default=None,
        help="select from 'collective' or 'volleyball or 'video', default by None.",
    )
    parser.add_argument(
        "--comp",
        required=False,
        default=False,
        action="store_true",
        help="compress output from float32 to float16.",
    )

    return parser.parse_args()


def main():
    args = parser()
    dataset_dir = args.dataset_dir
    dataset_type = args.dataset_type.lower()
    comp = args.comp

    if dataset_type is None:
        if dataset_dir.endswith("/"):
            dataset_type = os.path.basename(os.path.dirname(dataset_dir))
        else:
            dataset_type = os.path.basename(dataset_dir)

    if dataset_type == "collective":
        clip_dirs = sorted(glob(os.path.join(dataset_dir, "*/")))
    elif dataset_type == "volleyball":
        video_dirs = sorted(glob(os.path.join(dataset_dir, "*/")))
        clip_dirs = []
        for clip_dir in video_dirs:
            clip_dirs += sorted(glob(os.path.join(clip_dir, "*/")))
    elif dataset_type == "video":
        clip_dirs = sorted(glob(os.path.join(dataset_dir, "*/")))
    else:
        raise TypeError

    for clip_dir in tqdm(clip_dirs, ncols=100):
        # output_path = os.path.join(clip_dir, "flow.npy")
        # if os.path.exists(output_path):
        #     continue

        if dataset_type != "video":
            frame_paths = sorted(glob(os.path.join(clip_dir, "*.jpg")))
            frames = [cv2.imread(frame_path) for frame_path in tqdm(frame_paths, leave=False)]
        else:
            clip_path = os.path.dirname(clip_dir) + ".mp4"
            cap = video.Capture(clip_path)
            # frames = [cap.read()[1] for _ in tqdm(range(cap.frame_count), leave=False)]
            frames = [cap.read()[1] for _ in tqdm(range(20), leave=False)]

        pre_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        flows = []
        for frame in tqdm(frames[1:], ncols=100, leave=False):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                pre_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )

            # flow = video.flow_to_rgb(flow)  # for video output

            flows.append(flow)
            pre_gray = gray
        flows = [np.zeros_like(flow)] + flows

        output_path = os.path.join(clip_dir, "flow.npy")
        if comp:
            flows = np.array(flows, dtype=np.float16)
        else:
            flows = np.array(flows)
        np.save(output_path, flows)

        # output_path = os.path.join(clip_dir, "flow.mp4")
        # wrt = video.Writer(output_path, cap.fps, cap.size)
        # wrt.write_each(flows)
        # del wrt
        tqdm.write(f"saved {output_path}")

        del frames, flows
        if dataset_type == "video":
            del cap


if __name__ == "__main__":
    main()
