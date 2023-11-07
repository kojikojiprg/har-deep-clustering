import gc
import os
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from numpy.typing import NDArray


class Capture:
    def __init__(self, in_path: str):
        if not os.path.isfile(in_path):
            raise ValueError(f"not exist file {in_path}")

        self._cap = cv2.VideoCapture(in_path)

        self.fps = int(self._cap.get(cv2.CAP_PROP_FPS))
        self.size = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def __del__(self):
        self._cap.release()
        gc.collect()

    @property
    def frame_count(self) -> int:
        # cv2.CAP_PROP_FRAME_COUNT is not correct.
        self.set_pos_frame_count(int(1e10))  # set large value
        count = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.set_pos_frame_count(0)  # initialize
        return count

    @property
    def is_opened(self) -> bool:
        return self._cap.isOpened()

    def set_pos_frame_count(self, idx: int):
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

    def set_pos_frame_time(self, begin_sec: int):
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, begin_sec * self.fps)

    def read(self, idx: Optional[int] = None) -> Tuple[bool, Union[NDArray, None]]:
        if idx is not None:
            self.set_pos_frame_count(idx)

        ret, frame = self._cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR to RGB
            return True, frame
        else:
            return False, None


class Writer:
    def __init__(self, out_path, fps, size, fmt="mp4v"):
        out_dir = os.path.dirname(out_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        # writer object
        fmt = cv2.VideoWriter_fourcc(fmt[0], fmt[1], fmt[2], fmt[3])
        self._writer = cv2.VideoWriter(out_path, fmt, fps, size)

    def __del__(self):
        self._writer.release()
        gc.collect()

    def write(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # RGB to BGR
        self._writer.write(frame)

    def write_each(self, frames):
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # RGB to BGR
            self._writer.write(frame)


def concat_frames(frame1: NDArray, frame2: NDArray) -> NDArray:
    # change frame2 height and merge to frame1
    ratio = frame1.shape[0] / frame2.shape[0]
    size = (round(frame2.shape[1] * ratio), frame1.shape[0])
    frame2 = cv2.resize(frame2, size)
    frame1 = np.concatenate([frame1, frame2], axis=1)

    return frame1


def get_concat_frame_size(frame: NDArray, field: NDArray) -> Tuple[int, ...]:
    cmb_img = concat_frames(frame, field)
    return cmb_img.shape[1::-1]


def _adjust_ang(ang_min, ang_max):
    unique_ang_min = ang_min
    unique_ang_max = ang_max
    unique_ang_min %= 360
    unique_ang_max %= 360
    if unique_ang_min >= unique_ang_max:
        unique_ang_max += 360
    return unique_ang_min, unique_ang_max


def _any_angle_only(mag, ang, ang_min, ang_max):
    any_mag = np.copy(mag)
    any_ang = np.copy(ang)
    ang_min %= 360
    ang_max %= 360
    if ang_min < ang_max:
        any_mag[(ang < ang_min) | (ang_max < ang)] = np.nan
        any_ang[(ang < ang_min) | (ang_max < ang)] = np.nan
    else:
        any_mag[(ang_max < ang) & (ang < ang_min)] = np.nan
        any_ang[(ang_max < ang) & (ang < ang_min)] = np.nan
        any_ang[ang <= ang_max] += 360
    return any_mag, any_ang


def flow_to_rgb(flow):
    # 角度範囲のパラメータ
    ang_min = 0
    ang_max = 360
    _ang_min, _ang_max = _adjust_ang(ang_min, ang_max)  # 角度の表現を統一する

    # HSV色空間の配列に入れる
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    any_mag, any_ang = _any_angle_only(mag, ang, ang_min, ang_max)
    hsv[..., 0] = 180 * (any_ang - _ang_min) / (_ang_max - _ang_min)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(any_mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return flow_rgb
