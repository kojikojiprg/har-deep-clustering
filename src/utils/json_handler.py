import json
import os
from json import JSONEncoder

import numpy as np


def load(json_path):
    data = {}
    with open(json_path, "r") as f:
        data = json.load(f)

    # convert frame number to int in group data
    if "group" in os.path.basename(json_path):
        data = {
            k: {int(frame_num): lst for frame_num, lst in v.items()}
            for k, v in data.items()
        }

    return data


def dump(data, json_path):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    with open(json_path, "w") as f:
        json.dump(data, f, cls=MyEncoder)


class MyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
