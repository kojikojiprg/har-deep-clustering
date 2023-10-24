from types import SimpleNamespace

import yaml


def get_config(config_path: str) -> SimpleNamespace:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config = _get_config_reccursive(config)

    # resize img size
    config.img_size.w = int(config.img_size.w * config.resize_ratio)
    config.img_size.h = int(config.img_size.h * config.resize_ratio)

    return config


def _get_config_reccursive(config: dict):
    new_config = SimpleNamespace(**config)
    for name, values in new_config.__dict__.items():
        if type(values) is dict:
            new_config.__setattr__(name, _get_config_reccursive(values))
        else:
            continue
    return new_config
