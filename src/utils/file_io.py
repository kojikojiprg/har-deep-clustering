from types import SimpleNamespace

import yaml


def get_config(config_path: str) -> SimpleNamespace:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config = _get_config_reccursive(config)

    return config


def _get_config_reccursive(config: dict):
    new_config = SimpleNamespace(**config)
    for name, values in new_config.__dict__.items():
        if type(values) is dict:
            new_config.__setattr__(name, _get_config_reccursive(values))
        else:
            continue
    return new_config
