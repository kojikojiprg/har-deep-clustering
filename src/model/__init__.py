from types import SimpleNamespace
from typing import Optional

from .autoencoder import AutoencoderModule
from .flow import DeepClusteringModel as FlowModel
from .frame_flow import DeepClusteringModel as FrameFlowModel


def select_deep_clustering_module(
    model_type: str,
    cfg: SimpleNamespace,
    n_samples: int,
    n_samples_batch: int,
    checkpoint_dir: Optional[str] = None,
    version: Optional[int] = None,
    load_autoencoder_checkpoint: bool = True,
):
    if model_type == "frame_flow":
        return FrameFlowModel(
            model_type,
            cfg,
            n_samples,
            n_samples_batch,
            checkpoint_dir,
            version,
            load_autoencoder_checkpoint,
        )
    elif model_type == "flow":
        return FlowModel(
            model_type,
            cfg,
            n_samples,
            n_samples_batch,
            checkpoint_dir,
            version,
            load_autoencoder_checkpoint,
        )
    else:
        raise ValueError
