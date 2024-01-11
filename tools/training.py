import argparse
import os
import sys

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

# from lightning.pytorch.strategies.fsdp import FSDPStrategy

sys.path.append("src")
from dataset import Datamodule
from model import select_deep_clustering_module
from utils import file_io


def parser():
    parser = argparse.ArgumentParser()

    # positional
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("model_type", type=str, help="'frame_flow' or 'flow'")

    # optional
    parser.add_argument("-dt", "--dataset_type", type=str, required=False, default=None)
    parser.add_argument("-v", "--version", type=int, required=False, default=None)
    parser.add_argument(
        "-mc",
        "--model_config_dir",
        type=str,
        required=False,
        default="configs/",
    )
    parser.add_argument("--checkpoint_dir", type=str, required=False, default="models/")
    parser.add_argument("--log_dir", type=str, required=False, default="logs/")
    parser.add_argument(
        "-g", "--gpus", type=int, nargs="*", default=None, help="gpu ids"
    )

    args = parser.parse_args()

    return args


def main():
    # get args
    args = parser()
    dataset_dir = args.dataset_dir
    model_type = args.model_type
    dataset_type = args.dataset_type
    version = args.version
    model_config_dir = args.model_config_dir
    checkpoint_dir = args.checkpoint_dir
    log_dir = args.log_dir
    gpu_ids = args.gpus

    if dataset_type is None:
        if dataset_dir.endswith("/"):
            dataset_type = os.path.basename(os.path.dirname(dataset_dir))
        else:
            dataset_type = os.path.basename(dataset_dir)

    # get config
    if version is None:
        model_config_path = os.path.join(
            model_config_dir, dataset_type, "model_config.yaml"
        )
    else:
        model_config_path = os.path.join(
            model_config_dir, dataset_type, f"model_config-v{version}.yaml"
        )
    config = file_io.get_config(model_config_path)

    # create dataset
    print(f"=> creating dataset from {dataset_dir}")
    datamodule = Datamodule(
        dataset_dir, dataset_type, config, "train", augment_data=True
    )

    # create model
    print("=> create model")
    n_samples = datamodule.n_samples
    n_samples_batch = datamodule.n_samples_batch
    checkpoint_dir = os.path.join(checkpoint_dir, dataset_type)
    model = select_deep_clustering_module(
        model_type,
        config,
        n_samples,
        n_samples_batch,
        checkpoint_dir,
        version,
        load_autoencoder_checkpoint=True,
    )

    # training
    # fsdp = FSDPStrategy(cpu_offload=True)
    log_dir = os.path.join(log_dir, dataset_type)
    trainer = Trainer(
        logger=TensorBoardLogger(log_dir, name=model_type),
        callbacks=model.callbacks,
        max_epochs=config.epochs,
        accumulate_grad_batches=config.accumulate_grad_batches,
        accelerator="gpu",
        devices=gpu_ids,
        strategy="ddp",
        # strategy=fsdp,
    )
    print("=> training")
    trainer.fit(model, datamodule=datamodule)

    print("=> complete")


if __name__ == "__main__":
    main()
