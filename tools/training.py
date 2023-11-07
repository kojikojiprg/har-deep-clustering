import argparse
import os
import sys
from datetime import timedelta

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies.ddp import DDPStrategy
# from lightning.pytorch.strategies.fsdp import FSDPStrategy

sys.path.append("src")
from dataset import Datamodule
from model import DeepClusteringModel
from utils import file_io

# torch._dynamo.config.suppress_errors = True


def parser():
    parser = argparse.ArgumentParser()

    # positional
    parser.add_argument("dataset_dir", type=str)

    # optional
    parser.add_argument("-dt", "--dataset_type", type=str, required=False, default=None)
    parser.add_argument(
        "-mc",
        "--model_config_path",
        type=str,
        required=False,
        default="configs/model_config.yaml",
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
    dataset_type = args.dataset_type
    model_config_path = args.model_config_path
    checkpoint_dir = args.checkpoint_dir
    log_dir = args.log_dir
    gpu_ids = args.gpus

    # get config
    config = file_io.get_config(model_config_path)

    # create dataset
    print(f"=> creating dataset from {dataset_dir}")
    if dataset_type is None:
        if dataset_dir.endswith("/"):
            dataset_type = os.path.basename(os.path.dirname(dataset_dir))
        else:
            dataset_type = os.path.basename(dataset_dir)
    datamodule = Datamodule(
        dataset_dir, dataset_type, config, "train"
    )

    # create model
    print("=> create model")
    n_samples = datamodule.n_samples
    n_samples_batch = datamodule.n_samples_batch
    checkpoint_dir = os.path.join(checkpoint_dir, dataset_type)
    model = DeepClusteringModel(config, n_samples, n_samples_batch, checkpoint_dir)

    # TODO
    # model = torch.compile(model, dynamic=True)

    # training
    # ddp = DDPStrategy(find_unused_parameters=True, timeout=timedelta(seconds=1800))
    # fsdp = FSDPStrategy(cpu_offload=True)
    trainer = Trainer(
        logger=TensorBoardLogger(log_dir, name=dataset_type),
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
