import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from lightning.pytorch import Trainer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.append("src")
from dataset import Datamodule
from model import select_deep_clustering_module
from utils import file_io, json_handler, video


def parser():
    parser = argparse.ArgumentParser()

    # positional
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("model_type", type=str, help="'frame_flow' or 'flow'")
    parser.add_argument("stage", type=str, help="'train' or 'test'")

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
    stage = args.stage
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
        dataset_dir, dataset_type, config, stage, augment_data=False
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

    checkpoint_dir = "models"
    checkpoint_path = os.path.join(
        checkpoint_dir,
        dataset_type,
        model_type,
        f"dcm_seq{config.seq_len}_last-v{version}.ckpt",
    )

    # predicting
    print("=> predicting")
    log_dir = os.path.join(log_dir, dataset_type)
    trainer = Trainer(
        logger=False,
        accelerator="gpu",
        devices=gpu_ids,
        strategy="ddp",
    )
    pred_results = trainer.predict(
        model, datamodule=datamodule, return_predictions=True, ckpt_path=checkpoint_path
    )

    print("=> plotting scatter")
    save_dir = os.path.join("out", dataset_type, model_type, f"v{version}")
    os.makedirs(save_dir, exist_ok=True)

    # collect z
    cluster_idxs = {c: [] for c in range(config.clustering.n_clusters)}
    z_lst = []
    i = 0
    for results in pred_results:
        for result in results:
            c = result["label"]
            z = result["z"]
            cluster_idxs[c].append(i)
            z_lst.append(z)
            i += 1

    # pca
    cmap = plt.get_cmap("tab10")
    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(z_lst)
    for c, idxs in cluster_idxs.items():
        if len(idxs) > 0:
            plt.scatter(
                emb_pca[idxs, 0],
                emb_pca[idxs, 1],
                s=3,
                label=c,
                alpha=0.7,
                color=cmap(c),
            )
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    pca_path = os.path.join(save_dir, f"{stage}-v{version}-pca.png")
    plt.savefig(pca_path, bbox_inches="tight")
    plt.close()

    # t-sne
    tsne = TSNE(n_components=2)
    emb_tsne = tsne.fit_transform(np.array(z_lst))
    for c, idxs in cluster_idxs.items():
        if len(idxs) > 0:
            plt.scatter(
                emb_tsne[idxs, 0],
                emb_tsne[idxs, 1],
                s=3,
                label=c,
                alpha=0.7,
                color=cmap(c),
            )
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0)
    tsne_path = os.path.join(save_dir, f"{stage}-v{version}-tsne.png")
    plt.savefig(tsne_path, bbox_inches="tight")
    plt.close()

    print("=> writing video")
    path = os.path.join(save_dir, f"{stage}-v{version}.mp4")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w = datamodule.dataset.w
    h = datamodule.dataset.h
    wrt = video.Writer(path, 30, (w, h))

    for frame, _, _, _, idx in iter(datamodule.dataset):
        frame = frame.cpu().numpy().transpose(1, 2, 3, 0)[-1]
        frame = ((frame + 1) / 2 * 255).astype(np.uint8).copy()
        results = pred_results[idx]
        for result in results:
            bbox = result["bbox"]
            c = result["label"]
            if np.any(np.isnan(bbox)):
                continue

            pt1 = np.array((bbox[0], bbox[1])).astype(int)
            pt2 = np.array((bbox[2], bbox[3])).astype(int)
            color = (np.array(cmap(c)) * 255)[:3].astype(np.uint8).tolist()
            color = color[::-1]  # rgb to bgr
            frame = cv2.rectangle(frame, pt1, pt2, color, 2)
            frame = cv2.putText(
                frame, f"{c}", pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_4
            )
        wrt.write(frame)
    del wrt

    print("=> saving pred results")
    path = os.path.join(save_dir, f"{stage}-v{version}.json")
    json_handler.dump(pred_results, path)

    pred_results_clips = {}
    for idx, (clip_idx, data_idx) in enumerate(datamodule.dataset.start_idxs):
        if clip_idx + 1 not in pred_results_clips:
            pred_results_clips[clip_idx + 1] = []
        results = pred_results[idx]
        frame_num = data_idx + config.seq_len
        for result in results:
            new_result = {
                "frame": frame_num,
                "sample_idx": result["sample_idx"],
                "label": result["label"],
            }
            pred_results_clips[clip_idx + 1].append(new_result)

    json_save_dir = os.path.join(save_dir, "json")
    os.makedirs(json_save_dir, exist_ok=True)
    for clip_num, results_clip in pred_results_clips.items():
        path = os.path.join(json_save_dir, f"{stage}-{clip_num:02d}-v{version}.json")
        json_handler.dump(results_clip, path)

    print("=> complete")


if __name__ == "__main__":
    main()
