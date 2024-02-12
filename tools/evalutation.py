import argparse
import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

sys.path.append("src")
from utils import file_io, json_handler

label_name = {
    0: "surgeon",
    1: "anesthesist",
    2: "scrub nurse",
    3: "circulator",
    4: "other",
}


def parser():
    parser = argparse.ArgumentParser()

    # positional
    parser.add_argument("dataset_dir", type=str)
    parser.add_argument("stage", type=str, help="'train' or 'test'")
    parser.add_argument("version", type=int)

    # optional
    parser.add_argument("-dt", "--dataset_type", type=str, required=False, default=None)
    parser.add_argument(
        "-mc",
        "--model_config_dir",
        type=str,
        required=False,
        default="configs/",
    )

    args = parser.parse_args()

    return args


def main():
    args = parser()
    dataset_dir = args.dataset_dir
    stage = args.stage
    dataset_type = args.dataset_type
    version = args.version

    if dataset_type is None:
        if dataset_dir.endswith("/"):
            dataset_type = os.path.basename(os.path.dirname(dataset_dir))
        else:
            dataset_type = os.path.basename(dataset_dir)

    if dataset_type in ["collective", "volleyball"]:
        config_path = f"configs/model_config_{dataset_type}.yaml"
    else:
        config_path = f"configs/{dataset_type}/model_config-v{version}.yaml"
    config = file_io.get_config(config_path)

    json_save_dir = os.path.join("out", dataset_type, f"v{version}", "json")
    json_paths = glob(os.path.join(json_save_dir, f"{stage}*.json"))
    pred_labels = []
    for path in json_paths:
        pred_labels.append(json_handler.load(path))

    dataset_dir = os.path.join(dataset_dir, stage)
    json_paths = glob(
        os.path.join(dataset_dir, "**", "json", "annotation_clustering.json")
    )
    true_labels = []
    for path in json_paths:
        true_labels.append(json_handler.load(path))

    # linking cluster numbers to labels
    n_clusters = config.clustering.n_clusters
    sort_patterns = {}
    for clip_num, (pred_results, true_results) in enumerate(
        zip(pred_labels, true_labels)
    ):
        clip_num += 1

        sort_patterns[clip_num] = {}
        hist = {i: [] for i in range(n_clusters)}
        hist_plot = {i: [] for i in label_name.keys()}
        for pred_result in pred_results:
            frame_num = pred_result["frame"]
            sample_idx = pred_result["sample_idx"]

            try:
                true_result = [
                    result for result in true_results if result["frame"] == frame_num
                ][sample_idx]
            except IndexError:
                # check annotation
                print(clip_num, frame_num, sample_idx, pred_result)
                print(
                    [result for result in true_results if result["frame"] == frame_num]
                )
                raise IndexError

            hist[pred_result["label"]].append(true_result["label"])
            hist_plot[true_result["label"]].append(pred_result["label"])

        for true_label, pred_label_count in hist.items():
            sort_patterns[clip_num][true_label] = stats.mode(pred_label_count).mode

    # collect true labels and pred labels
    trues = []
    preds = []
    for clip_num, (pred_results, true_results) in enumerate(
        zip(pred_labels, true_labels)
    ):
        clip_num += 1

        sort_pattern = sort_patterns[clip_num]
        for pred_result in pred_results:
            frame_num = pred_result["frame"]
            sample_idx = pred_result["sample_idx"]

            true_result = [
                result for result in true_results if result["frame"] == frame_num
            ][sample_idx]
            trues.append(true_result["label"])
            preds.append(sort_pattern[pred_result["label"]])

    # calc accuracy
    acc = accuracy_score(trues, preds)
    eval_df = pd.DataFrame(
        classification_report(
            trues, preds, target_names=list(label_name.values()), output_dict=True
        )
    ).T.round(3)
    eval_df.to_csv(f"out/{dataset_type}/v{version}/eval_{stage}.csv")
    print(f"saved evaluation summary in 'out/{dataset_type}/v{version}/eval_{stage}.csv'.")

    # plot confusion matrix
    cm = confusion_matrix(trues, preds)
    cmd = ConfusionMatrixDisplay(cm, display_labels=list(label_name.values()))
    cmd.plot(cmap=plt.cm.Blues, xticks_rotation="vertical", colorbar=False)
    plt.title(f"accuracy = {np.round(acc, 3)}")
    plt.savefig(f"out/{dataset_type}/v{version}/cm_{stage}.png", bbox_inches="tight")
    plt.close()

    cm_det = np.round(cm / np.sum(cm, axis=1).reshape(-1, 1), decimals=2)
    cmd = ConfusionMatrixDisplay(cm_det)
    cmd.plot(cmap=plt.cm.Blues, colorbar=False)
    plt.savefig(
        f"out/{dataset_type}/v{version}/cm_det_{stage}.png", bbox_inches="tight"
    )
    plt.close()

    print(f"saved confusion matrix in 'out/{dataset_type}/v{version}/'.")
    print("accuracy", acc)


if __name__ == "__main__":
    main()
