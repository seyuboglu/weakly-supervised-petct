"""
Utility functions for visualizing data.
"""
import os
from datetime import datetime
import json
from collections import Counter

import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
import plotly.offline as ply
import plotly.graph_objs as go
import plotly.express as px
import seaborn as sns
import colorlover as cl
import pandas as pd
import cv2
import torch
from torch.nn import ReLU
from sklearn.metrics import roc_curve
import matplotlib.animation as animation
from ipywidgets import FloatSlider, FloatLogSlider, Button, VBox, HBox, jslink
import ipyvolume as ipv

from pet_ct.util.util import place_on_gpu, place_on_cpu, process
from pet_ct.learn.history import TrainHistory
from pet_ct.data.h5_dataset import H5Dataset

matplotlib.rcParams["animation.embed_limit"] = 2 ** 128


def get_frontal_plane(imgs1, imgs2=None, resolution=None):

    if resolution:
        L, H, W = resolution
        imgs1 = np.array([cv2.resize(img, (W, L)) for img in imgs1])
        print(imgs1.shape)
    imgs1 = np.transpose(imgs1, [1, 0, 2])
    imgs1 = np.flip(imgs1, axis=1)
    if resolution:
        imgs1 = np.array([cv2.resize(img, (W, H)) for img in imgs1])

    if imgs2 is None:
        return imgs1

    if resolution:
        imgs2 = np.array([cv2.resize(img, (W, L)) for img in imgs2])
    imgs2 = np.transpose(imgs2, [1, 0, 2])
    imgs2 = np.flip(imgs2, axis=1)
    if resolution:
        imgs2 = np.array([cv2.resize(img, (W, H)) for img in imgs2])

    return imgs1, imgs2


def show_multiple_videos(imgs_list):
    full = np.concatenate(imgs_list, axis=2)

    ani = show_simple_video(full, num_exams=len(imgs_list))
    return ani


def show_simple_video(imgs, resolution=None, num_exams=1):
    fig, ax = plt.subplots()
    ax.grid(True)

    L, H, W, C = imgs.shape

    # xticks = [(i / (W / num_exams)) for i in range(W)]
    # plt.xticks(xticks)

    # yticks = [(i / H) for i in range(H)]
    # plt.yticks(yticks)

    def x_ticks_fn(value, tick_number):
        tick = value / (W / num_exams) % 1
        return tick

    def y_ticks_fn(value, tick_number):
        tick = value / H
        return tick

    ax.xaxis.set_major_formatter(plt.FuncFormatter(x_ticks_fn))
    ax.xaxis.set_major_locator(plt.MultipleLocator(W / num_exams / 4))

    ax.yaxis.set_major_formatter(plt.FuncFormatter(y_ticks_fn))
    ax.yaxis.set_major_locator(plt.MaxNLocator(10))

    frames = []
    num_imgs = len(imgs)
    for i, img in tqdm(enumerate(imgs)):
        text = ax.text(0, -2, f"{round(100 * i / num_imgs, 2)}%")
        if resolution:
            img = cv2.resize(img, resolution)
        frame = ax.imshow(img)
        frames.append([frame, text])

    ani = animation.ArtistAnimation(
        fig, frames, interval=100, blit=True, repeat_delay=1000
    )
    fig.tight_layout()
    plt.show()

    return ani


def show_pet_ct_video(imgs_ct, imgs_pet, resolution=None):
    assert len(imgs_ct) == len(imgs_pet), "image types are not of equal length."
    imgs = []
    for i in tqdm(range(len(imgs_ct))):
        if resolution:
            img_ct = cv2.resize(imgs_ct[i], resolution)
            img_pet = cv2.resize(imgs_pet[i], resolution)
        else:
            img_ct = imgs_ct[i]
            img_pet = imgs_pet[i]
        img = overlay_imgs(img_ct, img_pet)
        imgs.append(img)

    ani = show_simple_video(imgs, resolution=None)
    return ani


def normalize_img(img):
    """
    """
    img = img.astype(np.float32)
    img -= img.min()  # ensure the minimal value is 0.0
    img /= img.max()  # maximum value in image is now 1.0
    img = np.uint8(img * 255)
    return img


def overlay_imgs(
    ct, pet, resolution=None, saliency=None, alpha=0.5, alpha_saliency=0.5
):
    """
    """
    orig = normalize_img(ct.copy())
    orig = plt.cm.bone(orig)

    over = normalize_img(pet.copy())
    over = plt.cm.hot(over)

    cv2.addWeighted(over, alpha, orig, 1 - alpha, 0, orig)

    if saliency is not None:
        saliency = plt.cm.jet(saliency)
        cv2.addWeighted(saliency, alpha_saliency, orig, 1 - alpha_saliency, 0, orig)
    return orig


def show_attention(attention_probs, token_idx=None, threshold=0.0007):
    """
    Plots

    @attention_probs    (torch.Tensor) shape (batch_size, num_heads, length, width, height) or
                        shape (batch_size, length, width, height)
    """
    # add heads dimension if not already included
    if len(attention_probs.shape) < 5:
        attention_probs = attention_probs.unsqueeze(1)

    for head_idx in range(attention_probs.shape[1]):
        attention_head = attention_probs[0, head_idx].squeeze()

        len_z, len_y, len_x = attention_head.shape
        all_data = [
            (z, y, x, float(attention_head[z, y, x]))
            for z in range(len_z)
            for y in range(len_y)
            for x in range(len_x)
        ]

        z, y, x, att = zip(*all_data)
        trace1 = go.Scatter3d(
            x=x,
            y=z,
            z=y,
            mode="markers",
            marker=dict(
                sizemode="diameter",
                opacity=0.1,
                sizeref=1,
                size=10,
                color=att,
                colorbar=dict(title="Attention"),
            ),
        )

        z, y, x, att = zip(
            *[(z, y, x, att) for z, y, x, att in all_data if att > threshold]
        )
        trace2 = go.Scatter3d(
            x=x,
            y=z,
            z=y,
            mode="markers",
            marker=dict(
                sizemode="diameter",
                opacity=0.2,
                sizeref=1,
                size=10,
                color=att,
                colorbar=dict(title="Attention"),
            ),
        )

        layout = go.Layout(
            autosize=False,
            scene=dict(
                aspectmode="manual",
                aspectratio=dict(x=0.7, y=3, z=0.7),
                xaxis=dict(range=[0, len_x], title="width"),
                yaxis=dict(range=[0, len_z], title="length",),
                zaxis=dict(range=[0, len_y], title="height"),
            ),
            height=300,
            margin=dict(r=20, b=10, l=100, t=10),
        )

        fig = go.Figure(data=[trace1, trace2], layout=layout)

        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=2.5, y=2, z=1),
        )
        fig["layout"].update(scene=dict(camera=camera),)
        ply.iplot(fig, filename="3DBubble", config={"scrollZoom": False})
        break


def show_saliency_maps(saliency_maps, name, video=True, top_slices=0, resolution=None):
    """ The number of
    args:
        saliency_maps  (list of dicts)
    """
    saliency_map = saliency_maps[name]

    images_grad = saliency_map["scan_grad"].numpy()
    images = saliency_map["scan"].numpy()

    # saliency_tensor = images_grad.abs().max(dim=4, keepdim=False)[0]
    # saliency_tensor = (saliency_tensor / saliency_tensor.max())
    images_grad = images_grad.max(axis=4, keepdims=False)
    images_grad = images_grad - images_grad.min()
    images_grad /= images_grad.max()
    saliency_tensor = images_grad

    imgs = []
    for j in tqdm(range(images.shape[1])):
        image_tensor = images[0, j, :, :]
        saliency = saliency_tensor[0, j, :, :]
        max_saliency = saliency.max()

        image = overlay_imgs(image_tensor[:, :, 0], image_tensor[:, :, 1], alpha=0.5)
        image_saliency = overlay_imgs(
            image_tensor[:, :, 0], image_tensor[:, :, 1], saliency=saliency
        )

        saliency = plt.cm.jet(saliency)
        imgs.append((max_saliency, image, saliency, image_saliency))

    if video:
        return show_saliency_video(imgs, resolution)

    if top_slices > 0:
        for i, (max_saliency, image, saliency, image_saliency) in enumerate(
            sorted(imgs, reverse=True)
        ):
            if i == num_slices:
                break
            full = np.concatenate([image, saliency, image_saliency], axis=1)
            plt.figure(figsize=(15, 15))
            plt.imshow(full)
            plt.axis("off")
            plt.show()


@process
def show_saliency_slice(
    exam_id: str,
    image_slice: int,
    show_pet: bool = False,
    saliency: np.array = None,
    data_dir: str = "/data/fdg-pet-ct",
    dataset_name: str = "pet_ct_dataset",
    ct_hu_range: tuple = 0,
    pet_percentile: float = 0,
    saliency_percentile: float = 0,
    process_dir: str = None,
):
    """
    ### <<EXPERIMENT SPEC>> ###
    # axillary
    #experiment_dir = "experiments/manuscript/_seed/single_task/08-01_12-42_st-pretrain_mt_full_26_e/candidates/exp_15"

    #cervical 
    #experiment_dir = "experiments/manuscript/_seed/single_task/08-01_12-42_st-pretrain_mt_full_26_a/candidates/exp_19"

    #inguinal
    experiment_dir = "experiments/manuscript/_seed/single_task/08-01_12-42_st-pretrain_mt_full_26_a/candidates/exp_5"

    """
    dataset = H5Dataset(dataset_name, data_dir, mode="read")
    ct = np.array(dataset.read_images(exam_id, "CT Images"))[image_slice]
    pet = np.array(dataset.read_images(exam_id, "PET_BODY_CTAC"))[image_slice]

    plt.figure(figsize=(10, 10))
    print(ct.max())
    print(ct.min())
    # plot ct
    plt.imshow(
        plt.cm.bone(
            plt.Normalize(vmin=ct_hu_range[0], vmax=ct_hu_range[1], clip=True,)(ct)
        )
    )

    # plot pet
    # set alpha based on pet instensity
    if show_pet:
        pet = cv2.resize(pet, ct.shape)
        alphas = plt.Normalize(clip=True)(np.abs(pet))
        pet = plt.cm.plasma(
            plt.Normalize(vmin=np.percentile(pet, pet_percentile), clip=True)(pet)
        )
        pet[..., -1] = alphas
        plt.imshow(pet)

    # plot saliency
    if saliency is not None:
        saliency = saliency.max(axis=4, keepdims=False)
        saliency = saliency - saliency.min()
        saliency /= saliency.max()
        saliency = saliency[0, image_slice]
        saliency = cv2.resize(saliency, ct.shape)
        alphas = plt.Normalize(
            vmin=np.percentile(saliency, saliency_percentile), clip=True
        )(np.abs(saliency))
        saliency = plt.cm.viridis(plt.Normalize()(saliency))
        saliency[..., -1] = alphas
        plt.imshow(saliency)

    if process_dir is not None:
        plt.savefig(os.path.join(process_dir, "scan.png"))


def show_saliency_video(imgs, resolution=None):
    """Generates the Animation object for data visualization.

    We deviate from prior implementations of data visualization due
    to Jupyter notebook weirdness. We recommend placing the returned
    object in an indvidual cell so Jupyter automatically renders its
    output.

    Args:
        imgs    A list of numpy matrices representing the DICOMs.

    Returns:
        An ArtistAnimation object containing the DICOMs of interest.
    """
    fig = plt.figure(figsize=(12, 4))
    frames = []

    for i, (max_saliency, image, saliency, image_saliency) in tqdm(enumerate(imgs)):
        title = plt.text(-5, -5, max_saliency, fontsize=7.5, wrap=True)
        full = np.concatenate([image, saliency, image_saliency], axis=1)
        if resolution:
            full_dims = full.shape
            full = cv2.resize(full, resolution)
        frame = plt.imshow(full)
        frames.append([frame, title])

    ani = animation.ArtistAnimation(
        fig, frames, interval=100, blit=True, repeat_delay=1000
    )
    fig.tight_layout()
    plt.show()
    return ani


def plot_saliency_curve(saliency_maps, dim=0):
    """
    """
    data = []
    for name, saliency_map in saliency_maps.items():
        scan_grad = saliency_map["scan_grad"].numpy()
        scan = saliency_map["scan"].numpy()

        scan_grad = scan_grad.max(axis=4, keepdims=False)
        scan_grad = scan_grad - scan_grad.min()
        scan_grad /= scan_grad.max()
        saliency_tensor = scan_grad

        if dim == 0:
            average_saliency = np.array(
                [saliency_tensor[0, j, :, :].mean() for j in range(scan.shape[1])]
            )
        elif dim == 1:
            average_saliency = np.array(
                [saliency_tensor[0, :, j, :].mean() for j in range(scan.shape[2])]
            )
        elif dim == 2:
            average_saliency = np.array(
                [saliency_tensor[0, :, :, j].mean() for j in range(scan.shape[3])]
            )

        average_saliency -= average_saliency.min()
        average_saliency /= average_saliency.max()

        data.append(
            go.Scatter(
                x=np.arange(len(average_saliency)), y=average_saliency, name=f"{name}"
            )
        )
    layout = dict(
        title="Average saliency over frames",
        xaxis=dict(title="frame"),
        yaxis=dict(title="average saliency"),
    )

    ply.iplot({"data": data, "layout": layout})


def show_3d_saliency(saliency_maps, tasks=None):
    # add heads dimension if not already included
    data = []
    color_scale = cl.scales["8"]["qual"]["Set1"]
    for idx, (name, saliency_map) in enumerate(saliency_maps.items()):
        if tasks is not None and name not in tasks:
            continue
        print(name)
        scan_grad = saliency_map["scan_grad"]
        scan_grad = torch.max(scan_grad, dim=4)[0].unsqueeze(0)

        scan_grad = torch.nn.functional.max_pool3d(
            scan_grad, kernel_size=(7, 21, 21)
        ).squeeze()

        scan_grad = scan_grad.numpy()

        scan_grad = scan_grad - scan_grad.min()
        scan_grad /= scan_grad.max()
        saliency_tensor = scan_grad
        saliency_tensor -= saliency_tensor.min()
        saliency_tensor /= saliency_tensor.max()

        len_z, len_y, len_x = saliency_tensor.shape
        all_data = [
            (z, y, x, float(saliency_tensor[z, y, x]))
            for z in range(len_z)
            for y in range(len_y)
            for x in range(len_x)
        ]

        z, y, x, att = zip(*all_data)
        trace = go.Scatter3d(
            x=x,
            y=z,
            z=y,
            name=name,
            mode="markers",
            marker=dict(
                color=color_scale[idx % 8],
                sizemode="diameter",
                opacity=0.4,
                sizeref=0.05,
                size=att,
            ),
        )
        data.append(trace)
    print("Plotting...")
    layout = go.Layout(
        autosize=False,
        scene=dict(
            aspectmode="manual",
            aspectratio=dict(x=0.7, y=3, z=0.7),
            xaxis=dict(range=[0, len_x], title="width"),
            yaxis=dict(range=[0, len_z], title="length",),
            zaxis=dict(range=[0, len_y], title="height"),
        ),
        height=600,
        margin=dict(r=20, b=10, l=100, t=10),
    )

    fig = go.Figure(data=data, layout=layout)

    camera = dict(
        up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=2.5, y=2, z=1)
    )
    fig["layout"].update(scene=dict(camera=camera),)
    ply.iplot(fig, filename="3DBubble", config={"scrollZoom": False})


def plot_training_curve(epoch_data, metric, tasks=["primary"], splits=None):
    ply.offline.init_notebook_mode(connected=True)
    color_scale = cl.scales["8"]["qual"]["Set1"]

    data = []

    for idx, task in enumerate(tasks):
        if splits is not None:
            for split in splits:
                line = {
                    "color": "rgba"
                    + color_scale[idx % 8][3:-1]
                    + (",1.0)" if split == "valid" else ",0.4)")
                }
                col = f"{split}_{task}_{metric}"
                data.append(
                    go.Scatter(
                        x=epoch_data.df.index,
                        y=epoch_data.df[col],
                        name=f"{task}_{split}",
                        line=line,
                    )
                )
        else:
            col = f"{task}_{metric}" if task is not None else metric
            data.append(
                go.Scatter(x=epoch_data.df.index, y=epoch_data.df[col], name=f"{task}")
            )

    layout = dict(
        title=f"{task}-{metric}",
        xaxis=dict(title="epoch"),
        yaxis=dict(title=f"{metric}"),
    )

    ply.iplot({"data": data, "layout": layout})


def plot_learning_curves(
    experiment_dirs, metric, tasks=["primary"], splits=["train", "valid"]
):
    """
    Plots the learning curve for a set of experiments specified by experiment_dirs
    Args:
    experiments     (list or dict)  dict from name to experiment dir OR list
    metric  (str)   the metric to be plotted
    task    (str)   the task
    splits  (list)  the list of splits to be plotted
    """
    ply.init_notebook_mode(connected=True)
    color_scale = cl.scales["9"]["qual"]["Set1"]

    if type(experiment_dirs) is list:
        experiments = {
            os.path.basename(exp_dir): exp_dir
            for idx, exp_dir in enumerate(experiment_dirs)
        }

    data = []
    for idx, (name, experiment_dir) in enumerate(experiments.items()):
        if os.path.isfile(os.path.join(experiment_dir, "epochs.csv")):
            train_history = TrainHistory(experiment_dir)
            if tasks is None and splits is None:
                line = {
                    # create rgba value
                    "color": ("rgba" + color_scale[idx % 9][3:-1] + ",1.0)")
                }
                data.append(
                    go.Scatter(
                        x=train_history.df.index,
                        y=train_history.df[f"{metric}"],
                        name=f"{name}",
                        line=line,
                    )
                )
            else:
                for task in tasks:
                    for split in splits:
                        if f"{split}_{task}_{metric}" in train_history.df:
                            line = {
                                # create rgba value
                                "color": (
                                    "rgba"
                                    + color_scale[idx % 9][3:-1]
                                    + (",1.0)" if split == "valid" else ",0.4)")
                                )
                            }
                            data.append(
                                go.Scatter(
                                    x=train_history.df.index,
                                    y=train_history.df[f"{split}_{task}_{metric}"],
                                    name=f"{task}_{split}_{name}",
                                    line=line,
                                )
                            )

    layout = dict(
        title=f"{metric}", xaxis=dict(title="epoch"), yaxis=dict(title=f"{metric}")
    )
    ply.iplot({"data": data, "layout": layout})


def plot_roc_curves(
    experiment_dirs, splits=["train", "valid"], tasks=["primary"], epoch="best"
):
    """
    Plots the learning curve for a set of experiments specified by experiment_dirs
    Args:
        experiments     (list or dict)  dict from name to experiment dir OR list
        metric  (str)   the metric to be plotted
        task    (str)   the task
        splits  (list)  the list of splits to be plotted
    """
    data = []
    color_scale = cl.scales["9"]["qual"]["Set1"]

    if type(experiment_dirs) is list:
        experiment_dirs = {
            os.path.basename(exp_dir): exp_dir
            for idx, exp_dir in enumerate(experiment_dirs)
        }

    for exp_idx, (name, experiment_dir) in enumerate(experiment_dirs.items()):
        for task_idx, task in enumerate(tasks):
            for split in splits:
                line = {
                    "color": (
                        "rgba"
                        + color_scale[(len(tasks) * exp_idx + task_idx) % 9][3:-1]
                        + (",1.0)" if split == "valid" else ",0.4)")
                    )
                }

                preds_df = pd.read_csv(
                    os.path.join(experiment_dir, f"{epoch}/{split}_preds.csv"),
                    index_col=[0],
                    header=[0, 1],
                )

                correct_probs = np.array(
                    list(
                        map(
                            lambda x: float(x.strip("[]")),
                            preds_df[task, "correct_prob"].values[1:],
                        )
                    )
                )
                targets = np.array(
                    list(
                        map(
                            lambda x: int(x.strip("[]")),
                            preds_df[task, "target"].values[1:],
                        )
                    )
                )
                probs = np.abs((1 - targets) - correct_probs)

                fpr, tpr, thresholds = roc_curve(targets, probs)

                data.append(go.Scatter(x=fpr, y=tpr, name=f"{task}-{split}", line=line))

    layout = dict(
        title=f"ROC Curve",
        xaxis=dict(title="False Positive Rate"),
        yaxis=dict(title="True Positive Rate"),
    )

    ply.iplot({"data": data, "layout": layout})


def plot_roc_curve(experiment_dir, split="valid", name="best", task="primary"):
    """
    """
    preds_df = pd.read_csv(
        os.path.join(experiment_dir, f"{name}/{split}_preds.csv"),
        index_col=[0],
        header=[0, 1],
    )

    correct_probs = np.array(
        list(
            map(
                lambda x: float(x.strip("[]")),
                preds_df[task, "correct_prob"].values[1:],
            )
        )
    )
    targets = np.array(
        list(map(lambda x: int(x.strip("[]")), preds_df[task, "target"].values[1:]))
    )
    probs = np.abs((1 - targets) - correct_probs)

    fpr, tpr, thresholds = roc_curve(targets, probs)

    data = []
    data.append(go.Scatter(x=fpr, y=tpr, name=f"{name}_{split}"))
    layout = dict(
        width=500,
        height=500,
        title=f"{split} ROC",
        xaxis=dict(title="False Positive Rate"),
        yaxis=dict(title="True Positive Rate"),
    )

    ply.iplot({"data": data, "layout": layout})


def plot_metric_scatter(
    experiment_dir,
    stats_path,
    stat,
    split="valid",
    metric="roc_auc",
    name="last",
    tasks={},
):
    with open(os.path.join(experiment_dir, f"{name}/{split}_metrics.json")) as f:
        metrics_dict = json.load(f)
    metric_dict = {task: metrics[metric] for task, metrics in metrics_dict.items()}
    metric_df = pd.DataFrame.from_dict(metric_dict, orient="index", columns=[metric])

    stats_df = pd.read_csv(stats_path, index_col=1)
    stats_series = stats_df[stat].loc[tasks]
    final_df = metric_df.join(stats_series)

    final_df["name"] = final_df.index
    fig = px.scatter(final_df, x=stat, y=metric, text="name")
    fig.update_layout(height=1000)
    fig.update_traces(textposition="top center")
    return fig


def compare_experiments(
    experiments=[],
    search_subdirs=True,
    metric="roc_auc",
    epoch="last",
    splits=["valid"],
    tasks=None,
    direct_compare=False,
    package=None,
    save_path=None,
):
    """
    """
    sns.set_style("whitegrid")

    data = []

    def add_metrics(experiment_dir, name):
        if type(epoch) == dict:
            exp_epoch = epoch[name]
        else:
            exp_epoch = epoch
        for split in splits:
            metrics_path = os.path.join(
                experiment_dir, f"{exp_epoch}/{split}_metrics.json"
            )
            if not os.path.isfile(metrics_path):
                return

            with open(metrics_path) as f:
                metrics_dict = json.load(f)

            params_path = os.path.join(experiment_dir, f"params.json")
            with open(params_path) as f:
                params_dict = json.load(f)

            for task, metrics in metrics_dict.items():
                if tasks is not None and task not in tasks:
                    continue
                entry = {
                    f"{metric}": metrics[metric],
                    "name": name,
                    f"experiment_dir": experiment_dir,
                    "task": params_dict["process_args"]["dataset_args"]["class_boundaries"][0],
                    f"{metric}_label": round(metrics[metric], 4),
                    f"split": split,
                }
                data.append(entry)

    for experiment in experiments:
        name = experiment["name"]
        experiment_dir = experiment["dir"]
        if search_subdirs and os.path.isdir(os.path.join(experiment_dir, "candidates")):
            for dirname in os.listdir(os.path.join(experiment_dir, "candidates")):
                subdir = os.path.join(experiment_dir, "candidates", dirname)
                add_metrics(subdir, name)
        add_metrics(experiment_dir, name)

    if direct_compare:
        tasks = []
        for entry in data:
            tasks.append(entry["task"])
        counts = Counter(tasks)
        max_counts = max(counts.values())

        direct_compare_data = []
        for entry in data:
            if counts[entry["task"]] == max_counts:
                direct_compare_data.append(entry)

        data = direct_compare_data
    data_df = pd.DataFrame(data)
    if package == "plotly":
        mean_data_df = data_df.groupby(["task", "name"]).median().reset_index()
        fig = px.bar(
            mean_data_df,
            x="task",
            y=f"{metric}",
            color=f"name",
            barmode="group",
            height=400,
        )
        fig.update_yaxes(range=[0, 1])
        return fig

    elif package == "sns":
        plt.figure(figsize=(15, 6))
        mean_data_df = data_df.groupby(["task"]).mean().reset_index()
        order = mean_data_df.sort_values(f"{metric}")["task"]
        f = sns.barplot(data=data_df, x="task", y=f"{metric}", hue=f"name", order=order)
        sns.despine()

        labels = []
        for item in f.get_xticklabels():
            item.set_rotation(30)
            item.set_horizontalalignment("right")

    elif package == "sns_dotplot":
        plt.figure(figsize=(20, 10))
        mean_data_df = data_df.groupby(["task"]).mean().reset_index()
        order = mean_data_df.sort_values(f"{metric}")["task"]
        f = sns.stripplot(
            data=data_df,
            x="task",
            y=f"{metric}",
            hue=f"name",
            order=order,
            linewidth=4,
            size=12,
        )
        sns.despine()

        for item in f.get_xticklabels():
            item.set_rotation(30)
            item.set_horizontalalignment("right")

        plt.grid(True, axis="y")

    elif package == "stripbarplot":
        plt.figure(figsize=(8, 4))

        order = [45, 90, 180, 365]
        #order = tasks
        f = sns.stripplot(
            data=data_df,
            x="task",
            y=f"{metric}",
            hue=f"name",
            dodge=True,
            order=order,
            linewidth=0,
            size=12,
            jitter=0,
            alpha=0.4,
            zorder=1,
        )
        f = sns.barplot(
            data=data_df,
            x="task",
            y=f"{metric}",
            hue=f"name",
            order=order,
            alpha=0,
            color="grey",
            capsize=0.05,
            zorder=10,
        )
        f = sns.barplot(
            data=data_df,
            x="task",
            y=f"{metric}",
            ci=None,
            hue=f"name",
            order=order,
            alpha=1,
            color="grey",
            capsize=0.05,
            zorder=0,
        )

        sns.despine()
    else:
        raise ValueError(f"Package {package} not recognized.")

    if package and save_path is not None:
        print(save_path)
        plt.tight_layout()
        plt.savefig(save_path)

    return data_df


def unpack_experiment_groups(experiment_groups):
    """
    """
    experiments = []
    for experiment_group in experiment_groups:
        name = experiment_group["name"]
        for experiment_dir in experiment_group["dirs"]:
            experiments.append({"name": name, "dir": experiment_dir})
    return experiments


def compare_experiment_groups(
    experiment_groups,
    search_subdirs=True,
    metric="roc_auc",
    epoch="last",
    splits=["valid"],
    tasks=None,
    direct_compare=False,
    package=None,
    save_path=None,
):
    """
    """
    experiments = unpack_experiment_groups(experiment_groups)
    return compare_experiments(
        experiments,
        search_subdirs,
        metric,
        epoch,
        splits,
        tasks,
        direct_compare,
        package,
        save_path,
    )


def show_saliency_volume(
    saliency_maps, tasks=[], show_ct=True, show_pet=False, colormap=None
):

    colors = plt.get_cmap("Set1").colors
    widgets = []

    def add_controls(volume, name, color=None):
        """
        """
        level = FloatLogSlider(
            base=10, min=-0.5, max=0, step=0.0002, description=f"{name} level:"
        )
        opacity = FloatLogSlider(
            base=10, min=-2, max=1.0, step=0.01, description=f"{name} opacity:"
        )
        jslink((volume.tf, "level1"), (level, "value"))
        jslink((volume.tf, "level2"), (level, "value"))
        jslink((volume.tf, "level3"), (level, "value"))
        jslink((volume, "opacity_scale"), (opacity, "value"))

        button = Button(description=name)
        if color is not None:
            button.style.button_color = color
        controls = HBox([button, level, opacity])
        widgets.append(controls)

    ipv.clear()
    ipv.figure()

    data = {}

    for idx, task in enumerate(tasks):
        saliency_map = saliency_maps[task]
        scan_grad = saliency_map["scan_grad"]
        scan_grad = torch.max(scan_grad, dim=4)[0].unsqueeze(0)

        scan_grad = (
            scan_grad.squeeze()
        )  # torch.nn.functional.max_pool3d(scan_grad, kernel_size=(7, 21, 21)).squeeze()

        scan_grad = scan_grad.numpy()

        scan_grad = scan_grad - scan_grad.min()
        scan_grad /= scan_grad.max()
        saliency_tensor = scan_grad
        saliency_tensor -= saliency_tensor.min()
        saliency_tensor /= saliency_tensor.max()
        if colormap is None:
            color = np.array(colors[idx % len(colors)])
        else:
            color = np.array(colormap[task]) / 256
        opacity = color * 0.2
        color = "#%02x%02x%02x" % tuple(map(int, color * 256))
        saliency_vol = ipv.volshow(
            saliency_tensor,
            downscale=100,
            level=(0.5, 0.5, 0.5),
            opacity=opacity,
            controls=True,
        )
        data[task] = saliency_tensor

        add_controls(saliency_vol, name=task, color=color)

    scan = saliency_map["scan"].numpy()
    if show_ct:
        ct = scan[:, :, :, :, 0].squeeze()
        data["ct"] = ct
        ct_vol = ipv.volshow(
            ct, downscale=100, level=(1.0, 1.0, 1.0), opacity=(0.2, 0.2, 0.2)
        )
        add_controls(ct_vol, name="ct")
    if show_pet:
        pet = scan[:, :, :, :, 1].squeeze()
        data["pet"] = pet
        opacity = (np.array((228, 26, 28)) / 256) * 0.2
        pet_vol = ipv.volshow(
            pet, downscale=100, level=(0.7, 0.7, 0.7), opacity=opacity
        )
        add_controls(pet_vol, name="pet")

    ipv.style.use("minimal")
    widgets.append(ipv.gcf())
    return VBox(widgets), data


def show_attention_volume(
    scan,
    attention_maps,
    tasks=[],
    show_ct=True,
    show_pet=False,
    downscale=1,
    exam_id="",
    movie_dir=None,
    colormap=None,
    flip_axes=None,
):

    colors = plt.get_cmap("Set1").colors
    widgets = []

    def add_controls(volume, name, color=None):
        """
        """
        level = FloatLogSlider(
            base=10, min=-0.5, max=0, step=0.01, description=f"{name} level:"
        )
        opacity = FloatLogSlider(
            base=10, min=-2, max=0.6, step=0.01, description=f"{name} opacity:"
        )
        jslink((volume.tf, "level1"), (level, "value"))
        jslink((volume.tf, "level2"), (level, "value"))
        jslink((volume.tf, "level3"), (level, "value"))
        jslink((volume, "opacity_scale"), (opacity, "value"))

        button = Button(description=name)
        if color is not None:
            button.style.button_color = color
        controls = HBox([button, level, opacity])
        widgets.append(controls)

    ipv.clear()
    f = ipv.figure()

    for idx, task in enumerate(tasks):
        attention_map = attention_maps[task]
        attention_map = attention_map.cpu().detach().numpy()
        attention_map = np.mean(attention_map, axis=1)
        attention_map = attention_map[0, :, ::].squeeze()

        # scale volume up
        scan_shape = scan[:, :, :, :, 0].squeeze().shape
        for dim_idx, scan_dim in enumerate(scan_shape):
            repeat = np.round(scan_dim / attention_map.shape[dim_idx])
            attention_map = np.repeat(attention_map, repeat, axis=dim_idx)

        # set color
        if colormap is None:
            color = np.array(colors[idx % len(colors)])
        else:
            color = np.array(colormap[task]) / 256
        opacity = color * 0.2
        color = "#%02x%02x%02x" % tuple(map(int, color * 255))

        attention_map = attention_map[::downscale, ::downscale, ::downscale]
        if flip_axes is not None:
            for flip_axis in flip_axes:
                attention_map = np.flip(attention_map, axis=flip_axis)
        saliency_vol = ipv.volshow(
            attention_map,
            level=(1.0, 1.0, 1.0),
            opacity=opacity,
            controls=True,
            extent=[
                [0, attention_map.shape[0]],
                [0, attention_map.shape[1]],
                [0, attention_map.shape[2]],
            ],
        )

        add_controls(saliency_vol, name=task, color=color)

    if show_ct:
        ct = scan[:, :, :, :, 0].squeeze()
        ct = ct[::downscale, ::downscale, ::downscale]
        if flip_axes is not None:
            for flip_axis in flip_axes:
                ct = np.flip(ct, axis=flip_axis)
        ct_vol = ipv.volshow(
            ct,
            downscale=100,
            level=(0.7, 0.7, 0.7),
            opacity=(0.2, 0.2, 0.2),
            extent=[[0, ct.shape[0]], [0, ct.shape[1]], [0, ct.shape[2]]],
        )
        add_controls(ct_vol, name="ct")
    if show_pet:
        pet = scan[:, :, :, :, 1].squeeze()
        pet = pet[::downscale, ::downscale, ::downscale]
        if flip_axes is not None:
            for flip_axis in flip_axes:
                pet = np.flip(pet, axis=flip_axis)
        color = np.array(colors[0 % len(colors)])
        opacity = color * 0.2
        pet_vol = ipv.volshow(
            pet, downscale=100, level=(0.7, 0.7, 0.7), opacity=opacity
        )
        add_controls(pet_vol, name="pet")

    # ipv.xlim(0,attention_map.shape[0])
    # ipv.ylim(0,attention_map.shape[1])
    # ipv.zlim(0,attention_map.shape[2])
    # ipv.squarelim()

    # f.camera.up = (-1, -1, -1)
    # f.camera.lookAt = (0, 0, 0)
    ipv.pylab.view(0, 0, 0)
    if movie_dir is not None:
        ipv.pylab.movie(f=os.path.join(movie_dir, f"{exam_id}.gif"))

    ipv.style.use("minimal")
    widgets.append(ipv.gcf())
    return VBox(widgets)


@process
def overall_scan_perf(
    process_dir,
    experiment_groups,
    search_subdirs=True,
    epoch="best",
    metric="roc_auc",
    splits=["test"],
    tasks=None,
    plot_type="stripplot",
    order_by=None,
):
    """
    """
    experiments = []
    for experiment_group in experiment_groups:
        name = experiment_group["name"]
        for experiment_dir in experiment_group["dirs"]:
            experiments.append(
                {
                    "name": name,
                    "dir": experiment_dir,
                    "splits": experiment_group.get("splits", splits),
                }
            )

    data = []

    def add_metrics(experiment_dir, name, splits):
        for split in splits:
            metrics_path = os.path.join(experiment_dir, f"{epoch}/{split}_metrics.json")
            if not os.path.isfile(metrics_path):
                return

            with open(metrics_path) as f:
                metrics_dict = json.load(f)
            for task, metrics in metrics_dict.items():
                if tasks is not None and task not in tasks:
                    continue
                entry = {
                    f"{metric}": metrics[metric],
                    "name": name,
                    f"exeriment_dir": experiment_dir,
                    "task": task,  # .capitalize(),
                    f"{metric}_label": round(metrics[metric], 4),
                    f"split": split,
                    f"epoch": epoch,
                }
                data.append(entry)
    for experiment in experiments:
        name = experiment["name"]
        experiment_dir = experiment["dir"]
        if search_subdirs and os.path.isdir(os.path.join(experiment_dir, "candidates")):
            for dirname in os.listdir(os.path.join(experiment_dir, "candidates")):
                subdir = os.path.join(experiment_dir, "candidates", dirname)
                add_metrics(subdir, name, splits=experiment["splits"])
        else:
            add_metrics(experiment_dir, name, splits=experiment["splits"])
    data_df = pd.DataFrame(data)

    if order_by is None:
        mean_data_df = data_df.groupby(["task"]).mean().reset_index()
        order = mean_data_df.sort_values(f"{metric}")["task"]
    else:
        mean_data_df = (
            data_df[data_df.name == order_by].groupby(["task"]).mean().reset_index()
        )
        order = mean_data_df.sort_values(f"{metric}")["task"]

    # rcParams.update(
    #    {"font.size": 22, "font.family": "sans-serif", "font.sans-serif": "Arial"}
    # )

    if plot_type == "stripbarplot":
        plt.figure(figsize=(15, 10))
        f = sns.stripplot(
            data=data_df,
            x="task",
            y=f"{metric}",
            hue=f"name",
            dodge=True,
            order=order,
            linewidth=0,
            size=12,
            jitter=0,
            alpha=0.8,
            zorder=1,
        )
        f = sns.barplot(
            data=data_df,
            x="task",
            y=f"{metric}",
            hue=f"name",
            order=order,
            alpha=0,
            color="grey",
            capsize=0.05,
            zorder=10,
        )
        f = sns.barplot(
            data=data_df,
            x="task",
            y=f"{metric}",
            ci=None,
            hue=f"name",
            order=order,
            alpha=1,
            color="grey",
            capsize=0.05,
            zorder=0,
        )

    elif plot_type == "stripplot":
        plt.figure(figsize=(30, 10))
        f = sns.stripplot(
            data=data_df,
            x="task",
            y=f"{metric}",
            hue=f"name",
            dodge=True,
            order=order,
            linewidth=0,
            size=12,
        )
    elif plot_type == "boxplot":
        f = sns.boxplot(data=data_df, x="task", y=f"{metric}", hue=f"name", order=order)
    elif plot_type == "barplot":
        f = sns.barplot(data=data_df, x="task", y=f"{metric}", hue=f"name", order=order)
    elif plot_type == "violinplot":
        plt.figure(figsize=(8, 10))
        f = sns.violinplot(data=data_df, x="name", y=f"{metric}")
    elif plot_type == "all_tasks":
        plt.figure(figsize=(5, 10))
        f = sns.stripplot(
            data=data_df,
            y=f"{metric}",
            x=f"name",
            dodge=True,
            linewidth=0,
            size=12,
            jitter=1,
            alpha=0.8,
            zorder=1,
        )
        f = sns.barplot(
            data=data_df,
            y=f"{metric}",
            x=f"name",
            alpha=0,
            color="grey",
            capsize=0.05,
            zorder=10,
        )
        f = sns.barplot(
            data=data_df,
            y=f"{metric}",
            ci=None,
            x=f"name",
            alpha=1,
            color="grey",
            capsize=0.0,
            zorder=0,
        )

    sns.despine()

    for item in f.get_xticklabels():
        item.set_rotation(0)
        item.set_horizontalalignment("right")

    plt.grid(True, axis="y", zorder=20)
    plt.tight_layout()
    plt.ylim((0.25, 1))
    if process_dir is not None:
        data_df.to_csv(os.path.join(process_dir, f"data.csv"))
        plt.savefig(os.path.join(process_dir, f"{plot_type}_{metric}.pdf"))
    return data_df

