import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import figure


def legend_without_duplicate_labels(
    figure, fontsize=20, ncols=1, loc="upper left", **kwargs
):
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    figure.legend(
        by_label.values(),
        by_label.keys(),
        loc=loc,
        ncols=ncols,
        fontsize=fontsize,
        facecolor="white",
        framealpha=1,
        **kwargs,
    )


def plot_marginal_predictions(
    predictions,
    confidences,
    vector_data,
    gt_marginal,
    is_available,
    x_range,
    y_range,
    add_env_legend,
    add_pred_legend,
    plot_predictions=True,
    plot_top1=False,
    dark_mode=False,
    fontsize=20,
    fontsize_legend=20,
    prediction_horizon=50,
    prediction_subsampling_rate=5,
    plot_subsampling_rate=2,
    colors_predictions=[
        "tab:purple",
        "tab:brown",
        "#39737c",
        "tab:pink",
        "tab:olive",
        "tab:orange",
    ],
):
    figure(figsize=(15, 15), dpi=80)
    vectors, idx = vector_data[:, :44], vector_data[:, 44]

    gt_marginal = gt_marginal[
        prediction_subsampling_rate
        - 1 : prediction_horizon : prediction_subsampling_rate
    ]
    is_available = is_available[
        prediction_subsampling_rate
        - 1 : prediction_horizon : prediction_subsampling_rate
    ]

    for i in np.unique(idx):
        _vectors = vectors[idx == i]
        if _vectors[:, 26:29].sum() > 0:
            label = "Road edges" if add_env_legend else ""
            plt.plot(
                _vectors[:, 0], _vectors[:, 1], color="grey", linewidth=4, label=label
            )
        elif _vectors[:, 13:16].sum() > 0:
            label = "Lane centerlines" if add_env_legend else ""
            color = "white" if dark_mode else "black"
            plt.plot(
                _vectors[:, 0], _vectors[:, 1], color=color, linewidth=2, label=label
            )
        elif _vectors[:, 16].sum() > 0:
            label = "Bike lane centerlines" if add_env_legend else ""
            plt.plot(
                _vectors[:, 0],
                _vectors[:, 1],
                "-",
                color="tab:red",
                linewidth=2,
                label=label,
            )
        elif _vectors[:, 18:26].sum() > 0:
            label = "Road lines" if add_env_legend else ""
            plt.plot(
                _vectors[:, 0],
                _vectors[:, 1],
                "--",
                color="grey",
                linewidth=2,
                label=label,
            )
        elif _vectors[:, 30:33].sum() > 0:
            label = "Misc. markings" if add_env_legend else ""
            plt.plot(
                _vectors[:, 0], _vectors[:, 1], color="grey", linewidth=2, label=label
            )

    if plot_top1:
        pred_id = np.argsort(confidences)[-1]
        confid = confidences[pred_id]
        plt.plot(
            np.concatenate(
                (
                    np.array([[0.0, 0.0]]),
                    predictions[pred_id][is_available > 0][::plot_subsampling_rate],
                )
            )[:, 0],
            np.concatenate(
                (
                    np.array([[0.0, 0.0]]),
                    predictions[pred_id][is_available > 0][::plot_subsampling_rate],
                )
            )[:, 1],
            "-o",
            color="tab:orange",
            label=f"Top 1, confid: {confid:.2f}",
            linewidth=4,
            markersize=10,
        )
    elif plot_predictions:
        for pred_id, color in zip(np.argsort(confidences), colors_predictions):
            confid = confidences[pred_id]
            label = f"Pred {pred_id}, confid: {confid:.2f}" if add_pred_legend else ""
            plt.plot(
                np.concatenate(
                    (
                        np.array([[0.0, 0.0]]),
                        predictions[pred_id][is_available > 0][::plot_subsampling_rate],
                    )
                )[:, 0],
                np.concatenate(
                    (
                        np.array([[0.0, 0.0]]),
                        predictions[pred_id][is_available > 0][::plot_subsampling_rate],
                    )
                )[:, 1],
                "-o",
                color=color,
                label=label,
                linewidth=4,
                markersize=10,
            )

    label = "Ground truth" if add_pred_legend else ""
    plt.plot(
        np.concatenate(
            (
                np.array([0.0]),
                gt_marginal[is_available > 0][:, 0][::plot_subsampling_rate],
            )
        ),
        np.concatenate(
            (
                np.array([0.0]),
                gt_marginal[is_available > 0][:, 1][::plot_subsampling_rate],
            )
        ),
        "--o",
        color="tab:cyan",
        label=label,
        linewidth=4,
        markersize=10,
    )

    # 2nd loop to plot agents on top
    for i in np.unique(idx):
        _vectors = vectors[idx == i]
        if _vectors[:, 8].sum() > 0:
            label = "Vehicles" if add_env_legend else ""
            if len(_vectors) >= 7:
                plt.plot(
                    _vectors[0:2, 0],
                    _vectors[0:2, 1],
                    linewidth=15,
                    color="tab:blue",
                    alpha=0.1,
                )
                plt.plot(
                    _vectors[2:4, 0],
                    _vectors[2:4, 1],
                    linewidth=15,
                    color="tab:blue",
                    alpha=0.2,
                )
                plt.plot(
                    _vectors[3:7, 0],
                    _vectors[3:7, 1],
                    linewidth=15,
                    color="tab:blue",
                    alpha=0.5,
                )
                plt.plot(
                    _vectors[6:, 0],
                    _vectors[6:, 1],
                    linewidth=15,
                    color="tab:blue",
                    alpha=1.0,
                    label=label,
                )
            elif len(_vectors) >= 5:
                plt.plot(
                    _vectors[0:3, 0],
                    _vectors[0:3, 1],
                    linewidth=15,
                    color="tab:blue",
                    alpha=0.2,
                )
                plt.plot(
                    _vectors[2:5, 0],
                    _vectors[2:5, 1],
                    linewidth=15,
                    color="tab:blue",
                    alpha=0.5,
                )
                plt.plot(
                    _vectors[4:6, 0],
                    _vectors[4:6, 1],
                    linewidth=15,
                    color="tab:blue",
                    alpha=1.0,
                )
            elif len(_vectors) == 1:
                plt.plot(
                    _vectors[0, 0],
                    _vectors[0, 1],
                    "s",
                    markersize=8,
                    color="tab:blue",
                    alpha=1,
                )
            else:
                plt.plot(
                    _vectors[0:-1, 0],
                    _vectors[0:-1, 1],
                    linewidth=15,
                    color="tab:blue",
                    alpha=1,
                )
        elif _vectors[:, 9].sum() > 0:
            label = "Pedestrians" if add_env_legend else ""
            plt.plot(
                _vectors[0:-1, 0],
                _vectors[0:-1, 1],
                linewidth=4,
                color="tab:red",
                alpha=0.6,
            )
            plt.plot(
                _vectors[-1, 0],
                _vectors[-1, 1],
                "o",
                markersize=8,
                color="tab:red",
                alpha=1.0,
                label=label,
            )
        elif _vectors[:, 10].sum() > 0:
            label = "Cyclists" if add_env_legend else ""
            plt.plot(
                _vectors[0:-1, 0],
                _vectors[0:-1, 1],
                "-",
                linewidth=4,
                color="tab:green",
                alpha=0.6,
            )
            plt.plot(
                _vectors[-1, 0],
                _vectors[-1, 1],
                "D",
                markersize=10,
                color="tab:green",
                alpha=1.0,
                label=label,
            )

    plt.xlim([x_range[0], x_range[1]])
    plt.ylim([y_range[0], y_range[1]])
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    if add_env_legend or add_pred_legend:
        legend_without_duplicate_labels(
            plt, fontsize=fontsize_legend, loc="upper left", bbox_to_anchor=(-0.65, 1)
        )
