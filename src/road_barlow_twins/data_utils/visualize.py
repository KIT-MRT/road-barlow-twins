import numpy as np
import matplotlib.pyplot as plt

from matplotlib.pyplot import figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_marginal_predictions_3d(
    vector_data,
    predictions=None,
    confidences=None,
    is_available=None,
    gt_marginal=None,
    plot_subsampling_rate=2,
    prediction_subsampling_rate=5,
    prediction_horizon=50,
    x_range=(-50, 50),
    y_range=(-50, 50),
    dpi=80,
):
    ax = plt.figure(figsize=(15, 15), dpi=dpi).add_subplot(projection="3d")
    V = vector_data
    X, idx = V[:, :44], V[:, 44].flatten()

    car = np.array(
        [
            (-2.25, -1, 0),  # left bottom front
            (-2.25, 1, 0),  # left bottom back
            (2.25, -1, 0),  # right bottom front
            (-2.25, -1, 1.5),  # left top front -> height
        ]
    )

    pedestrian = np.array(
        [
            (-0.3, -0.3, 0),  # left bottom front
            (-0.3, 0.3, 0),  # left bottom back
            (0.3, -0.3, 0),  # right bottom front
            (-0.3, -0.3, 2),  # left top front -> height
        ]
    )

    cyclist = np.array(
        [
            (-1, -0.3, 0),  # left bottom front
            (-1, 0.3, 0),  # left bottom back
            (1, -0.3, 0),  # right bottom front
            (-1, -0.3, 2),  # left top front -> height
        ]
    )

    for i in np.unique(idx):
        _X = X[
            (idx == i)
            & (X[:, 0] < x_range[1])
            & (X[:, 1] < y_range[1])
            & (X[:, 0] > x_range[0])
            & (X[:, 1] > y_range[0])
        ]
        if _X[:, 8].sum() > 0:
            if _X[-1, 0] == 0 and _X[-1, 1] == 0:
                plt.plot(_X[:, 0], _X[:, 1], 0, linewidth=4, color="blue")
                plt.plot(_X[-1, 0], _X[-1, 1], 0, "o", markersize=10, color="blue")

            bbox = rotate_bbox_zxis(car, _X[-1, 4])
            bbox = shift_cuboid(_X[-1, 0], _X[-1, 1], bbox)

            if _X[-1, 2]:  # speed to determine dynamic or static
                add_cube(bbox, ax, color="tab:blue", alpha=0.5)
            else:
                add_cube(bbox, ax, color="tab:grey", alpha=0.5)
        elif _X[:, 9].sum() > 0:
            if _X[-1, 0] == 0 and _X[-1, 1] == 0:
                plt.plot(_X[:, 0], _X[:, 1], 0, linewidth=4, color="orange")
                plt.plot(_X[-1, 0], _X[-1, 1], 0, "o", markersize=10, color="orange")
            bbox = rotate_bbox_zxis(pedestrian, _X[-1, 4])
            bbox = shift_cuboid(_X[-1, 0], _X[-1, 1], bbox)
            add_cube(bbox, ax, color="tab:orange", alpha=0.5)
        elif _X[:, 10].sum() > 0:
            if _X[-1, 0] == 0 and _X[-1, 1] == 0:
                plt.plot(_X[:, 0], _X[:, 1], 0, linewidth=4, color="green")
                plt.plot(_X[-1, 0], _X[-1, 1], 0, "o", markersize=10, color="green")
            bbox = rotate_bbox_zxis(cyclist, _X[-1, 4])
            bbox = shift_cuboid(_X[-1, 0], _X[-1, 1], bbox)
            add_cube(bbox, ax, color="tab:green", alpha=0.5)
        elif _X[:, 13:16].sum() > 0:  # Traffic lanes
            plt.plot(_X[:, 0], _X[:, 1], 0, color="black")
        elif _X[:, 16].sum() > 0:  # Bike lanes
            plt.plot(_X[:, 0], _X[:, 1], 0, color="tab:red")
        elif _X[:, 18:26].sum() > 0:  # Road lines
            plt.plot(_X[:, 0], _X[:, 1], 0, "--", color="white")
        elif _X[:, 26:29].sum() > 0:  # Road edges
            plt.plot(_X[:, 0], _X[:, 1], 0, linewidth=2, color="white")

    ax.set_zlim(bottom=0, top=5)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_facecolor("tab:grey")

    is_available = is_available[
        prediction_subsampling_rate
        - 1 : prediction_horizon : prediction_subsampling_rate
    ]
    gt_marginal = gt_marginal[
        prediction_subsampling_rate
        - 1 : prediction_horizon : prediction_subsampling_rate
    ]

    confids_scaled = sigmoid(confidences)
    colors = plt.cm.viridis(confidences * 4)

    for pred_id in np.argsort(confidences):
        confid = confidences[pred_id]
        label = f"Pred {pred_id}, confid: {confid:.2f}" if False else ""
        confid_scaled = confids_scaled[pred_id]
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
            color=colors[pred_id],
            label=label,
            linewidth=3,  # linewidth,
            markersize=10,  # linewidth+3,
        )

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
        linewidth=4,  # 4
        markersize=10,
    )


def add_cube(cube_definition, ax, color="b", edgecolor="k", alpha=0.2):
    cube_definition_array = [np.array(list(item)) for item in cube_definition]

    points = []
    points += cube_definition_array
    vectors = [
        cube_definition_array[1] - cube_definition_array[0],
        cube_definition_array[2] - cube_definition_array[0],
        cube_definition_array[3] - cube_definition_array[0],
    ]

    points += [cube_definition_array[0] + vectors[0] + vectors[1]]
    points += [cube_definition_array[0] + vectors[0] + vectors[2]]
    points += [cube_definition_array[0] + vectors[1] + vectors[2]]
    points += [cube_definition_array[0] + vectors[0] + vectors[1] + vectors[2]]

    points = np.array(points)

    edges = [
        [points[0], points[3], points[5], points[1]],
        [points[1], points[5], points[7], points[4]],
        [points[4], points[2], points[6], points[7]],
        [points[2], points[6], points[3], points[0]],
        [points[0], points[2], points[4], points[1]],
        [points[3], points[6], points[7], points[5]],
    ]

    faces = Poly3DCollection(
        edges, linewidths=1, edgecolors=edgecolor, facecolors=color, alpha=alpha
    )

    ax.add_collection3d(faces)
    # Plot the points themselves to force the scaling of the axes
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0)


def shift_cuboid(x_shift, y_shift, cuboid):
    cuboid = np.copy(cuboid)
    cuboid[:, 0] += x_shift
    cuboid[:, 1] += y_shift

    return cuboid


def rotate_point_zaxis(p, angle):
    rot_matrix = np.array(
        [
            [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0],
            [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle)), 0],
            [0, 0, 1],
        ]
    )
    return np.matmul(p, rot_matrix)


def rotate_bbox_zxis(bbox, angle):
    bbox = np.copy(bbox)
    _bbox = []
    angle = np.rad2deg(-angle)
    for point in bbox:
        _bbox.append(rotate_point_zaxis(point, angle))

    return np.array(_bbox)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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
