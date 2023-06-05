import torch
import numpy as np
import pandas as pd

from glob import glob
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset
from l5kit.evaluation.metrics import (
    average_displacement_error_oracle,
    final_displacement_error_oracle,
    rmse,
    neg_multi_log_likelihood,
)

from data_utils.dataset_modules import WaymoLoader, WaymoRoadEnvGraphDataset


def run_eval_dataframe(
    model,
    data,
    use_top1=False,
    n_samples=20_181,
    prediction_horizons=[30, 50],
    red_model=False,
):
    model.to("cuda")
    slicing_step_size = len(glob(f"{data}/*.npz")) // n_samples

    if red_model:
        loader = DataLoader(
            Subset(
                WaymoRoadEnvGraphDataset(data),
                torch.arange(0, n_samples * slicing_step_size, slicing_step_size),
            ),
            batch_size=1,
            num_workers=16,
            shuffle=False,
        )
        model.road_env_encoder.batch_size = 1
        model.road_env_encoder.range_decoder_embedding = torch.arange(100).expand(
            1, 100
        )
    else:
        loader = DataLoader(
            Subset(
                WaymoLoader(data, return_vector=True),
                torch.arange(0, n_samples * slicing_step_size, slicing_step_size),
            ),
            batch_size=1,
            num_workers=16,
            shuffle=False,
        )

    n_prediction_horizons = len(prediction_horizons)
    neg_log_likelihood_scores = [[] for _ in range(n_prediction_horizons)]
    rmse_scores = [[] for _ in range(n_prediction_horizons)]
    min_ade_scores = [[] for _ in range(n_prediction_horizons)]
    min_fde_scores = [[] for _ in range(n_prediction_horizons)]
    n_sample = 0

    with torch.no_grad():
        # for x, y, is_available, _ in tqdm(loader):
        for batch in tqdm(loader):
            # x, y, is_available = map(lambda x: x.cuda(), (x, y, is_available))
            if red_model:
                is_available = batch["future_ego_trajectory"]["is_available"].to("cuda")
                y = batch["future_ego_trajectory"]["trajectory"].to("cuda")

                env_idxs_src_tokens = batch["sample_a"]["idx_src_tokens"].to("cuda")
                env_pos_src_tokens = batch["sample_a"]["pos_src_tokens"].to("cuda")
                env_src_mask = batch["src_attn_mask"].to("cuda")
                ego_idxs_semantic_embedding = batch["past_ego_trajectory"][
                    "idx_semantic_embedding"
                ].to("cuda")
                ego_pos_src_tokens = batch["past_ego_trajectory"]["pos_src_tokens"].to(
                    "cuda"
                )

                confidences_logits, logits = model(
                    env_idxs_src_tokens,
                    env_pos_src_tokens,
                    env_src_mask,
                    ego_idxs_semantic_embedding,
                    ego_pos_src_tokens,
                )
            else:
                x, y, is_available, _ = batch
                x, y, is_available = map(lambda x: x.cuda(), (x, y, is_available))
                confidences_logits, logits = model(x)

            argmax = confidences_logits.argmax()
            if use_top1:
                confidences_logits = confidences_logits[:, argmax].unsqueeze(1)
                logits = logits[:, argmax].unsqueeze(1)

            confidences = torch.softmax(confidences_logits, dim=1)

            logits_np = logits.squeeze(0).cpu().numpy()
            y_np = y.squeeze(0).cpu().numpy()
            is_available_np = is_available.squeeze(0).long().cpu().numpy()
            confidences_np = confidences.squeeze(0).cpu().numpy()

            for idx, prediction_horizon in enumerate(prediction_horizons[::-1]):
                y_np = y_np[:prediction_horizon]
                is_available_np = is_available_np[:prediction_horizon]
                logits_np = logits_np[:, :prediction_horizon]

                neg_log_likelihood_scores[idx].append(
                    neg_multi_log_likelihood(
                        ground_truth=y_np,
                        pred=logits_np,
                        confidences=confidences_np,
                        avails=is_available_np,
                    )
                )

                rmse_scores[idx].append(
                    rmse(
                        ground_truth=y_np,
                        pred=logits_np,
                        confidences=confidences_np,
                        avails=is_available_np,
                    )
                )

                min_ade_scores[idx].append(
                    average_displacement_error_oracle(
                        ground_truth=y_np,
                        pred=logits_np,
                        confidences=confidences_np,
                        avails=is_available_np,
                    )
                )

                min_fde_scores[idx].append(
                    final_displacement_error_oracle(
                        ground_truth=y_np,
                        pred=logits_np,
                        confidences=confidences_np,
                        avails=is_available_np,
                    )
                )

            n_sample += 1

            if n_sample == n_samples:
                break

    res = pd.DataFrame(
        {
            **{
                f"NLL @{prediction_horizon}": np.round(
                    np.mean(neg_log_likelihood_scores[idx]), 3
                )
                for idx, prediction_horizon in enumerate(prediction_horizons[::-1])
            },
            **{
                f"RMSE @{prediction_horizon}": np.round(np.mean(rmse_scores[idx]), 3)
                for idx, prediction_horizon in enumerate(prediction_horizons[::-1])
            },
            **{
                f"minADE @{prediction_horizon}": np.round(
                    np.mean(min_ade_scores[idx]), 3
                )
                for idx, prediction_horizon in enumerate(prediction_horizons[::-1])
            },
            **{
                f"minFDE @{prediction_horizon}": np.round(
                    np.mean(min_fde_scores[idx]), 3
                )
                for idx, prediction_horizon in enumerate(prediction_horizons[::-1])
            },
        },
        index=[0],
    )

    res["meanNLL"] = np.mean(
        [
            res[f"NLL @{prediction_horizon}"]
            for prediction_horizon in prediction_horizons
        ]
    )
    res["meanRMSE"] = np.mean(
        [
            res[f"RMSE @{prediction_horizon}"]
            for prediction_horizon in prediction_horizons
        ]
    )
    res["meanADE"] = np.mean(
        [
            res[f"minADE @{prediction_horizon}"]
            for prediction_horizon in prediction_horizons
        ]
    )
    res["meanFDE"] = np.mean(
        [
            res[f"minFDE @{prediction_horizon}"]
            for prediction_horizon in prediction_horizons
        ]
    )

    return res


def run_waymo_eval_per_class(
    model,
    data,
    use_top1=False,
    prediction_horizons=[30, 50],
    red_model=False,
    n_samples_per_class=5_000,
    prediction_subsampling_rate=1,
):
    """Agent classes: Vehicle, pedestrian, cyclist"""
    model.to("cuda")

    class_names = ["vehicle", "pedestrian", "cyclist"]
    vehicle_samples = f"{data}/vehicle*.npz"
    pedestrian_samples = f"{data}/pedestrian*.npz"
    cyclist_samples = f"{data}/cyclist*.npz"

    slicing_step_size_vehicle = len(glob(vehicle_samples)) // n_samples_per_class
    slicing_step_size_pedestrian = len(glob(pedestrian_samples)) // n_samples_per_class
    slicing_step_size_cyclist = len(glob(cyclist_samples)) // n_samples_per_class

    if red_model:
        vehicle_loader = DataLoader(
            Subset(
                WaymoRoadEnvGraphDataset(directory="", glob_path=vehicle_samples),
                torch.arange(
                    0,
                    n_samples_per_class * slicing_step_size_vehicle,
                    slicing_step_size_vehicle,
                ),
            ),
            batch_size=1,
            num_workers=16,
            shuffle=False,
        )
        pedestrian_loader = DataLoader(
            Subset(
                WaymoRoadEnvGraphDataset(directory="", glob_path=pedestrian_samples),
                torch.arange(
                    0,
                    n_samples_per_class * slicing_step_size_pedestrian,
                    slicing_step_size_pedestrian,
                ),
            ),
            batch_size=1,
            num_workers=16,
            shuffle=False,
        )
        cyclist_loader = DataLoader(
            Subset(
                WaymoRoadEnvGraphDataset(directory="", glob_path=cyclist_samples),
                torch.arange(
                    0,
                    n_samples_per_class * slicing_step_size_cyclist,
                    slicing_step_size_cyclist,
                ),
            ),
            batch_size=1,
            num_workers=16,
            shuffle=False,
        )
        data_loaders = [vehicle_loader, pedestrian_loader, cyclist_loader]

        model.road_env_encoder.batch_size = 1
        model.road_env_encoder.range_decoder_embedding = torch.arange(100).expand(
            1, 100
        )

    res_per_class = pd.DataFrame()
    n_prediction_horizons = len(prediction_horizons)

    with torch.no_grad():
        for loader, class_name in zip(data_loaders, class_names):
            print(f"Starting eval for agent class: {class_name}")

            neg_log_likelihood_scores = [[] for _ in range(n_prediction_horizons)]
            rmse_scores = [[] for _ in range(n_prediction_horizons)]
            min_ade_scores = [[] for _ in range(n_prediction_horizons)]
            min_fde_scores = [[] for _ in range(n_prediction_horizons)]

            for batch in tqdm(loader):
                if red_model:
                    is_available = batch["future_ego_trajectory"]["is_available"].to(
                        "cuda"
                    )
                    y = batch["future_ego_trajectory"]["trajectory"].to("cuda")

                    env_idxs_src_tokens = batch["sample_a"]["idx_src_tokens"].to("cuda")
                    env_pos_src_tokens = batch["sample_a"]["pos_src_tokens"].to("cuda")
                    env_src_mask = batch["src_attn_mask"].to("cuda")
                    ego_idxs_semantic_embedding = batch["past_ego_trajectory"][
                        "idx_semantic_embedding"
                    ].to("cuda")
                    ego_pos_src_tokens = batch["past_ego_trajectory"][
                        "pos_src_tokens"
                    ].to("cuda")

                    confidences_logits, logits = model(
                        env_idxs_src_tokens,
                        env_pos_src_tokens,
                        env_src_mask,
                        ego_idxs_semantic_embedding,
                        ego_pos_src_tokens,
                    )
                else:
                    x, y, is_available, _ = batch
                    x, y, is_available = map(lambda x: x.cuda(), (x, y, is_available))
                    confidences_logits, logits = model(x)

                argmax = confidences_logits.argmax()
                if use_top1:
                    confidences_logits = confidences_logits[:, argmax].unsqueeze(1)
                    logits = logits[:, argmax].unsqueeze(1)

                confidences = torch.softmax(confidences_logits, dim=1)

                logits_np = logits.squeeze(0).cpu().numpy()
                y_np = y.squeeze(0).cpu().numpy()
                is_available_np = is_available.squeeze(0).long().cpu().numpy()
                confidences_np = confidences.squeeze(0).cpu().numpy()

                for idx, prediction_horizon in enumerate(prediction_horizons[::-1]):
                    y_np = y_np[
                        (
                            prediction_subsampling_rate - 1
                        ) : prediction_horizon : prediction_subsampling_rate
                    ]
                    is_available_np = is_available_np[
                        (
                            prediction_subsampling_rate - 1
                        ) : prediction_horizon : prediction_subsampling_rate
                    ]
                    logits_np = logits_np[
                        :, : (prediction_horizon // prediction_subsampling_rate)
                    ]

                    neg_log_likelihood_scores[idx].append(
                        neg_multi_log_likelihood(
                            ground_truth=y_np,
                            pred=logits_np,
                            confidences=confidences_np,
                            avails=is_available_np,
                        )
                    )

                    rmse_scores[idx].append(
                        rmse(
                            ground_truth=y_np,
                            pred=logits_np,
                            confidences=confidences_np,
                            avails=is_available_np,
                        )
                    )

                    min_ade_scores[idx].append(
                        average_displacement_error_oracle(
                            ground_truth=y_np,
                            pred=logits_np,
                            confidences=confidences_np,
                            avails=is_available_np,
                        )
                    )

                    min_fde_scores[idx].append(
                        final_displacement_error_oracle(
                            ground_truth=y_np,
                            pred=logits_np,
                            confidences=confidences_np,
                            avails=is_available_np,
                        )
                    )
            for idx, prediction_horizon in enumerate(prediction_horizons[::-1]):
                res_per_class[f"{class_name} NLL @{prediction_horizon}"] = [
                    np.round(np.mean(neg_log_likelihood_scores[idx]), 3)
                ]
                res_per_class[f"{class_name} RMSE @{prediction_horizon}"] = [
                    np.round(np.mean(rmse_scores[idx]), 3)
                ]
                res_per_class[f"{class_name} minADE @{prediction_horizon}"] = [
                    np.round(np.mean(min_ade_scores[idx]), 3)
                ]
                res_per_class[f"{class_name} minFDE @{prediction_horizon}"] = [
                    np.round(np.mean(min_fde_scores[idx]), 3)
                ]

    res = pd.DataFrame(
        {
            **{
                f"NLL @{prediction_horizon}": np.round(
                    np.mean(
                        [
                            res_per_class[f"{class_name} NLL @{prediction_horizon}"]
                            for class_name in class_names
                        ]
                    ),
                    3,
                )
                for prediction_horizon in prediction_horizons[::-1]
            },
            **{
                f"RMSE @{prediction_horizon}": np.round(
                    np.mean(
                        [
                            res_per_class[f"{class_name} RMSE @{prediction_horizon}"]
                            for class_name in class_names
                        ]
                    ),
                    3,
                )
                for prediction_horizon in prediction_horizons[::-1]
            },
            **{
                f"minADE @{prediction_horizon}": np.round(
                    np.mean(
                        [
                            res_per_class[f"{class_name} minADE @{prediction_horizon}"]
                            for class_name in class_names
                        ]
                    ),
                    3,
                )
                for prediction_horizon in prediction_horizons[::-1]
            },
            **{
                f"minFDE @{prediction_horizon}": np.round(
                    np.mean(
                        [
                            res_per_class[f"{class_name} minFDE @{prediction_horizon}"]
                            for class_name in class_names
                        ]
                    ),
                    3,
                )
                for prediction_horizon in prediction_horizons[::-1]
            },
        },
        index=[0],
    )

    res["meanNLL"] = np.mean(
        [
            res[f"NLL @{prediction_horizon}"]
            for prediction_horizon in prediction_horizons
        ]
    )
    res["meanRMSE"] = np.mean(
        [
            res[f"RMSE @{prediction_horizon}"]
            for prediction_horizon in prediction_horizons
        ]
    )
    res["meanADE"] = np.mean(
        [
            res[f"minADE @{prediction_horizon}"]
            for prediction_horizon in prediction_horizons
        ]
    )
    res["meanFDE"] = np.mean(
        [
            res[f"minFDE @{prediction_horizon}"]
            for prediction_horizon in prediction_horizons
        ]
    )

    return res, res_per_class
