import torch
import numpy as np

from torch.distributions.uniform import Uniform


class RoadEnvGraphAugmentations:
    def __init__(
        self,
        min_rot_angle: float = -10.0,
        max_rot_angle: float = 10.0,
        min_shift_x: float = -1.0,
        max_shift_x: float = 1.0,
        min_shift_y: float = -1.0,
        max_shift_y: float = 1.0,
    ) -> None:
        self.rot_distribution = Uniform(low=min_rot_angle, high=max_rot_angle)
        self.shift_x_distribution = Uniform(low=min_shift_x, high=max_shift_x)
        self.shift_y_distribution = Uniform(low=min_shift_y, high=max_shift_y)

    def rotate_vectors(self, v, angle):
        x, y = v[:, 0], v[:, 1]
        x_rot = x * torch.cos(torch.deg2rad(angle)) - y * torch.sin(
            torch.deg2rad(angle)
        )
        y_rot = y * torch.cos(torch.deg2rad(angle)) + x * torch.sin(
            torch.deg2rad(angle)
        )
        v[:, 0], v[:, 1] = x_rot, y_rot

        return v

    def __call__(self, sample):
        sample_a = sample
        sample_b = np.copy(sample)

        sample_a = torch.from_numpy(sample_a)
        sample_b = torch.from_numpy(sample_b)

        rot_angle = self.rot_distribution.sample()
        x_shift = self.shift_x_distribution.sample()
        y_shift = self.shift_y_distribution.sample()

        sample_b = self.rotate_vectors(sample_b, rot_angle)
        sample_b[:, 0] += x_shift
        sample_b[:, 1] += y_shift

        return (sample_a, sample_b)


def waymo_one_hot_to_embedding_idx(waymo_vectors):
    # Road graph vector: (x, y, idx_embedding)
    vectors = np.zeros((len(waymo_vectors), 3))
    vectors[:, 0:2] = waymo_vectors[:, 0:2]
    vectors[:, 2] = (
        0 * waymo_vectors[:, 13]
        + 1 * waymo_vectors[:, 14]
        + 2 * waymo_vectors[:, 15]
        + 3 * waymo_vectors[:, 16]
        + 4 * waymo_vectors[:, 8]
        + 6 * waymo_vectors[:, 9]
        + 8 * waymo_vectors[:, 10]
        + (waymo_vectors[:, 2] > 0.0).astype(float)
    )  # Speed to determine if static or dynamic

    return vectors


def waymo_vectors_to_road_env_graph(
    waymo_vectors: np.ndarray,
    max_dist: float = 55.0,
    lane_sampling_rate: int = 3,
    agent_radius: float = 30.0,
) -> np.ndarray:
    vectors, idx_global = waymo_vectors[:, :45], waymo_vectors[:, 44].flatten()
    road_lanes = []
    agents = []

    for idx in np.unique(idx_global):
        _vectors = vectors[idx_global == idx]

        if _vectors[:, 13:17].sum() > 0:
            road_lanes.append(_vectors)
        # Agent trajectories to current agent position if in radius of interest
        elif _vectors[:, 5:12].sum() > 0:
            distance = np.sqrt(_vectors[-1][0] ** 2 + _vectors[-1][1] ** 2)
            if distance <= agent_radius:
                agents.append(_vectors[-1])

    road_lanes_sub = np.array([])

    for lane in road_lanes:
        last_idx = lane.shape[0] - 1
        _lane = [
            elem
            for idx, elem in enumerate(lane)
            if (not idx % lane_sampling_rate or idx in [0, last_idx])
            and (abs(elem[0] / max_dist) < 1 and abs(elem[1] / max_dist) < 1)
        ]
        _lane = np.array(_lane)

        if len(_lane):
            if not len(road_lanes_sub):
                road_lanes_sub = _lane
            else:
                road_lanes_sub = np.concatenate((road_lanes_sub, _lane))

    agents = np.array(agents)

    if len(agents) and len(road_lanes_sub):
        lanes_and_agents = np.concatenate((road_lanes_sub, agents))
    elif not len(agents):
        lanes_and_agents = road_lanes_sub
    else:
        lanes_and_agents = agents
    
    road_graph = waymo_one_hot_to_embedding_idx(lanes_and_agents)

    return road_graph


def waymo_vectors_to_past_ego_trajectory(waymo_vectors, semantic_offset=4):
    vectors, idx_global = waymo_vectors[:, :45], waymo_vectors[:, 44].flatten()

    for idx in np.unique(idx_global):
        _vectors = vectors[idx_global == idx]

        if _vectors[:, 5:12].sum() > 0:
            distance = np.sqrt(_vectors[-1][0]**2 + _vectors[-1][1]**2)
            if distance == 0.0:
                past_ego_trajectory = waymo_one_hot_to_embedding_idx(_vectors)

    past_ego_trajectory[:, 2] -= semantic_offset

    return past_ego_trajectory