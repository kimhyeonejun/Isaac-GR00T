#
# Custom dataset utilities for BEHAVIOR-1K finetuning with Isaac-GR00T.
#

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2

from gr00t.data.dataset import (
    LE_ROBOT_EPISODE_FILENAME,
    LE_ROBOT_MODALITY_FILENAME,
    LE_ROBOT_STATS_FILENAME,
    LE_ROBOT_TASKS_FILENAME,
    LeRobotSingleDataset,
    ModalityConfig,
)
from gr00t.data.schema import (
    DatasetMetadata,
    DatasetModalities,
    DatasetStatisticalValues,
    DatasetStatistics,
    EmbodimentTag,
    LeRobotModalityMetadata,
    LeRobotStateActionMetadata,
    StateActionMetadata,
    VideoMetadata,
)
from gr00t.data.transform.base import ComposedModalityTransform


@dataclass
class _AggregatedFeatureStats:
    """Helper container used while aggregating per-episode statistics."""

    min: np.ndarray
    max: np.ndarray
    mean_sum: np.ndarray
    square_sum: np.ndarray
    counts: np.ndarray
    q01_values: List[np.ndarray]
    q99_values: List[np.ndarray]

    @classmethod
    def from_episode(cls, stats: Dict[str, List[float]]) -> "_AggregatedFeatureStats":
        count = np.asarray(stats["count"], dtype=np.float64)
        mean = np.asarray(stats["mean"], dtype=np.float64)
        std = np.asarray(stats["std"], dtype=np.float64)
        min_ = np.asarray(stats["min"], dtype=np.float64)
        max_ = np.asarray(stats["max"], dtype=np.float64)
        q01 = np.asarray(stats["q01"], dtype=np.float64)
        q99 = np.asarray(stats["q99"], dtype=np.float64)
        return cls(
            min=min_,
            max=max_,
            mean_sum=mean * count,
            square_sum=(std**2 + mean**2) * count,
            counts=count,
            q01_values=[q01],
            q99_values=[q99],
        )

    def update(self, stats: Dict[str, List[float]]) -> None:
        count = np.asarray(stats["count"], dtype=np.float64)
        mean = np.asarray(stats["mean"], dtype=np.float64)
        std = np.asarray(stats["std"], dtype=np.float64)
        min_ = np.asarray(stats["min"], dtype=np.float64)
        max_ = np.asarray(stats["max"], dtype=np.float64)
        q01 = np.asarray(stats["q01"], dtype=np.float64)
        q99 = np.asarray(stats["q99"], dtype=np.float64)

        self.min = np.minimum(self.min, min_)
        self.max = np.maximum(self.max, max_)
        self.mean_sum = self.mean_sum + mean * count
        self.square_sum = self.square_sum + (std**2 + mean**2) * count
        self.counts = self.counts + count
        self.q01_values.append(q01)
        self.q99_values.append(q99)

    def finalize(self) -> Dict[str, List[float]]:
        total_count = np.maximum(self.counts, 1e-8)
        mean = self.mean_sum / total_count
        variance = np.maximum(self.square_sum / total_count - mean**2, 0.0)
        std = np.sqrt(variance)
        q01 = np.percentile(np.stack(self.q01_values, axis=0), 1, axis=0)
        q99 = np.percentile(np.stack(self.q99_values, axis=0), 99, axis=0)
        return {
            "min": self.min.tolist(),
            "max": self.max.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "q01": q01.tolist(),
            "q99": q99.tolist(),
            "count": self.counts.tolist(),
        }


def _load_task_name_to_description(dataset_path: Path) -> Dict[str, str]:
    tasks_file = dataset_path / LE_ROBOT_TASKS_FILENAME
    with open(tasks_file, "r") as f:
        entries = [json.loads(line) for line in f]
    return {entry["task_name"]: entry["task"] for entry in entries}


def _filter_episode_metadata(
    dataset_path: Path,
    task_filter: Optional[Iterable[str]],
) -> List[Dict[str, Any]]:
    episode_path = dataset_path / LE_ROBOT_EPISODE_FILENAME
    with open(episode_path, "r") as f:
        episodes = [json.loads(line) for line in f]

    if not task_filter:
        return episodes

    task_name_to_desc = _load_task_name_to_description(dataset_path)
    allowed_descriptions = {
        task_name_to_desc[name]
        for name in task_filter
        if name in task_name_to_desc
    }

    if not allowed_descriptions:
        raise ValueError(
            f"None of the requested tasks {list(task_filter)} were found in {LE_ROBOT_TASKS_FILENAME}"
        )

    filtered = [
        episode
        for episode in episodes
        if any(task in allowed_descriptions for task in episode.get("tasks", []))
    ]
    if not filtered:
        raise ValueError(
            f"No episodes matched the requested tasks {list(task_filter)}. "
            "Please double-check the task names."
        )
    return filtered


def _aggregate_episode_stats(
    dataset_path: Path,
    selected_episode_ids: Optional[set[int]],
) -> Dict[str, Dict[str, List[float]]]:
    episodes_stats_path = dataset_path / "meta" / "episodes_stats.jsonl"
    if not episodes_stats_path.exists():
        raise FileNotFoundError(
            f"episodes_stats.jsonl not found at {episodes_stats_path}. "
            "Please ensure the BEHAVIOR dataset metadata is complete."
        )

    aggregated: Dict[str, _AggregatedFeatureStats] = {}
    with open(episodes_stats_path, "r") as f:
        for line in f:
            entry = json.loads(line)
            episode_index = entry.get("episode_index")
            if selected_episode_ids is not None and episode_index not in selected_episode_ids:
                continue

            stats_dict = entry["stats"]
            for key, stats in stats_dict.items():
                if key not in aggregated:
                    aggregated[key] = _AggregatedFeatureStats.from_episode(stats)
                else:
                    aggregated[key].update(stats)

    if not aggregated:
        raise ValueError(
            "No statistics were aggregated. Ensure that the requested episodes exist "
            "in episodes_stats.jsonl."
        )

    return {key: agg.finalize() for key, agg in aggregated.items()}


class BehaviorLeRobotSingleDataset(LeRobotSingleDataset):
    """
    A thin wrapper around `LeRobotSingleDataset` that adds two BEHAVIOR-specific conveniences:

    1. If `meta/stats.json` is missing, dataset statistics are derived from
       `meta/episodes_stats.jsonl` to avoid fully recomputing statistics from parquet files.
    2. Allows filtering the dataset by a list of `task_name` entries defined in `meta/tasks.jsonl`.
    """

    def __init__(
        self,
        dataset_path: Path | str,
        modality_configs: Dict[str, ModalityConfig],
        embodiment_tag: str | EmbodimentTag,
        video_backend: str = "torchcodec",
        video_backend_kwargs: Optional[Dict[str, Any]] = None,
        transforms: Optional[ComposedModalityTransform] = None,
        selected_tasks: Optional[Iterable[str]] = None,
    ):
        self._selected_task_names = set(selected_tasks) if selected_tasks else None
        self._cached_episode_metadata: Optional[List[Dict[str, Any]]] = None
        self._cached_episode_ids: Optional[List[int]] = None
        self._video_target_resolutions: Dict[str, Tuple[int, int]] = {}
        super().__init__(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            video_backend_kwargs=video_backend_kwargs,
            transforms=transforms,
        )
        if self._selected_task_names:
            print(
                f"Filtered dataset to {len(self.trajectory_ids)} episodes "
                f"matching tasks: {sorted(self._selected_task_names)}"
            )

    # --------------------------------------------------------------------- #
    # Helper properties
    # --------------------------------------------------------------------- #
    @property
    def _filtered_episode_metadata(self) -> List[Dict[str, Any]]:
        if self._cached_episode_metadata is None:
            self._cached_episode_metadata = _filter_episode_metadata(
                self.dataset_path,
                self._selected_task_names,
            )
        return self._cached_episode_metadata

    @property
    def _filtered_episode_ids(self) -> List[int]:
        if self._cached_episode_ids is None:
            self._cached_episode_ids = [
                episode["episode_index"] for episode in self._filtered_episode_metadata
            ]
        return self._cached_episode_ids

    def get_trajectory_data(self, trajectory_id: int):  # type: ignore[override]
        df = super().get_trajectory_data(trajectory_id)
        if (
            "annotation.human.action.task_description" not in df.columns
            and "task_index" in df.columns
        ):
            df = df.copy()
            df["annotation.human.action.task_description"] = df["task_index"]
            self.curr_traj_data = df
        return df

    def get_video(self, trajectory_id: int, key: str, base_index: int) -> np.ndarray:  # type: ignore[override]
        frames = super().get_video(trajectory_id, key, base_index)
        target = self._video_target_resolutions.get(key)
        if target is None:
            return frames
        target_h, target_w = target
        current_h, current_w = frames.shape[-3], frames.shape[-2]
        if (current_h, current_w) == (target_h, target_w):
            return frames
        resized = [
            cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            for frame in frames
        ]
        return np.stack(resized, axis=0)

    # --------------------------------------------------------------------- #
    # Overrides
    # --------------------------------------------------------------------- #
    def _get_metadata(self, embodiment_tag: EmbodimentTag) -> DatasetMetadata:
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        if not modality_meta_path.exists():
            raise FileNotFoundError(
                f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"
            )

        le_modality_meta = LeRobotModalityMetadata.model_validate(
            json.loads(modality_meta_path.read_text())
        )

        info_meta_path = self.dataset_path / "meta" / "info.json"
        le_info_meta = json.loads(info_meta_path.read_text())

        state_modalities: Dict[str, StateActionMetadata] = {}
        action_modalities: Dict[str, StateActionMetadata] = {}
        for modality in ["state", "action"]:
            le_state_action_meta: Dict[str, LeRobotStateActionMetadata] = getattr(
                le_modality_meta, modality
            )
            for subkey, meta in le_state_action_meta.items():
                continuous = np.issubdtype(np.dtype(meta.dtype), np.floating)
                metadata_obj = StateActionMetadata(
                    absolute=meta.absolute,
                    rotation_type=meta.rotation_type,
                    shape=(meta.end - meta.start,),
                    continuous=bool(continuous),
                )
                if modality == "state":
                    state_modalities[subkey] = metadata_obj
                else:
                    action_modalities[subkey] = metadata_obj

        video_modalities: Dict[str, VideoMetadata] = {}
        video_resolutions: Dict[str, Tuple[int, int]] = {}
        video_channels_fps: Dict[str, Tuple[int, float]] = {}
        target_height = 0
        target_width = 0
        for new_key, field in le_modality_meta.video.items():
            original_key = field.original_key or new_key
            video_meta = le_info_meta["features"][original_key]
            names = video_meta.get("names", [])
            shape = video_meta.get("shape", [])
            info = video_meta.get("info", video_meta.get("video_info", {}))

            shape = list(shape)
            names = list(names)

            def _get_from_shape(key: str, fallback: Any):
                try:
                    return shape[names.index(key)]
                except ValueError:
                    return fallback

            height = _get_from_shape("height", video_meta.get("height") or info.get("video.height"))
            width = _get_from_shape("width", video_meta.get("width") or info.get("video.width"))
            channels = _get_from_shape("channels", info.get("video.channels"))
            fps = info.get("video.fps")

            if height is None or width is None:
                raise ValueError(f"Failed to infer resolution for video modality {new_key}")
            if channels is None:
                raise ValueError(f"Failed to infer channel count for video modality {new_key}")
            if fps is None:
                raise ValueError(f"Failed to infer FPS for video modality {new_key}")

            height = int(height)
            width = int(width)
            video_resolutions[f"video.{new_key}"] = (height, width)
            video_channels_fps[new_key] = (int(channels), float(fps))
            target_height = max(target_height, height)
            target_width = max(target_width, width)

        for new_key, (channels_val, fps_val) in video_channels_fps.items():
            video_modalities[new_key] = VideoMetadata(
                resolution=(target_width, target_height),
                channels=channels_val,
                fps=fps_val,
            )

        stats_path = self.dataset_path / LE_ROBOT_STATS_FILENAME
        if stats_path.exists():
            le_statistics = json.loads(stats_path.read_text())
        else:
            selected_episode_ids = set(self._filtered_episode_ids) if self._selected_task_names else None
            le_statistics = _aggregate_episode_stats(self.dataset_path, selected_episode_ids)

        dataset_statistics: Dict[str, Dict[str, DatasetStatisticalValues]] = {
            "state": {},
            "action": {},
        }
        for our_modality in ["state", "action"]:
            modality_dict = state_modalities if our_modality == "state" else action_modalities
            for subkey in modality_dict.keys():
                stats_dict: Dict[str, Any] = {}
                le_key = f"{our_modality}.{subkey}"
                le_meta = le_modality_meta.get_key_meta(le_key)
                channel_indices = np.arange(le_meta.start, le_meta.end)
                le_modality_key = le_meta.original_key
                modality_stats = le_statistics[le_modality_key]
                for stat_name in ["min", "max", "mean", "std", "q01", "q99"]:
                    values = np.asarray(modality_stats[stat_name])[channel_indices]
                    stats_dict[stat_name] = values
                dataset_statistics[our_modality][subkey] = DatasetStatisticalValues(**stats_dict)

        metadata = DatasetMetadata(
            statistics=DatasetStatistics(**dataset_statistics),
            modalities=DatasetModalities(
                state=state_modalities,
                action=action_modalities,
                video=video_modalities,
            ),
            embodiment_tag=embodiment_tag,
        )
        self._video_target_resolutions = {
            key: (target_height, target_width) for key in video_resolutions
        }
        return metadata

    def _get_trajectories(self) -> Tuple[np.ndarray, np.ndarray]:
        filtered_episodes = self._filtered_episode_metadata
        trajectory_ids = [episode["episode_index"] for episode in filtered_episodes]
        trajectory_lengths = [episode["length"] for episode in filtered_episodes]
        return np.asarray(trajectory_ids), np.asarray(trajectory_lengths)

