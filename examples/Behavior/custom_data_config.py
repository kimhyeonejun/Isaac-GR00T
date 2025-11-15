#
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#

from gr00t.data.transform.base import ComposedModalityTransform, ModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import StateActionToTensor, StateActionTransform
from gr00t.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from gr00t.experiment.data_config import BaseDataConfig
from gr00t.model.transforms import GR00TTransform


class BehaviorDataConfig(BaseDataConfig):
    """
    Data configuration for the BEHAVIOR-1K LeRobot dataset.

    Although the simulator logs a 256-D proprio state, we only feed a curated 64-D subset:
    the 23-D Open-PI joints plus trunk/arm velocities, end-effector poses, robot pose, and
    linear/angular velocities. This keeps the model input compact while exposing enough
    dynamics for the action head.
    defined in `PROPRIOCEPTION_INDICES["R1Pro"]`. This matches the signals consumed by
    downstream policies while keeping the original 23-D absolute joint actions.
    """

    # Visual observations: include head plus both wrist RGB streams.
    video_keys = [
        "video.rgb_head",
        "video.rgb_left_wrist",
        "video.rgb_right_wrist",
    ]

    # Proprioceptive state keys (mapped in modality.json).
    state_keys = [
        "state.base_qvel",
        "state.robot_pos",
        "state.trunk_qpos",
        "state.arm_left_qpos",
        "state.arm_right_qpos",
        "state.gripper_left_qpos",
        "state.gripper_right_qpos",
        "state.trunk_qvel",
        "state.arm_left_qvel",
        "state.arm_right_qvel",
        "state.eef_left_pos",
        "state.eef_right_pos",
        "state.eef_left_quat",
        "state.eef_right_quat",
        "state.robot_lin_vel",
        "state.robot_ang_vel",
    ]

    # Absolute joint position actions (23 dims).
    action_keys = [
        "action.base",
        "action.torso",
        "action.left_arm",
        "action.left_gripper",
        "action.right_arm",
        "action.right_gripper",
    ]

    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def transform(self, action_norm: str = "min_max") -> ModalityTransform:
        """
        Build the modality transform pipeline.

        Args:
            action_norm: Normalization strategy for actions.
                * "min_max" (default) uses min / max statistics per dimension.
                * "mean_std" uses mean / standard deviation for arms & torso while
                  keeping grippers on min / max to respect joint limits.
        """

        if action_norm not in {"min_max", "mean_std"}:
            raise ValueError(f"Unsupported action normalization mode: {action_norm}")

        state_norm_modes = {key: "min_max" for key in self.state_keys}

        if action_norm == "min_max":
            action_norm_modes = {key: "min_max" for key in self.action_keys}
        else:
            action_norm_modes = {
                "action.base": "mean_std",
                "action.torso": "mean_std",
                "action.left_arm": "mean_std",
                "action.left_gripper": "min_max",
                "action.right_arm": "mean_std",
                "action.right_gripper": "min_max",
            }

        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes=state_norm_modes,
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes=action_norm_modes,
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]

        return ComposedModalityTransform(transforms=transforms)


class BehaviorDataConfigMeanStd(BehaviorDataConfig):
    """Variant that applies mean/std normalization to arm/base joints."""

    def transform(self) -> ModalityTransform:  # type: ignore[override]
        return super().transform(action_norm="mean_std")

