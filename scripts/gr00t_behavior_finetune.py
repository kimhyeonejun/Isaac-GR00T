#!/usr/bin/env python3
"""
Fine-tuning entry point for BEHAVIOR-1K using the custom BehaviorLeRobotSingleDataset.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

SCRIPT_DIR = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import tyro
from transformers import TrainingArguments

from gr00t.data.dataset import LeRobotMixtureDataset, ModalityConfig
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import load_data_config
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING, GR00TTransform
from gr00t.utils.peft import get_lora_model

from examples.Behavior.behavior_dataset import BehaviorLeRobotSingleDataset


@dataclass
class BehaviorArgsConfig:
    """Configuration for GR00T BEHAVIOR-1K fine-tuning."""

    dataset_path: List[str]
    """Path(s) to LeRobot-formatted BEHAVIOR datasets."""

    output_dir: str = "/tmp/gr00t_behavior"
    batch_size: int = 16
    max_steps: int = 10000
    save_steps: int = 1000
    num_gpus: int = 1

    data_config: str = "examples.Behavior.custom_data_config:BehaviorDataConfig"
    """Data config to use. Defaults to the head + wrist RGB setup."""

    tasks: Optional[List[str]] = field(default=None)
    """List of BEHAVIOR `task_name` entries (see meta/tasks.jsonl)."""

    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = True
    tune_diffusion_model: bool = True

    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05

    gradient_accumulation_steps: int = 1
    dataloader_num_workers: int = 8
    dataloader_prefetch_factor: int = 4

    lora_rank: int = 0
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_full_model: bool = False

    report_to: Literal["wandb", "tensorboard", "azure_ml"] = "wandb"
    resume: bool = False

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    video_backend: Literal["torchcodec", "decord", "torchvision_av"] = "torchcodec"

    balance_dataset_weights: bool = True
    balance_trajectory_weights: bool = True


def _build_dataset(
    path: str,
    modality_configs: dict[str, ModalityConfig],
    transforms,
    embodiment_tag: EmbodimentTag,
    video_backend: str,
    tasks: Optional[List[str]],
) -> BehaviorLeRobotSingleDataset:
    dataset = BehaviorLeRobotSingleDataset(
        dataset_path=path,
        modality_configs=modality_configs,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend,
        selected_tasks=tasks,
    )
    return dataset


def _prepare_model(config: BehaviorArgsConfig, action_horizon: int, action_dim: int):
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        tune_llm=config.tune_llm,
        tune_visual=config.tune_visual,
        tune_projector=config.tune_projector,
        tune_diffusion_model=config.tune_diffusion_model,
    )

    action_head_cfg = model.action_head.config
    if (
        action_head_cfg.action_horizon != action_horizon
        or action_head_cfg.action_dim != action_dim
    ):
        import copy
        from gr00t.model.action_head.flow_matching_action_head import (
            FlowmatchingActionHead,
        )

        old_config = copy.deepcopy(action_head_cfg)
        new_config = copy.deepcopy(action_head_cfg)
        new_config.action_horizon = action_horizon
        new_config.action_dim = action_dim

        new_head = FlowmatchingActionHead(new_config)
        if old_config.action_dim == action_dim:
            new_head.load_state_dict(model.action_head.state_dict(), strict=False)
        else:
            print(
                f"Adjusting action head: horizon {old_config.action_horizon}->{action_horizon}, "
                f"dim {old_config.action_dim}->{action_dim}"
            )
            new_head.load_state_dict(model.action_head.state_dict(), strict=False)

        model.action_head = new_head
        model.config.action_head_cfg["action_horizon"] = action_horizon
        model.config.action_head_cfg["action_dim"] = action_dim
        model.config.action_dim = action_dim
        model.config.action_horizon = action_horizon
        model.action_dim = action_dim
        model.action_horizon = action_horizon
        model.action_head.set_trainable_parameters(
            tune_projector=config.tune_projector,
            tune_diffusion_model=config.tune_diffusion_model,
        )

    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    if config.lora_rank > 0:
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            action_head_only=not config.lora_full_model,
        )
    return model


def _build_training_args(config: BehaviorArgsConfig) -> TrainingArguments:
    return TrainingArguments(
        output_dir=config.output_dir,
        run_name=None,
        remove_unused_columns=False,
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        #dataloader_prefetch_factor=config.dataloader_prefetch_factor,
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=5,
        report_to=config.report_to,
        seed=42,
        do_eval=False,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
    )


def main(config: BehaviorArgsConfig):
    embodiment_tag = EmbodimentTag(config.embodiment_tag)
    data_config_cls = load_data_config(config.data_config)
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    resolved_video_backend = config.video_backend
    if resolved_video_backend == "torchcodec":
        try:
            __import__("torchcodec")
        except ImportError:
            print(
                "[WARN] torchcodec backend requested but not available. "
                "Falling back to 'torchvision_av'."
            )
            resolved_video_backend = "torchvision_av"

    if config.num_gpus > torch.cuda.device_count():
        raise ValueError(
            f"Requested num_gpus={config.num_gpus}, but only {torch.cuda.device_count()} available."
        )

    if len(config.dataset_path) == 1:
        train_dataset = _build_dataset(
            path=config.dataset_path[0],
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,
            video_backend=resolved_video_backend,
            tasks=config.tasks,
        )
    else:
        datasets = [
            _build_dataset(
                path=path,
                modality_configs=modality_configs,
                transforms=transforms,
                embodiment_tag=embodiment_tag,
                video_backend=resolved_video_backend,
                tasks=config.tasks,
            )
            for path in config.dataset_path
        ]
        train_dataset = LeRobotMixtureDataset(
            data_mixture=[(dataset, 1.0) for dataset in datasets],
            mode="train",
            balance_dataset_weights=config.balance_dataset_weights,
            balance_trajectory_weights=config.balance_trajectory_weights,
            seed=42,
            metadata_config={"percentile_mixing_method": "weighted_average"},
        )

    action_horizon = len(data_config_cls.action_indices)
    last_transform = transforms.transforms[-1]
    if not isinstance(last_transform, GR00TTransform):
        raise ValueError("The final transform must be a GR00TTransform.")
    action_dim = last_transform.max_action_dim

    model = _prepare_model(config, action_horizon, action_dim)
    training_args = _build_training_args(config)

    os.makedirs(config.output_dir, exist_ok=True)

    experiment = TrainRunner(
        train_dataset=train_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )
    experiment.train()


if __name__ == "__main__":
    args = tyro.cli(BehaviorArgsConfig)
    print("=" * 60)
    print("GR00T BEHAVIOR FINE-TUNING CONFIGURATION")
    print("=" * 60)
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("=" * 60)

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if available_gpus == 0:
        print("WARNING: No CUDA devices detected. Training will fall back to CPU.")

    if args.num_gpus > 1:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(args)
        else:
            script_path = Path(__file__).absolute()
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={args.num_gpus}",
                "--nnodes=1",
                str(script_path),
                *os.sys.argv[1:],
            ]
            print("Launching torchrun:", " ".join(cmd))
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            os.execvpe(cmd[0], cmd, env)
    else:
        if os.environ.get("CUDA_VISIBLE_DEVICES") is None and available_gpus > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        main(args)

