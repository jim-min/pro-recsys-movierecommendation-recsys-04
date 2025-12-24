"""
Path utility functions for managing Hydra output directories, checkpoints, and TensorBoard logs.
"""

import os
import glob
from typing import Tuple
from omegaconf import DictConfig


def get_directories(
    cfg: DictConfig,
    stage: str = "fit",
) -> Tuple[str, str]:
    """
    Get checkpoint and TensorBoard directories based on stage.

    Args:
        cfg: DictConfig containing model_name and tensorboard settings
        stage: Stage of execution ('fit' or 'predict')
            - 'fit': Creates new directories under current run_dir
            - 'predict': Finds the most recent run directory for the model

    Returns:
        Tuple[str, str]: (checkpoint_dir, tensorboard_dir)

    Raises:
        ValueError: If stage is not 'fit' or 'predict'
        FileNotFoundError: If no previous runs found in 'predict' stage

    Note:
        - cfg: 사용자 정의 설정 (model_name, tensorboard.save_dir 등)
        - HydraConfig는 함수 내부에서 자동으로 가져옴 (런타임 정보용)

    Examples:
        >>> from hydra.core.hydra_config import HydraConfig
        >>> # Training
        >>> checkpoint_dir, tensorboard_dir = get_directories(cfg, stage='fit')
        >>> # Inference
        >>> checkpoint_dir, tensorboard_dir = get_directories(cfg, stage='predict')
    """

    if stage not in ["fit", "predict"]:
        raise ValueError(f"stage must be 'fit' or 'predict', got: {stage}")

    # cfg에서 사용자 정의 설정 가져오기
    model_name = cfg.model_name
    tensorboard_base = cfg.tensorboard.save_dir

    if stage == "fit":
        # Training: Use current Hydra output directory
        from hydra.core.hydra_config import HydraConfig

        hydra_cfg = HydraConfig.get()
        run_dir = hydra_cfg.runtime.output_dir

        # Checkpoint directory: run_dir/checkpoints
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # TensorBoard directory: saved/tensorboard_logs/{model_name}/YYYY-MM-DD/HH-MM-SS
        run_parts = run_dir.split(os.sep)
        run_timestamp = os.path.join(run_parts[-2], run_parts[-1])  # 날짜/시간
        tensorboard_dir = os.path.join(tensorboard_base, model_name, run_timestamp)
        os.makedirs(tensorboard_dir, exist_ok=True)

    else:  # stage == 'predict'
        # Inference: Find the most recent run directory
        # HydraConfig에서 실행 디렉토리 패턴 가져오기
        from hydra.core.hydra_config import HydraConfig

        hydra_cfg = HydraConfig.get()
        current_run_dir = hydra_cfg.runtime.output_dir

        # run_dir 구조: {base}/{model_name}/YYYY-MM-DD/HH-MM-SS
        # 예: saved/hydra_logs/bert4rec/2025-12-22/12-34-56
        # hydra_logs_base 추출: saved/hydra_logs/bert4rec
        run_parts = current_run_dir.split(os.sep)
        # 마지막 2개(날짜/시간)를 제외한 부분
        hydra_logs_base = os.sep.join(run_parts[:-2])

        if not os.path.exists(hydra_logs_base):
            raise FileNotFoundError(
                f"No previous runs found for model '{model_name}' at {hydra_logs_base}"
            )

        # Find all run directories with checkpoints: {hydra_logs_base}/YYYY-MM-DD/HH-MM-SS/checkpoints
        pattern = os.path.join(hydra_logs_base, "*", "*", "checkpoints")
        checkpoint_dirs = glob.glob(pattern)

        if not checkpoint_dirs:
            raise FileNotFoundError(
                f"No checkpoint directories found in {hydra_logs_base}/**/checkpoints"
            )

        # Sort by modification time and get the most recent checkpoint directory
        checkpoint_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        checkpoint_dir = checkpoint_dirs[0]

        # Extract run_dir from checkpoint_dir
        # checkpoint_dir: saved/hydra_logs/bert4rec/2025-12-22/12-34-56/checkpoints
        # run_dir: saved/hydra_logs/bert4rec/2025-12-22/12-34-56
        run_dir = os.path.dirname(checkpoint_dir)

        # TensorBoard directory: saved/tensorboard_logs/{model_name}/YYYY-MM-DD/HH-MM-SS
        run_parts = run_dir.split(os.sep)
        run_timestamp = os.path.join(run_parts[-2], run_parts[-1])  # 날짜/시간
        tensorboard_dir = os.path.join(tensorboard_base, model_name, run_timestamp)

    return checkpoint_dir, tensorboard_dir


def get_latest_checkpoint(
    checkpoint_dir: str, checkpoint_name: str = "last.ckpt"
) -> str:
    """
    Get the path to a checkpoint file in the checkpoint directory.

    Args:
        checkpoint_dir: Directory containing checkpoints
        checkpoint_name: Name of the checkpoint file (default: "last.ckpt")

    Returns:
        str: Full path to the checkpoint file

    Raises:
        FileNotFoundError: If checkpoint file not found

    Examples:
        >>> checkpoint_path = get_latest_checkpoint("/path/to/checkpoints")
        >>> checkpoint_path = get_latest_checkpoint("/path/to/checkpoints", "best.ckpt")
    """
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    if not os.path.exists(checkpoint_path):
        # Try to find any .ckpt file
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))

        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

        # Return the most recent checkpoint
        ckpt_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        checkpoint_path = ckpt_files[0]

    return checkpoint_path
