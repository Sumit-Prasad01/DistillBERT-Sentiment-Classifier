import os
import random
import numpy as np
import torch
from utils.logger import logger
from utils.custom_exception import CustomException
import sys


def setup_cuda_environment() -> None:
    """
    Configures CUDA allocator settings to prevent memory fragmentation on 4GB VRAM.
    """
    # Prevents virtual memory fragmentation on 4GB VRAM GPUs like the RTX 3050
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def get_device_info() -> dict:
    """
    Detects hardware, GPU name, compute capability, VRAM size, and CUDA version.
    """
    try:
        setup_cuda_environment()
        info = {
            "device": "cpu",
            "device_name": "CPU",
            "vram_gb": 0.0,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "supports_bf16": False,
            "supports_fp16": False,
        }

        if torch.cuda.is_available():
            info["device"] = "cuda"
            info["device_name"] = torch.cuda.get_device_name(0)
            total_mem_bytes = torch.cuda.get_device_properties(0).total_memory
            info["vram_gb"] = round(total_mem_bytes / (1024 ** 3), 2)
            info["supports_bf16"] = torch.cuda.is_bf16_supported()
            info["supports_fp16"] = True

            logger.info("=" * 60)
            logger.info("GPU HARDWARE DETECTED:")
            logger.info(f"  Device: {info['device_name']}")
            logger.info(f"  Total VRAM: {info['vram_gb']} GB")
            logger.info(f"  CUDA Version: {info['cuda_version']}")
            logger.info(f"  BF16 Supported: {info['supports_bf16']}")
            logger.info(f"  FP16 Supported: {info['supports_fp16']}")
            if info["vram_gb"] <= 4.5:
                logger.info("  [NOTICE] 4GB VRAM detected: 4GB VRAM optimization mode is ACTIVE.")
            logger.info("=" * 60)
        else:
            logger.warning("CUDA is not available. Execution will fall back to CPU.")

        return info
    except Exception as e:
        raise CustomException(e, sys)


def log_gpu_memory(stage: str = "") -> None:
    """
    Logs current allocated and max reserved memory on the GPU.
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 2)
        max_reserved = torch.cuda.max_memory_reserved(0) / (1024 ** 2)
        logger.info(
            f"[VRAM Monitor - {stage}] Allocated: {allocated:.1f} MB | "
            f"Reserved: {reserved:.1f} MB | Peak Reserved: {max_reserved:.1f} MB"
        )


def set_seed(seed: int = 42) -> None:
    """
    Sets random seed for Python, NumPy, and PyTorch (CPU & CUDA) for reproducibility.
    """
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"Random seed set to {seed} across all modules.")
    except Exception as e:
        raise CustomException(e, sys)


def setup_mlflow_environment(
    tracking_uri: str = "sqlite:///mlflow.db",
    experiment_name: str = "DistilBERT-Sentiment-Classifier"
) -> None:
    """
    Configures MLflow tracking environment variables and backend options.
    Enables support for both SQLite DB and local filesystem tracking (opt-out of deprecation exception in MLflow 3.x).
    Disables redundant raw checkpoint artifact dumping from Hugging Face Trainer.
    """
    try:
        import mlflow
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = "0"
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow initialized: Tracking URI='{tracking_uri}', Experiment='{experiment_name}'")
    except Exception as e:
        raise CustomException(e, sys)
