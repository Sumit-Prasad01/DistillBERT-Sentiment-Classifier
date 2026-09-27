import os
import sys
import yaml
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv
from utils.custom_exception import CustomException
from utils.logger import logger

load_dotenv()


@dataclass(frozen=True)
class ModelConfig:
    model_name: str
    num_labels: int
    label_names: List[str]


@dataclass(frozen=True)
class DatasetConfig:
    dataset_name: str
    dataset_subset: str
    revision: str
    max_length: int
    seed: int


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: str
    learning_rate: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    weight_decay: float
    eval_strategy: str
    save_strategy: str
    load_best_model_at_end: bool
    metric_for_best_model: str
    fp16: bool
    bf16: bool
    optim: str
    logging_steps: int
    report_to: str
    seed: int
    save_total_limit: int


@dataclass(frozen=True)
class HardwareConfig:
    force_device: str
    pin_memory: bool
    num_workers: int
    enable_cpp_acceleration: bool


@dataclass(frozen=True)
class PathConfig:
    raw_data_dir: str
    transformed_data_dir: str
    model_save_dir: str
    metrics_dir: str
    log_dir: str


@dataclass(frozen=True)
class MLflowConfig:
    tracking_uri: str
    experiment_name: str
    run_name: str
    log_models: bool


class ConfigurationManager:
    """
    Manages loading and validation of system configuration from YAML.
    """
    def __init__(self, config_filepath: str = "config/config.yaml"):
        try:
            self.config_filepath = config_filepath
            self.config = self._read_yaml_file(config_filepath)
            self._create_directories()
        except Exception as e:
            raise CustomException(e, sys)

    def _read_yaml_file(self, filepath: str) -> dict:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found at: {filepath}")
        with open(filepath, "r", encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"Loaded configuration file from: {filepath}")
            return content

    def _create_directories(self) -> None:
        paths = self.get_path_config()
        dirs = [
            paths.raw_data_dir,
            paths.transformed_data_dir,
            paths.model_save_dir,
            paths.metrics_dir,
            paths.log_dir,
        ]
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Ensured directory exists: {directory}")

    def get_model_config(self) -> ModelConfig:
        cfg = self.config["model"]
        return ModelConfig(
            model_name=cfg["model_name"],
            num_labels=cfg["num_labels"],
            label_names=cfg["label_names"],
        )

    def get_dataset_config(self) -> DatasetConfig:
        cfg = self.config["dataset"]
        return DatasetConfig(
            dataset_name=cfg["dataset_name"],
            dataset_subset=cfg["dataset_subset"],
            revision=cfg["revision"],
            max_length=cfg["max_length"],
            seed=cfg["seed"],
        )

    def get_training_config(self) -> TrainingConfig:
        cfg = self.config["training"]
        return TrainingConfig(
            output_dir=cfg["output_dir"],
            learning_rate=float(cfg["learning_rate"]),
            per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
            per_device_eval_batch_size=int(cfg["per_device_eval_batch_size"]),
            gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
            num_train_epochs=int(cfg["num_train_epochs"]),
            weight_decay=float(cfg["weight_decay"]),
            eval_strategy=cfg["eval_strategy"],
            save_strategy=cfg["save_strategy"],
            load_best_model_at_end=bool(cfg["load_best_model_at_end"]),
            metric_for_best_model=cfg["metric_for_best_model"],
            fp16=bool(cfg["fp16"]),
            bf16=bool(cfg["bf16"]),
            optim=cfg["optim"],
            logging_steps=int(cfg["logging_steps"]),
            report_to=cfg["report_to"],
            seed=int(cfg["seed"]),
            save_total_limit=int(cfg["save_total_limit"]),
        )

    def get_hardware_config(self) -> HardwareConfig:
        cfg = self.config["hardware"]
        return HardwareConfig(
            force_device=cfg["force_device"],
            pin_memory=bool(cfg["pin_memory"]),
            num_workers=int(cfg["num_workers"]),
            enable_cpp_acceleration=bool(cfg["enable_cpp_acceleration"]),
        )

    def get_path_config(self) -> PathConfig:
        cfg = self.config["paths"]
        return PathConfig(
            raw_data_dir=cfg["raw_data_dir"],
            transformed_data_dir=cfg["transformed_data_dir"],
            model_save_dir=cfg["model_save_dir"],
            metrics_dir=cfg["metrics_dir"],
            log_dir=cfg["log_dir"],
        )

    def get_mlflow_config(self) -> MLflowConfig:
        cfg = self.config.get("mlflow", {})
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or cfg.get("tracking_uri", "sqlite:///mlflow.db")
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME") or cfg.get("experiment_name", "DistilBERT-Sentiment-Classifier")
        run_name = os.getenv("MLFLOW_RUN_NAME") or cfg.get("run_name", "distilbert-base-uncased-run")
        
        env_log_models = os.getenv("MLFLOW_LOG_MODELS")
        if env_log_models is not None:
            log_models = env_log_models.strip().lower() in ("true", "1", "yes")
        else:
            log_models = bool(cfg.get("log_models", True))

        return MLflowConfig(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            run_name=run_name,
            log_models=log_models,
        )
