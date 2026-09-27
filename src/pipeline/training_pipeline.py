import os
import sys
import mlflow
import mlflow.transformers
from src.configuration_manager import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from utils.logger import logger
from utils.custom_exception import CustomException
from utils.helpers import set_seed, get_device_info, setup_mlflow_environment


class TrainingPipeline:
    """
    Orchestrates the entire fine-tuning pipeline from raw data to trained model evaluation,
    with end-to-end MLflow experiment tracking, metrics, artifacts, and model logging.
    """
    def __init__(
        self,
        config_filepath: str = "config/config.yaml",
        experiment_name: str = None,
        run_name: str = None,
        tracking_uri: str = None,
    ):
        self.config_filepath = config_filepath
        self.config_manager = ConfigurationManager(config_filepath)
        self.model_config = self.config_manager.get_model_config()
        self.dataset_config = self.config_manager.get_dataset_config()
        self.training_config = self.config_manager.get_training_config()
        self.hardware_config = self.config_manager.get_hardware_config()
        self.path_config = self.config_manager.get_path_config()
        self.mlflow_config = self.config_manager.get_mlflow_config()

        # Configurable overrides for MLflow
        self.tracking_uri = tracking_uri or self.mlflow_config.tracking_uri
        self.experiment_name = experiment_name or self.mlflow_config.experiment_name
        self.run_name = run_name or self.mlflow_config.run_name
        self.log_models = self.mlflow_config.log_models

    def run_pipeline(self) -> dict:
        """
        Executes Ingestion -> Transformation -> Training -> Evaluation
        under an active MLflow tracking run.
        """
        try:
            logger.info("*" * 70)
            logger.info("STARTING DISTILBERT MODULAR TRAINING PIPELINE")
            logger.info("*" * 70)

            # 1. Environment & Hardware Detection
            device_info = get_device_info()
            set_seed(self.training_config.seed)

            # 2. Setup MLflow Tracking
            setup_mlflow_environment(
                tracking_uri=self.tracking_uri,
                experiment_name=self.experiment_name
            )

            with mlflow.start_run(run_name=self.run_name) as run:
                run_id = run.info.run_id
                artifact_uri = run.info.artifact_uri
                logger.info(f"Active MLflow Run started: ID='{run_id}', Name='{self.run_name}'")

                # Set Run Tags
                mlflow.set_tags({
                    "model_architecture": self.model_config.model_name,
                    "task": "sentiment_classification",
                    "dataset": f"{self.dataset_config.dataset_name}:{self.dataset_config.dataset_subset}",
                    "device": device_info.get("device", "cpu"),
                    "gpu_name": device_info.get("device_name", "None"),
                    "cuda_available": str(device_info.get("cuda_available", False)),
                    "cpp_acceleration": str(self.hardware_config.enable_cpp_acceleration),
                })

                # Log High-Level Pipeline Parameters
                mlflow.log_params({
                    "base_model": self.model_config.model_name,
                    "num_labels": self.model_config.num_labels,
                    "label_names": ", ".join(self.model_config.label_names),
                    "dataset_name": self.dataset_config.dataset_name,
                    "dataset_subset": self.dataset_config.dataset_subset,
                    "max_seq_length": self.dataset_config.max_length,
                    "effective_batch_size": self.training_config.per_device_train_batch_size * self.training_config.gradient_accumulation_steps,
                    "fp16_enabled": self.training_config.fp16,
                    "optimizer": self.training_config.optim,
                    "metric_for_best_model": self.training_config.metric_for_best_model,
                    "hardware_device": device_info.get("device_name", "cpu"),
                    "vram_gb": device_info.get("vram_gb", 0.0),
                })

                # Log Configuration Artifact
                if os.path.exists(self.config_filepath):
                    mlflow.log_artifact(self.config_filepath, artifact_path="config")

                # 3. Data Ingestion
                ingestion = DataIngestion(self.dataset_config, self.path_config)
                raw_paths = ingestion.initiate_data_ingestion()

                # 4. Data Transformation
                transformation = DataTransformation(
                    self.model_config,
                    self.dataset_config,
                    self.path_config,
                    self.hardware_config
                )
                transformed_data = transformation.initiate_data_transformation(raw_paths)

                # Log Split Sizes
                tokenized_datasets = transformed_data["tokenized_datasets"]
                mlflow.log_params({
                    "train_samples": len(tokenized_datasets["train"]),
                    "validation_samples": len(tokenized_datasets["validation"]),
                    "test_samples": len(tokenized_datasets["test"]),
                })

                # 5. Model Training (with 4GB VRAM optimizations & MLflow step logging)
                trainer_comp = ModelTrainer(
                    self.model_config,
                    self.training_config,
                    self.path_config,
                    self.hardware_config
                )
                training_results = trainer_comp.initiate_model_training(transformed_data)

                # 6. Model Evaluation on Test Split (logs test metrics & confusion matrix plot)
                evaluation = ModelEvaluation(self.model_config, self.path_config)
                eval_results = evaluation.initiate_model_evaluation(
                    training_results["trainer"],
                    tokenized_datasets["test"]
                )

                # 7. Model Artifact Logging to MLflow
                if self.log_models:
                    save_dir = training_results["model_save_dir"]
                    logger.info(f"Logging saved model directory to MLflow from: {save_dir}...")
                    mlflow.log_artifacts(save_dir, artifact_path="model")

                    # Log MLflow transformers flavor for pyfunc serving
                    try:
                        mlflow.transformers.log_model(
                            transformers_model={
                                "model": training_results["trainer"].model,
                                "tokenizer": transformed_data["tokenizer"],
                            },
                            artifact_path="mlflow_transformers_model",
                            task="text-classification",
                        )
                        logger.info("Logged MLflow transformers flavor model successfully.")
                    except Exception as e:
                        logger.warning(f"MLflow transformers flavor model notice: {e}")

                logger.info("*" * 70)
                logger.info("DISTILBERT PIPELINE EXECUTED SUCCESSFULLY! 🎉")
                logger.info(f"MLflow Run ID: {run_id}")
                logger.info(f"MLflow Tracking URI: {self.tracking_uri}")
                logger.info("*" * 70)

                return {
                    "training_results": training_results,
                    "evaluation_results": eval_results,
                    "mlflow": {
                        "run_id": run_id,
                        "experiment_name": self.experiment_name,
                        "run_name": self.run_name,
                        "tracking_uri": self.tracking_uri,
                        "artifact_uri": artifact_uri,
                    },
                }
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
