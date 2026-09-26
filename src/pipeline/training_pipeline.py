import sys
from src.configuration_manager import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from utils.logger import logger
from utils.custom_exception import CustomException
from utils.helpers import set_seed, get_device_info


class TrainingPipeline:
    """
    Orchestrates the entire fine-tuning pipeline from raw data to trained model evaluation.
    """
    def __init__(self, config_filepath: str = "config/config.yaml"):
        self.config_manager = ConfigurationManager(config_filepath)
        self.model_config = self.config_manager.get_model_config()
        self.dataset_config = self.config_manager.get_dataset_config()
        self.training_config = self.config_manager.get_training_config()
        self.hardware_config = self.config_manager.get_hardware_config()
        self.path_config = self.config_manager.get_path_config()

    def run_pipeline(self) -> dict:
        """
        Executes Ingestion -> Transformation -> Training -> Evaluation.
        """
        try:
            logger.info("*" * 70)
            logger.info("STARTING DISTILBERT MODULAR TRAINING PIPELINE")
            logger.info("*" * 70)

            # 1. Environment & Hardware Detection
            get_device_info()
            set_seed(self.training_config.seed)

            # 2. Data Ingestion
            ingestion = DataIngestion(self.dataset_config, self.path_config)
            raw_paths = ingestion.initiate_data_ingestion()

            # 3. Data Transformation
            transformation = DataTransformation(
                self.model_config,
                self.dataset_config,
                self.path_config,
                self.hardware_config
            )
            transformed_data = transformation.initiate_data_transformation(raw_paths)

            # 4. Model Training (with 4GB VRAM optimizations)
            trainer_comp = ModelTrainer(
                self.model_config,
                self.training_config,
                self.path_config,
                self.hardware_config
            )
            training_results = trainer_comp.initiate_model_training(transformed_data)

            # 5. Model Evaluation on Test Split
            evaluation = ModelEvaluation(self.model_config, self.path_config)
            eval_results = evaluation.initiate_model_evaluation(
                training_results["trainer"],
                transformed_data["tokenized_datasets"]["test"]
            )

            logger.info("*" * 70)
            logger.info("DISTILBERT PIPELINE EXECUTED SUCCESSFULLY! 🎉")
            logger.info("*" * 70)

            return {
                "training_results": training_results,
                "evaluation_results": eval_results,
            }
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
