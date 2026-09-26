import os
import sys
import pandas as pd
from datasets import load_dataset, DatasetDict
from src.configuration_manager import DatasetConfig, PathConfig
from utils.logger import logger
from utils.custom_exception import CustomException


class DataIngestion:
    """
    Component for downloading and caching the Tweet Sentiment Multilingual dataset.
    """
    def __init__(self, dataset_config: DatasetConfig, path_config: PathConfig):
        self.dataset_config = dataset_config
        self.path_config = path_config

    def initiate_data_ingestion(self) -> dict:
        """
        Loads the dataset from Hugging Face Hub and saves raw parquet files.
        """
        try:
            logger.info("=" * 60)
            logger.info("STARTING DATA INGESTION")
            logger.info(f"Dataset: {self.dataset_config.dataset_name} ({self.dataset_config.dataset_subset})")
            logger.info("=" * 60)

            raw_dir = self.path_config.raw_data_dir
            train_path = os.path.join(raw_dir, "train.parquet")
            val_path = os.path.join(raw_dir, "validation.parquet")
            test_path = os.path.join(raw_dir, "test.parquet")

            # Check if cached files already exist
            if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
                logger.info(f"Cached raw dataset found in: {raw_dir}. Loading from local disk...")
                return {
                    "train_path": train_path,
                    "validation_path": val_path,
                    "test_path": test_path,
                }

            logger.info("Downloading dataset from Hugging Face Hub...")
            dataset: DatasetDict = load_dataset(
                self.dataset_config.dataset_name,
                revision=self.dataset_config.revision,
                data_dir=self.dataset_config.dataset_subset,
            )

            logger.info(f"Successfully loaded dataset structure:\n{dataset}")

            # Convert to pandas for local parquet caching
            train_df = dataset["train"].to_pandas()
            val_df = dataset["validation"].to_pandas()
            test_df = dataset["test"].to_pandas()

            train_df.to_parquet(train_path, index=False)
            val_df.to_parquet(val_path, index=False)
            test_df.to_parquet(test_path, index=False)

            logger.info(f"Saved raw splits to: {raw_dir}")
            logger.info(f"  Train samples: {len(train_df)}")
            logger.info(f"  Validation samples: {len(val_df)}")
            logger.info(f"  Test samples: {len(test_df)}")

            # Log class distribution
            class_counts = train_df["label"].value_counts().sort_index()
            logger.info("Class distribution in training set:")
            label_names = ["negative", "neutral", "positive"]
            for label_idx, count in class_counts.items():
                name = label_names[label_idx] if label_idx < len(label_names) else f"class_{label_idx}"
                logger.info(f"  {name:10s}: {count:4d} samples ({100 * count / len(train_df):.1f}%)")

            return {
                "train_path": train_path,
                "validation_path": val_path,
                "test_path": test_path,
            }
        except Exception as e:
            raise CustomException(e, sys)
