import os
import sys
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding
from src.configuration_manager import ModelConfig, DatasetConfig, PathConfig, HardwareConfig
from utils.logger import logger
from utils.custom_exception import CustomException

# Check if C++ acceleration module is available
try:
    import sentiment_cpp_accel
    HAS_CPP_ACCEL = True
except ImportError:
    HAS_CPP_ACCEL = False


class DynamicPaddingCollator:
    """
    Data collator that dynamically pads each batch to the maximum sequence length
    in that batch, rather than padding the entire dataset to 128 upfront.
    Uses C++ fast_collator when compiled; otherwise falls back to PyTorch DataCollatorWithPadding.
    """
    def __init__(self, tokenizer, max_allowed_len: int = 128, use_cpp: bool = True):
        self.tokenizer = tokenizer
        self.max_allowed_len = max_allowed_len
        self.use_cpp = use_cpp and HAS_CPP_ACCEL
        self.fallback_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

        if self.use_cpp:
            logger.info("DynamicPaddingCollator: [C++ ACCELERATION ACTIVE] Using fast_collator_cpp.")
        else:
            logger.info("DynamicPaddingCollator: Using PyTorch DataCollatorWithPadding.")

    def __call__(self, features: list) -> dict:
        if self.use_cpp:
            try:
                token_ids_list = [f["input_ids"] for f in features]
                labels_list = [f["labels"] for f in features]
                pad_token_id = self.tokenizer.pad_token_id or 0

                batch_res = sentiment_cpp_accel.collate_batch_cpp(
                    token_ids_list,
                    labels_list,
                    pad_token_id,
                    self.max_allowed_len
                )

                b_size = batch_res.batch_size
                seq_len = batch_res.max_seq_len

                input_ids = torch.tensor(batch_res.input_ids, dtype=torch.long).view(b_size, seq_len)
                attention_mask = torch.tensor(batch_res.attention_mask, dtype=torch.long).view(b_size, seq_len)
                labels = torch.tensor(batch_res.labels, dtype=torch.long)

                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                }
            except Exception as e:
                logger.warning(f"C++ collation failed with error: {e}. Falling back to PyTorch collator.")
                return self.fallback_collator(features)
        else:
            return self.fallback_collator(features)


class DataTransformation:
    """
    Component for tokenizing datasets and configuring dynamic batch padding.
    """
    def __init__(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        path_config: PathConfig,
        hardware_config: HardwareConfig
    ):
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.path_config = path_config
        self.hardware_config = hardware_config

    def initiate_data_transformation(self, raw_data_paths: dict) -> dict:
        """
        Loads raw parquet splits, applies tokenization, and saves transformed datasets.
        """
        try:
            logger.info("=" * 60)
            logger.info("STARTING DATA TRANSFORMATION")
            logger.info(f"Tokenizer: {self.model_config.model_name}")
            logger.info(f"Max sequence length clamp: {self.dataset_config.max_length}")
            logger.info("=" * 60)

            # Load Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_name)
            tokenizer.model_max_length = self.dataset_config.max_length

            # Load Parquet splits into datasets.Dataset
            train_df = pd.read_parquet(raw_data_paths["train_path"])
            val_df = pd.read_parquet(raw_data_paths["validation_path"])
            test_df = pd.read_parquet(raw_data_paths["test_path"])

            raw_datasets = DatasetDict({
                "train": Dataset.from_pandas(train_df),
                "validation": Dataset.from_pandas(val_df),
                "test": Dataset.from_pandas(test_df),
            })

            def tokenize_function(examples):
                # Truncate without static padding for dynamic batching efficiency
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=self.dataset_config.max_length,
                )

            logger.info("Applying tokenization across train, validation, and test splits...")
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                desc="Tokenizing splits"
            )

            # Clean and prepare columns
            tokenized_datasets = tokenized_datasets.remove_columns(["text"])
            if "label" in tokenized_datasets["train"].column_names:
                tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

            # Save transformed datasets to disk
            transformed_dir = self.path_config.transformed_data_dir
            save_path = os.path.join(transformed_dir, "tokenized_datasets")
            tokenized_datasets.save_to_disk(save_path)
            logger.info(f"Saved tokenized datasets to: {save_path}")

            # Instantiate dynamic collator
            data_collator = DynamicPaddingCollator(
                tokenizer=tokenizer,
                max_allowed_len=self.dataset_config.max_length,
                use_cpp=self.hardware_config.enable_cpp_acceleration
            )

            # Log sample verification
            sample = tokenized_datasets["train"][0]
            logger.info("Sample tokenized record:")
            logger.info(f"  Input IDs length: {len(sample['input_ids'])}")
            logger.info(f"  Label: {sample['labels']}")

            return {
                "transformed_dataset_path": save_path,
                "tokenized_datasets": tokenized_datasets,
                "tokenizer": tokenizer,
                "data_collator": data_collator,
            }
        except Exception as e:
            raise CustomException(e, sys)
