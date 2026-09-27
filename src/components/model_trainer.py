import os
import sys
import numpy as np
import torch
import evaluate
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from src.configuration_manager import ModelConfig, TrainingConfig, PathConfig, HardwareConfig
from utils.logger import logger
from utils.custom_exception import CustomException
from utils.helpers import get_device_info, log_gpu_memory

# Check for C++ metrics acceleration
try:
    import sentiment_cpp_accel
    HAS_CPP_METRICS = True
except ImportError:
    HAS_CPP_METRICS = False


class ModelTrainer:
    """
    Component for fine-tuning DistilBERT on 4GB VRAM GPU with FP16,
    gradient accumulation, fused AdamW, and TorchScript export.
    """
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        path_config: PathConfig,
        hardware_config: HardwareConfig
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.path_config = path_config
        self.hardware_config = hardware_config

    def _get_metrics_function(self):
        """
        Builds compute_metrics callable, prioritizing C++ fast_metrics when available.
        """
        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)

            if HAS_CPP_METRICS and self.hardware_config.enable_cpp_acceleration:
                try:
                    preds_list = predictions.tolist()
                    refs_list = labels.tolist()
                    cpp_res = sentiment_cpp_accel.compute_metrics_cpp(
                        preds_list, refs_list, self.model_config.num_labels
                    )
                    return {
                        "accuracy": cpp_res.accuracy,
                        "f1_weighted": cpp_res.weighted_f1,
                        "f1_macro": cpp_res.macro_f1,
                    }
                except Exception as e:
                    logger.warning(f"C++ metrics calculation failed: {e}. Using fallback evaluate metrics.")

            # Fallback to evaluate package
            acc = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
            f1_w = f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
            f1_m = f1_metric.compute(predictions=predictions, references=labels, average="macro")["f1"]

            return {
                "accuracy": acc,
                "f1_weighted": f1_w,
                "f1_macro": f1_m,
            }

        return compute_metrics

    def initiate_model_training(self, transformed_data: dict) -> dict:
        """
        Executes model initialization, training with 4GB VRAM optimizations,
        and saves artifacts.
        """
        try:
            device_info = get_device_info()
            device_type = device_info["device"]

            logger.info("=" * 60)
            logger.info("STARTING MODEL TRAINING")
            logger.info(f"Base Model: {self.model_config.model_name}")
            logger.info(f"Target Classes: {self.model_config.label_names}")
            logger.info(f"Micro-Batch Size: {self.training_config.per_device_train_batch_size}")
            logger.info(f"Gradient Accumulation Steps: {self.training_config.gradient_accumulation_steps}")
            logger.info(
                f"Effective Batch Size: {self.training_config.per_device_train_batch_size * self.training_config.gradient_accumulation_steps}"
            )
            logger.info(f"Mixed Precision (FP16): {self.training_config.fp16 and device_type == 'cuda'}")
            logger.info("=" * 60)

            # Label mappings
            label2id = {label: i for i, label in enumerate(self.model_config.label_names)}
            id2label = {i: label for i, label in enumerate(self.model_config.label_names)}

            # Initialize Model
            logger.info("Loading pre-trained model weights...")
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_config.model_name,
                num_labels=self.model_config.num_labels,
                id2label=id2label,
                label2id=label2id,
            )

            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Total Parameters: {total_params:,}")
            logger.info(f"Trainable Parameters: {trainable_params:,}")

            # Determine precision and optimizer for 4GB VRAM
            use_fp16 = self.training_config.fp16 and (device_type == "cuda")
            use_bf16 = self.training_config.bf16 and (device_type == "cuda") and device_info["supports_bf16"]
            optimizer_name = "adamw_torch_fused" if device_type == "cuda" else "adamw_torch"

            training_args = TrainingArguments(
                output_dir=self.training_config.output_dir,
                learning_rate=self.training_config.learning_rate,
                per_device_train_batch_size=self.training_config.per_device_train_batch_size,
                per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
                gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
                num_train_epochs=self.training_config.num_train_epochs,
                weight_decay=self.training_config.weight_decay,
                eval_strategy=self.training_config.eval_strategy,
                save_strategy=self.training_config.save_strategy,
                load_best_model_at_end=self.training_config.load_best_model_at_end,
                metric_for_best_model=self.training_config.metric_for_best_model,
                fp16=use_fp16,
                bf16=use_bf16,
                optim=optimizer_name,
                logging_steps=self.training_config.logging_steps,
                report_to=self.training_config.report_to,
                seed=self.training_config.seed,
                save_total_limit=self.training_config.save_total_limit,
                dataloader_pin_memory=self.hardware_config.pin_memory,
                dataloader_num_workers=self.hardware_config.num_workers,
            )

            tokenized_datasets = transformed_data["tokenized_datasets"]
            data_collator = transformed_data["data_collator"]
            tokenizer = transformed_data["tokenizer"]

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["validation"],
                data_collator=data_collator,
                compute_metrics=self._get_metrics_function(),
            )

            log_gpu_memory("Pre-Training")

            logger.info("Executing training loop...")
            train_result = trainer.train()

            log_gpu_memory("Post-Training")

            # Save best fine-tuned model and tokenizer
            save_dir = self.path_config.model_save_dir
            os.makedirs(save_dir, exist_ok=True)

            logger.info(f"Saving fine-tuned model to: {save_dir}")
            trainer.save_model(save_dir)
            tokenizer.save_pretrained(save_dir)

            # Export TorchScript model for C++ LibTorch inference
            torchscript_path = os.path.join(save_dir, "model.pt")
            self._export_torchscript(model, tokenizer, torchscript_path)

            logger.info("Training completed successfully!")
            for metric, value in train_result.metrics.items():
                if isinstance(value, float):
                    logger.info(f"  {metric}: {value:.4f}")

            # Log summary training metrics to MLflow if active run exists
            try:
                import mlflow
                if mlflow.active_run():
                    clean_train_metrics = {
                        f"train_{k}" if not k.startswith("train_") else k: float(v)
                        for k, v in train_result.metrics.items()
                        if isinstance(v, (int, float))
                    }
                    mlflow.log_metrics(clean_train_metrics)
                    logger.info("Logged training summary metrics to active MLflow run.")
            except Exception as e:
                logger.warning(f"Could not log training metrics to MLflow: {e}")

            return {
                "model_save_dir": save_dir,
                "torchscript_path": torchscript_path,
                "trainer": trainer,
                "train_metrics": train_result.metrics,
            }
        except Exception as e:
            raise CustomException(e, sys)

    def _export_torchscript(self, model, tokenizer, output_path: str) -> None:
        """
        Exports the fine-tuned model to TorchScript for standalone C++ LibTorch deployment.
        """
        try:
            logger.info(f"Exporting model to TorchScript at: {output_path}...")
            model.eval()
            dummy_text = "Sample sentence for sentiment tracing"
            inputs = tokenizer(dummy_text, return_tensors="pt")

            # Trace model
            with torch.no_grad():
                traced_model = torch.jit.trace(
                    model,
                    (inputs["input_ids"], inputs["attention_mask"]),
                    strict=False,
                )
                traced_model.save(output_path)
            logger.info("TorchScript model export successful!")
        except Exception as e:
            logger.warning(f"TorchScript export skipped or encountered notice: {e}")
