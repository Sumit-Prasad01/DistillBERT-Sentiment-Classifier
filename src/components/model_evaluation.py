import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from src.configuration_manager import ModelConfig, PathConfig
from utils.logger import logger
from utils.custom_exception import CustomException

# Check for C++ metrics acceleration
try:
    import sentiment_cpp_accel
    HAS_CPP_METRICS = True
except ImportError:
    HAS_CPP_METRICS = False


class ModelEvaluation:
    """
    Component for evaluating model performance on the test split,
    generating classification reports, and plotting confusion matrices.
    """
    def __init__(self, model_config: ModelConfig, path_config: PathConfig):
        self.model_config = model_config
        self.path_config = path_config

    def initiate_model_evaluation(self, trainer, test_dataset) -> dict:
        """
        Runs prediction on the test dataset, generates metrics report and plots confusion matrix.
        """
        try:
            logger.info("=" * 60)
            logger.info("STARTING MODEL EVALUATION (TEST SET)")
            logger.info("=" * 60)

            metrics_dir = self.path_config.metrics_dir
            os.makedirs(metrics_dir, exist_ok=True)

            # Predict on test split
            logger.info("Generating predictions on test set...")
            predictions_output = trainer.predict(test_dataset)
            logits = predictions_output.predictions
            labels = predictions_output.label_ids
            predictions = np.argmax(logits, axis=-1)

            # 1. Classification Report
            report_str = classification_report(
                labels, predictions, target_names=self.model_config.label_names
            )
            report_dict = classification_report(
                labels, predictions, target_names=self.model_config.label_names, output_dict=True
            )

            logger.info(f"\nClassification Report:\n{report_str}")

            # 2. Confusion Matrix Calculation (via C++ if available or sklearn)
            if HAS_CPP_METRICS:
                try:
                    cpp_metrics = sentiment_cpp_accel.compute_metrics_cpp(
                        predictions.tolist(), labels.tolist(), self.model_config.num_labels
                    )
                    cm = np.array(cpp_metrics.confusion_matrix)
                    logger.info("Confusion matrix computed via C++ fast_metrics engine.")
                except Exception as e:
                    logger.warning(f"C++ confusion matrix calculation fallback: {e}")
                    cm = confusion_matrix(labels, predictions)
            else:
                cm = confusion_matrix(labels, predictions)

            # 3. Plot Confusion Matrix
            cm_plot_path = os.path.join(metrics_dir, "confusion_matrix.png")
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=self.model_config.label_names,
                yticklabels=self.model_config.label_names,
            )
            plt.title("DistilBERT Tweet Sentiment Confusion Matrix")
            plt.xlabel("Predicted Sentiment")
            plt.ylabel("True Sentiment")
            plt.tight_layout()
            plt.savefig(cm_plot_path, dpi=300)
            plt.close()
            logger.info(f"Saved confusion matrix plot to: {cm_plot_path}")

            # 4. Save JSON Metrics
            eval_metrics = {
                "accuracy": report_dict["accuracy"],
                "macro_f1": report_dict["macro avg"]["f1-score"],
                "weighted_f1": report_dict["weighted avg"]["f1-score"],
                "detailed_report": report_dict,
                "confusion_matrix": cm.tolist(),
            }

            metrics_json_path = os.path.join(metrics_dir, "evaluation_metrics.json")
            with open(metrics_json_path, "w", encoding="utf-8") as f:
                json.dump(eval_metrics, f, indent=4)
            logger.info(f"Saved metrics summary to: {metrics_json_path}")

            logger.info("=" * 60)
            logger.info("FINAL TEST RESULTS:")
            logger.info(f"  Test Accuracy:    {eval_metrics['accuracy']:.4f}")
            logger.info(f"  Macro F1 Score:   {eval_metrics['macro_f1']:.4f}")
            logger.info(f"  Weighted F1 Score:{eval_metrics['weighted_f1']:.4f}")
            logger.info("=" * 60)

            return {
                "evaluation_metrics": eval_metrics,
                "confusion_matrix_path": cm_plot_path,
                "metrics_json_path": metrics_json_path,
            }
        except Exception as e:
            raise CustomException(e, sys)
