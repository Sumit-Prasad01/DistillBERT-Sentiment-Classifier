import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.logger import logger
from utils.custom_exception import CustomException
from utils.helpers import get_device_info


class PredictionPipeline:
    """
    Inference pipeline for real-time and batch sentiment predictions.
    """
    def __init__(self, model_dir: str = "artifacts/model", tracking_uri: str = None):
        try:
            self.model_dir = model_dir

            # Support loading models directly from MLflow URIs (runs:/... or models:/...)
            if model_dir.startswith(("runs:/", "models:/")):
                logger.info(f"Downloading model artifact from MLflow URI: '{model_dir}'...")
                import mlflow
                from mlflow.tracking import MlflowClient
                os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

                if tracking_uri:
                    mlflow.set_tracking_uri(tracking_uri)

                # Resolve run_name to run_id if runs:/<run_name>/... was supplied
                if model_dir.startswith("runs:/"):
                    uri_body = model_dir[len("runs:/"):]
                    parts = uri_body.split("/", 1)
                    identifier = parts[0]
                    subpath = parts[1] if len(parts) > 1 else ""

                    client = MlflowClient()
                    run_exists = False
                    try:
                        client.get_run(identifier)
                        run_exists = True
                    except Exception:
                        run_exists = False

                    if not run_exists:
                        try:
                            exp_ids = [e.experiment_id for e in client.search_experiments()]
                            matching_runs = client.search_runs(
                                experiment_ids=exp_ids,
                                filter_string=f"attributes.run_name = '{identifier}'",
                                order_by=["start_time DESC"],
                                max_results=1,
                            )
                            if matching_runs:
                                resolved_run_id = matching_runs[0].info.run_id
                                logger.info(
                                    f"Resolved MLflow run name '{identifier}' to Run ID: '{resolved_run_id}'"
                                )
                                model_dir = f"runs:/{resolved_run_id}/{subpath}"
                            else:
                                logger.warning(
                                    f"No MLflow run found with run name '{identifier}'. Attempting direct download."
                                )
                        except Exception as search_err:
                            logger.warning(f"Could not resolve run name '{identifier}': {search_err}")

                model_dir = mlflow.artifacts.download_artifacts(artifact_uri=model_dir)
                logger.info(f"Downloaded MLflow model artifact to local path: '{model_dir}'")
                self.model_dir = model_dir

            if not os.path.exists(model_dir):
                raise FileNotFoundError(
                    f"Model directory '{model_dir}' not found. Please run the training pipeline first."
                )

            device_info = get_device_info()
            self.device = torch.device(device_info["device"])
            logger.info(f"PredictionPipeline: Loading model from '{model_dir}' on {self.device}...")

            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.model.to(self.device)
            self.model.eval()

            self.id2label = self.model.config.id2label or {0: "negative", 1: "neutral", 2: "positive"}
            logger.info("PredictionPipeline: Model and tokenizer loaded successfully!")
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, text: str) -> dict:
        """
        Predicts sentiment for a single input text.
        """
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]
                pred_idx = torch.argmax(probs).item()
                confidence = probs[pred_idx].item()

            prob_dict = {
                self.id2label[i]: round(probs[i].item(), 4)
                for i in range(len(self.id2label))
            }

            return {
                "text": text,
                "predicted_label": self.id2label[pred_idx],
                "confidence": round(confidence, 4),
                "probabilities": prob_dict,
            }
        except Exception as e:
            raise CustomException(e, sys)

    def predict_batch(self, texts: list) -> list:
        """
        Predicts sentiment for a batch of input texts using dynamic padding.
        """
        try:
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred_indices = torch.argmax(probs, dim=-1).tolist()

            results = []
            for i, text in enumerate(texts):
                pred_idx = pred_indices[i]
                confidence = probs[i][pred_idx].item()
                prob_dict = {
                    self.id2label[j]: round(probs[i][j].item(), 4)
                    for j in range(len(self.id2label))
                }
                results.append({
                    "text": text,
                    "predicted_label": self.id2label[pred_idx],
                    "confidence": round(confidence, 4),
                    "probabilities": prob_dict,
                })
            return results
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    predictor = PredictionPipeline()
    sample_text = "I absolutely love this product! Best purchase ever!"
    res = predictor.predict(sample_text)
    print(res)
