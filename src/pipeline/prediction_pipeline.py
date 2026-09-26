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
    def __init__(self, model_dir: str = "artifacts/model"):
        try:
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
