import argparse
import sys
import json
from src.pipeline.training_pipeline import TrainingPipeline
from src.pipeline.prediction_pipeline import PredictionPipeline
from utils.logger import logger
from utils.custom_exception import CustomException


def parse_args():
    parser = argparse.ArgumentParser(
        description="Modular DistilBERT Sentiment Classifier (Python & C++ with RTX 3050 4GB VRAM optimizations)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "predict"],
        default="train",
        help="Execution mode: 'train' (full training pipeline) or 'predict' (inference on text)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="I absolutely love this product! Best purchase ever!",
        help="Text input for prediction mode",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="artifacts/model",
        help="Path to the saved model directory for prediction",
    )
    return parser.parse_args()


def main():
    try:
        args = parse_args()

        if args.mode == "train":
            logger.info("Initializing DistilBERT Training Pipeline...")
            pipeline = TrainingPipeline(config_filepath=args.config)
            results = pipeline.run_pipeline()
            eval_metrics = results["evaluation_results"]["evaluation_metrics"]

            print("\n" + "=" * 60)
            print("TRAINING & EVALUATION SUMMARY:")
            print("=" * 60)
            print(f"Accuracy:    {eval_metrics['accuracy']:.4f}")
            print(f"Weighted F1: {eval_metrics['weighted_f1']:.4f}")
            print(f"Macro F1:    {eval_metrics['macro_f1']:.4f}")
            print(f"Model saved to: {results['training_results']['model_save_dir']}")
            print("=" * 60 + "\n")

        elif args.mode == "predict":
            logger.info(f"Running inference on: \"{args.text}\"")
            predictor = PredictionPipeline(model_dir=args.model_dir)
            result = predictor.predict(args.text)

            print("\n" + "=" * 60)
            print("PREDICTION RESULT:")
            print("=" * 60)
            print(f"Text:       \"{result['text']}\"")
            print(f"Sentiment:  {result['predicted_label'].upper()} ({result['confidence']:.2%})")
            print("Probabilities:")
            for label, prob in result["probabilities"].items():
                print(f"  {label:<10s}: {prob:.2%}")
            print("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"Execution failed: {e}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()
