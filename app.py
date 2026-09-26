import sys
from src.pipeline.prediction_pipeline import PredictionPipeline
from utils.helpers import get_device_info


def display_result(res: dict):
    print("\n" + "-" * 60)
    print(f"Input:       \"{res['text']}\"")
    sentiment = res['predicted_label'].upper()
    confidence = res['confidence'] * 100
    print(f"Sentiment:   {sentiment} (Confidence: {confidence:.1f}%)")
    print("Class Probabilities:")
    for label, prob in res["probabilities"].items():
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {label:<10s} [{bar}] {prob * 100:5.1f}%")
    print("-" * 60 + "\n")


def run_benchmark_test(predictor: PredictionPipeline):
    print("\nRunning benchmark test on sample texts from notebook...")
    test_texts = [
        "I absolutely love this product! Best purchase ever!",
        "This is terrible. Complete waste of money.",
        "The package arrived on time. Nothing special.",
        "Can't believe how amazing the customer service was!",
        "Not impressed. Expected much better quality.",
        "It's okay, does what it's supposed to do.",
    ]
    results = predictor.predict_batch(test_texts)
    for res in results:
        display_result(res)


def main():
    print("=" * 65)
    print("  DistilBERT Sentiment Classifier - Interactive Console")
    print("  Optimized for NVIDIA RTX 3050 Laptop GPU (4GB VRAM) with CUDA")
    print("=" * 65)

    info = get_device_info()
    print(f"Active Device: {info['device_name']} ({info['vram_gb']} GB VRAM)")

    try:
        predictor = PredictionPipeline()
    except Exception as e:
        print(f"\n[Error loading model]: {e}")
        print("Please train the model first by running: python main.py --mode train\n")
        return

    while True:
        print("\nOptions:")
        print("  1. Enter custom text for sentiment analysis")
        print("  2. Run benchmark test (notebook test examples)")
        print("  3. Exit")
        choice = input("Select an option (1/2/3): ").strip()

        if choice == "1":
            text = input("\nEnter text to analyze: ").strip()
            if text:
                res = predictor.predict(text)
                display_result(res)
            else:
                print("Text cannot be empty.")
        elif choice == "2":
            run_benchmark_test(predictor)
        elif choice in ("3", "exit", "quit", "q"):
            print("Exiting application. Goodbye!")
            break
        else:
            print("Invalid selection. Please choose 1, 2, or 3.")


if __name__ == "__main__":
    main()
