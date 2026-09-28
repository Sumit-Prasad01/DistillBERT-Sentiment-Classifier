"""
push_to_hf.py
-------------
Module and CLI script to publish the fine-tuned DistilBERT sentiment classifier
(Run-2 Champion model) to the Hugging Face Hub.

Features:
- Automatic resolution of Run-2 artifacts from MLflow tracking or local directories.
- Preservation of complete 3-class label mapping (negative: 0, neutral: 1, positive: 2).
- Automatic Hugging Face Model Card (README.md) generation with benchmark metrics and usage examples.
- Upload of evaluation artifacts (confusion_matrix.png) if present.
- Support for token passing via CLI, .env (HF_TOKEN), or local cached credentials.
- Works as a standalone CLI script and as an imported library function for main.py.
"""

import os
import sys
import json
import time
import shutil
import argparse
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from dotenv import load_dotenv
from huggingface_hub import HfApi, get_token
from utils.logger import logger
from utils.custom_exception import CustomException

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
CHAMPION_RUN_ID = "57e38e355f684a0e8d15290c3e9f58ba"
_CHAMPION_ARTIFACTS = os.path.join("mlruns", "1", CHAMPION_RUN_ID, "artifacts")

DEFAULT_LOCAL_CHAMPION_PATH = os.path.join(_CHAMPION_ARTIFACTS, "model")
DEFAULT_METRICS_PATH = os.path.join(_CHAMPION_ARTIFACTS, "evaluation", "evaluation_metrics.json")
DEFAULT_CONFUSION_MATRIX_PATH = os.path.join(_CHAMPION_ARTIFACTS, "evaluation", "confusion_matrix.png")

LOCAL_FALLBACK_MODEL_PATH = os.path.join("artifacts", "model")
LOCAL_FALLBACK_METRICS_PATH = os.path.join("artifacts", "metrics", "evaluation_metrics.json")
LOCAL_FALLBACK_CM_PATH = os.path.join("artifacts", "metrics", "confusion_matrix.png")

METRICS_FILENAME = "evaluation_metrics.json"
CM_FILENAME = "confusion_matrix.png"
RUN2_ALIASES = {"run-2", "default", "runs:/run-2/model"}

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# Run-2 champion numbers, used ONLY when no evaluation_metrics.json is found.
FALLBACK_SCORES = {
    "accuracy": 0.6690,
    "macro_f1": 0.6651,
    "weighted_f1": 0.6651,
    "neg_f1": 0.7298,
    "neg_rec": 0.8241,
    "neu_f1": 0.5284,
    "neu_rec": 0.4966,
    "pos_f1": 0.7370,
    "pos_prec": 0.7960,
}


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #
def _first_existing(*candidates: Optional[str]) -> Optional[str]:
    return next((c for c in candidates if c and os.path.exists(c)), None)


def _sibling_eval_artifacts(model_path: str, use_local_fallback: bool) -> Tuple[Optional[str], Optional[str]]:
    """Look for evaluation artifacts in '<model_parent>/evaluation'."""
    eval_dir = os.path.join(os.path.dirname(model_path), "evaluation")
    metrics = _first_existing(
        os.path.join(eval_dir, METRICS_FILENAME),
        LOCAL_FALLBACK_METRICS_PATH if use_local_fallback else None,
    )
    cm = _first_existing(
        os.path.join(eval_dir, CM_FILENAME),
        LOCAL_FALLBACK_CM_PATH if use_local_fallback else None,
    )
    return metrics, cm


def _resolve_mlflow_uri(
    uri: str, tracking_uri: Optional[str]
) -> Tuple[str, Optional[str], Optional[str]]:
    """Download an MLflow model URI; also fetch the run's evaluation artifacts."""
    import mlflow
    from mlflow.tracking import MlflowClient

    logger.info(f"Resolving MLflow URI: '{uri}'...")
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri(tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

    run_id = None
    if uri.startswith("runs:/"):
        identifier, _, subpath = uri[len("runs:/"):].partition("/")
        client = MlflowClient()
        try:
            client.get_run(identifier)
            run_id = identifier
        except Exception:
            # Not a run ID -> try it as a run name (most recent match wins)
            try:
                exp_ids = [e.experiment_id for e in client.search_experiments()]
                runs = client.search_runs(
                    experiment_ids=exp_ids,
                    filter_string=f"attributes.run_name = '{identifier}'",
                    order_by=["start_time DESC"],
                    max_results=1,
                )
                if runs:
                    run_id = runs[0].info.run_id
                    logger.info(f"Resolved run name '{identifier}' to Run ID: '{run_id}'")
                    uri = f"runs:/{run_id}/{subpath}"
            except Exception as err:
                logger.warning(f"Could not resolve run name '{identifier}': {err}")

    model_path = mlflow.artifacts.download_artifacts(artifact_uri=uri)

    def _fetch(filename: str) -> Optional[str]:
        if not run_id:
            return None
        try:
            return mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=f"evaluation/{filename}"
            )
        except Exception:
            return None

    metrics = _fetch(METRICS_FILENAME)
    cm = _fetch(CM_FILENAME)
    if not (metrics and cm):
        sib_metrics, sib_cm = _sibling_eval_artifacts(model_path, use_local_fallback=False)
        metrics, cm = metrics or sib_metrics, cm or sib_cm
    return model_path, metrics, cm


def resolve_model_path(
    model_dir: Optional[str] = None,
    tracking_uri: Optional[str] = None,
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Resolves the physical directory path for the model, along with evaluation
    metrics and confusion matrix paths if available.

    Args:
        model_dir: Local path, MLflow URI ('runs:/...', 'models:/...'), or None.
        tracking_uri: MLflow tracking URI (defaults to sqlite:///mlflow.db).

    Returns:
        Tuple of (resolved_model_path, metrics_file_path, confusion_matrix_path).
    """
    try:
        model_path = metrics_path = cm_path = None

        if not model_dir or model_dir in RUN2_ALIASES:
            if os.path.exists(DEFAULT_LOCAL_CHAMPION_PATH):
                model_path = DEFAULT_LOCAL_CHAMPION_PATH
                metrics_path = _first_existing(DEFAULT_METRICS_PATH)
                cm_path = _first_existing(DEFAULT_CONFUSION_MATRIX_PATH)
            elif os.path.exists(LOCAL_FALLBACK_MODEL_PATH):
                model_path = LOCAL_FALLBACK_MODEL_PATH
                metrics_path = _first_existing(LOCAL_FALLBACK_METRICS_PATH)
                cm_path = _first_existing(LOCAL_FALLBACK_CM_PATH)

        elif model_dir.startswith(("runs:/", "models:/")):
            model_path, metrics_path, cm_path = _resolve_mlflow_uri(model_dir, tracking_uri)

        elif os.path.exists(model_dir):
            model_path = model_dir
            metrics_path, cm_path = _sibling_eval_artifacts(model_dir, use_local_fallback=True)

        if not model_path or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Could not resolve valid model path for '{model_dir}'. "
                f"Checked default Run-2 path: {DEFAULT_LOCAL_CHAMPION_PATH}"
            )

        logger.info(f"Resolved model source path: '{model_path}'")
        return model_path, metrics_path, cm_path

    except Exception as e:
        raise CustomException(e, sys)


# --------------------------------------------------------------------------- #
# Model card
# --------------------------------------------------------------------------- #
def _extract_scores(metrics: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Merge scores from evaluation_metrics.json over the Run-2 fallbacks."""
    scores = dict(FALLBACK_SCORES)
    if not metrics:
        logger.warning(
            "No evaluation_metrics.json found; model card will use bundled Run-2 fallback numbers."
        )
        return scores

    for key in ("accuracy", "macro_f1", "weighted_f1"):
        scores[key] = metrics.get(key, scores[key])

    report = metrics.get("detailed_report", {})
    for prefix, cls in (("neg", "negative"), ("neu", "neutral"), ("pos", "positive")):
        cls_report = report.get(cls, {})
        scores[f"{prefix}_f1"] = cls_report.get("f1-score", scores[f"{prefix}_f1"])
    scores["neg_rec"] = report.get("negative", {}).get("recall", scores["neg_rec"])
    scores["neu_rec"] = report.get("neutral", {}).get("recall", scores["neu_rec"])
    scores["pos_prec"] = report.get("positive", {}).get("precision", scores["pos_prec"])
    return scores


def generate_model_card(
    repo_id: str,
    metrics: Optional[Dict[str, Any]] = None,
    has_confusion_matrix: bool = False,
) -> str:
    """
    Generates a Hugging Face Model Card (README.md) with metadata YAML header,
    evaluation benchmark scores, and usage code snippets.
    """
    s = _extract_scores(metrics)
    acc, macro_f1, weighted_f1 = s["accuracy"], s["macro_f1"], s["weighted_f1"]

    cm_section = (
        f"\n### Confusion Matrix\n![Confusion Matrix]({CM_FILENAME})\n"
        if has_confusion_matrix
        else ""
    )

    return f"""---
language:
- en
license: apache-2.0
tags:
- text-classification
- sentiment-analysis
- distilbert
- pytorch
- transformers
- cardiffnlp
- twitter
datasets:
- cardiffnlp/tweet_sentiment_multilingual
metrics:
- accuracy
- f1
pipeline_tag: text-classification
model-index:
- name: {repo_id}
  results:
  - task:
      type: text-classification
      name: Sentiment Analysis
    dataset:
      name: cardiffnlp/tweet_sentiment_multilingual (English)
      type: cardiffnlp/tweet_sentiment_multilingual
      args: english
    metrics:
    - name: Test Accuracy
      type: accuracy
      value: {acc:.4f}
    - name: Test Macro F1
      type: f1
      value: {macro_f1:.4f}
    - name: Test Weighted F1
      type: f1
      value: {weighted_f1:.4f}
---

# DistilBERT Sentiment Classifier (Run-2 Champion)

Fine-tuned [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) for 3-class sentiment classification (`negative`, `neutral`, `positive`) on the English subset of the [cardiffnlp/tweet_sentiment_multilingual](https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual) dataset.

Trained with hardware optimizations designed for a **4GB VRAM NVIDIA GeForce RTX 3050 Laptop GPU**, including FP16 Tensor Core mixed precision and CUDA-fused AdamW (`adamw_torch_fused`).

---

## Benchmark Results (Test Set: 870 Unseen Samples)

| Metric | Score | Notes |
| :--- | :--- | :--- |
| **Accuracy** | **{acc * 100:.2f}%** | +1.04% over 3-epoch baseline |
| **Macro F1** | **{macro_f1 * 100:.2f}%** | Balanced across all 3 classes |
| **Weighted F1** | **{weighted_f1 * 100:.2f}%** | Metric used for checkpoint selection |
| **Negative F1 / Recall** | **{s["neg_f1"] * 100:.2f}% / {s["neg_rec"] * 100:.2f}%** | Robust detection of complaints & negative sentiment |
| **Positive F1 / Precision** | **{s["pos_f1"] * 100:.2f}% / {s["pos_prec"] * 100:.2f}%** | High precision when flagging positive sentiment |
| **Neutral F1 / Recall** | **{s["neu_f1"] * 100:.2f}% / {s["neu_rec"] * 100:.2f}%** | Weakest class; neutral is often confused with its neighbors |
| **Severe Polarity Inversion** | **3.56%** | Low confusion between opposing sentiments (31/870 samples) |

{cm_section}
---

## Training Configuration & Hyperparameters

- **Base Architecture:** `distilbert-base-uncased` (66.95M parameters)
- **Epochs:** 10 (Champion run with best model selection)
- **Learning Rate:** 2.0e-5 (Linear warmup & decay)
- **Per-Device Batch Size:** 16
- **Gradient Accumulation Steps:** 2 (Effective Batch Size = 32)
- **Precision:** Mixed Precision FP16 (AMP)
- **Optimizer:** `adamw_torch_fused` (Weight Decay = 0.01)
- **Hardware Footprint:** Peak reserved VRAM of 1,456.0 MB (~1.42 GB / 36.4% utilization)

---

## Quickstart & Usage

### Option 1: Hugging Face Pipeline (Simplest)

```python
from transformers import pipeline

# Load pipeline directly from Hugging Face Hub
classifier = pipeline("sentiment-analysis", model="{repo_id}")

# Run inference
sample_text = "I absolutely love this product! Best purchase ever!"
prediction = classifier(sample_text)
print(prediction)
# Output: [{{'label': 'positive', 'score': 0.7588}}]
```

### Option 2: PyTorch AutoModel & AutoTokenizer

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

text = "Amazing customer service and quick resolution!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]

id2label = model.config.id2label
for idx, prob in enumerate(probs):
    print(f"{{id2label[idx]:<10s}}: {{prob.item():.2%}}")
```

---

## Label Mapping

| Label ID | Sentiment Class |
| :--- | :--- |
| `0` | `negative` |
| `1` | `neutral` |
| `2` | `positive` |

---

## License

This model is open-sourced under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
"""


# --------------------------------------------------------------------------- #
# Publishing
# --------------------------------------------------------------------------- #
def _resolve_token(token: Optional[str]) -> str:
    load_dotenv()
    hf_token = (
        token
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or get_token()
    )
    if not hf_token:
        raise ValueError(
            "Hugging Face write token not found!\n"
            "Please provide a token using one of the following methods:\n"
            "  1. Pass --token <your_token> on the CLI\n"
            "  2. Add HF_TOKEN=<your_token> in your .env file\n"
            "  3. Set $env:HF_TOKEN='<your_token>' in PowerShell\n"
            "  4. Run 'huggingface-cli login' in your terminal"
        )
    return hf_token


def _load_metrics(metrics_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not metrics_path:
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        logger.info(f"Loaded evaluation metrics from '{metrics_path}'")
        return metrics
    except Exception as e:
        logger.warning(f"Could not parse metrics file: {e}")
        return None


def _upload_with_retries(api: HfApi, max_attempts: int = 5, **upload_kwargs):
    """
    Retry upload_folder on transient network errors. Already-transferred chunks
    are deduplicated by the Hub, so each retry resumes instead of starting over.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return api.upload_folder(**upload_kwargs)
        except Exception as e:
            if attempt == max_attempts:
                raise
            wait = min(10 * 2 ** (attempt - 1), 120)
            logger.warning(
                f"Upload attempt {attempt}/{max_attempts} failed ({e}). Retrying in {wait}s..."
            )
            time.sleep(wait)


def push_to_huggingface(
    repo_id: str,
    model_dir: Optional[str] = None,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: str = "Upload fine-tuned DistilBERT sentiment classifier (Run-2)",
    tracking_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Push model weights, tokenizer, model card, and evaluation plots to the
    Hugging Face Hub in a single atomic commit.

    Args:
        repo_id: Target repo on HF, e.g. 'username/distilbert-sentiment-classifier'.
        model_dir: Source path or MLflow URI.
        token: Hugging Face API write token.
        private: Visibility for a NEWLY created repo (existing repos are not changed).
        commit_message: Git commit message for upload.
        tracking_uri: MLflow tracking URI.

    Returns:
        Dict with status, repo_id, repo_url, source_path, private, commit_url.
    """
    try:
        hf_token = _resolve_token(token)

        # ---- Local work first: fail fast before touching the network ------- #
        resolved_path, metrics_path, cm_path = resolve_model_path(
            model_dir=model_dir, tracking_uri=tracking_uri
        )
        metrics = _load_metrics(metrics_path)

        # Lazy import: transformers is slow to import and not needed for --help
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        logger.info(f"Loading tokenizer and model from: '{resolved_path}'...")
        tokenizer = AutoTokenizer.from_pretrained(resolved_path)
        model = AutoModelForSequenceClassification.from_pretrained(resolved_path)
        model.config.id2label = dict(ID2LABEL)
        model.config.label2id = dict(LABEL2ID)

        # ---- Network ------------------------------------------------------- #
        api = HfApi(token=hf_token)

        if "/" not in repo_id:
            try:
                username = api.whoami().get("name")
                if username:
                    repo_id = f"{username}/{repo_id}"
                    logger.info(f"Target repository namespace auto-resolved to: '{repo_id}'")
            except Exception as e:
                logger.warning(f"Could not auto-resolve username namespace: {e}")

        logger.info(f"Connecting to Hugging Face Hub (target repo: '{repo_id}')...")
        repo_url = api.create_repo(
            repo_id=repo_id, private=private, repo_type="model", exist_ok=True
        )
        logger.info(f"Target repository ready: {repo_id} (URL: {repo_url})")

        # ---- Stage everything in a temp dir, upload once ------------------- #
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp)
            tokenizer.save_pretrained(staging)
            model.save_pretrained(staging, safe_serialization=True)

            has_cm = False
            if cm_path and os.path.exists(cm_path):
                shutil.copy2(cm_path, staging / CM_FILENAME)
                has_cm = True

            (staging / "README.md").write_text(
                generate_model_card(repo_id, metrics=metrics, has_confusion_matrix=has_cm),
                encoding="utf-8",
            )

            logger.info(f"Uploading {sum(1 for _ in staging.iterdir())} files in a single commit...")
            commit_info = _upload_with_retries(
                api,
                folder_path=str(staging),
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
            )

        logger.info(f"Model successfully published to Hugging Face Hub: https://huggingface.co/{repo_id}")
        return {
            "status": "success",
            "repo_id": repo_id,
            "repo_url": f"https://huggingface.co/{repo_id}",
            "source_path": resolved_path,
            "private": private,
            "commit_url": getattr(commit_info, "commit_url", None),
        }

    except CustomException:
        raise  # already wrapped by resolve_model_path
    except Exception as e:
        raise CustomException(e, sys)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Publish fine-tuned DistilBERT (Run-2 Champion) to Hugging Face Hub"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="distilbert-sentiment-classifier",
        help="Hugging Face repository ID (e.g., 'your-username/distilbert-sentiment-classifier' "
        "or just 'distilbert-sentiment-classifier'). Defaults to 'distilbert-sentiment-classifier'",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help=(
            "Model directory path or MLflow URI. "
            f"Defaults to Run-2 champion artifact ({DEFAULT_LOCAL_CHAMPION_PATH})"
        ),
    )
    parser.add_argument(
        "--token",
        "--hf_token",
        dest="token",
        type=str,
        default=None,
        help="Hugging Face API write token (overrides HF_TOKEN from environment/.env)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository on Hugging Face Hub",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload fine-tuned DistilBERT sentiment classifier (Run-2)",
        help="Git commit message for the Hub upload",
    )
    parser.add_argument(
        "--tracking_uri",
        type=str,
        default=None,
        help="MLflow tracking URI for resolving 'runs:/...' URIs",
    )
    return parser.parse_args()


def main():
    try:
        args = parse_cli_args()
        result = push_to_huggingface(
            repo_id=args.repo_id,
            model_dir=args.model_dir,
            token=args.token,
            private=args.private,
            commit_message=args.commit_message,
            tracking_uri=args.tracking_uri,
        )

        print("\n" + "=" * 60)
        print("HUGGING FACE HUB UPLOAD SUCCESSFUL:")
        print("=" * 60)
        print(f"Repository:   {result['repo_id']}")
        print(f"Model URL:    {result['repo_url']}")
        print(f"Source Path:  {result['source_path']}")
        print(f"Visibility:   {'Private' if result['private'] else 'Public'}")
        print("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        print(f"\n[ERROR] Upload failed: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()