# DistilBERT Sentiment Classifier: Modular Python & C++ Implementation

A production-grade, modular refactoring of the DistilBERT sentiment classification pipeline, specifically engineered for **NVIDIA GeForce RTX 3050 Laptop GPUs (4GB VRAM)** and **CUDA** on Windows, featuring native C++ acceleration and MLflow experiment governance.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5%2B%20CUDA-ee4c2c.svg)](https://pytorch.org/)
[![Transformers 4.x](https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg)](https://huggingface.co/)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ZyroGod%2Fdistilbert--sentiment--classifier-ffcc00.svg)](https://huggingface.co/ZyroGod/distilbert-sentiment-classifier)
[![MLflow Tracking](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2.svg)](https://mlflow.org/)
[![C++17 Accelerated](https://img.shields.io/badge/C%2B%2B17-pybind11%20Extension-00599C.svg)](https://github.com/pybind/pybind11)

---

## Documentation Index

- 🤗 [**Hugging Face Hub Model (ZyroGod/distilbert-sentiment-classifier)**](https://huggingface.co/ZyroGod/distilbert-sentiment-classifier): Official repository with fine-tuned weights, tokenizer, and interactive inference widget.
- 📋 [**System Architecture Documentation (System_Architecture.md)**](System_Architecture.md): Complete architecture diagrams (flowcharts, sequence flows, ER model), subsystem breakdowns, C++ pybind11 translation layer, and deployment topologies.
- 📊 [**Model Evaluation Report (Eval_Report.md)**](Eval_Report.md): Empirical benchmark report across training runs, confusion matrix analysis, class-level precision/recall/F1 metrics, and VRAM memory profiling.

---

## Key Highlights

- **4GB VRAM Hardware Optimization (RTX 3050)**:
  - **FP16 Mixed Precision (AMP)**: Uses NVIDIA Tensor Cores to cut activation memory in half and accelerate matrix multiplications.
  - **Micro-Batching + Gradient Accumulation**: Micro-batch size of `16` with accumulation steps of `2` yields an effective batch size of `32` while maintaining peak VRAM at only **1,456.0 MB (~1.42 GB, 36.4% of 4GB capacity)**.
  - **Dynamic Padding**: Batches are dynamically padded to the longest sequence in that batch rather than statically to 128 tokens, eliminating redundant computations on pad tokens.
  - **Fused Optimizer**: Leverages PyTorch's `adamw_torch_fused` CUDA kernel for lower launch overhead and reduced state memory.
  - **Allocator Guard**: Automatically configures `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to prevent virtual memory fragmentation.
- **Native C++ Acceleration Layer (`csrc/` & `sentiment_cpp_accel`)**:
  - **Dynamic Collator (`fast_collator.cpp`)**: Pre-allocates contiguous memory buffers for batch tokens and attention masks.
  - **Fast Metrics Engine (`fast_metrics.cpp`)**: Single-pass $O(N)$ computation of confusion matrix, class precision, recall, and Macro/Weighted F1.
  - **Seamless Fallback**: Pure Python fallback to PyTorch `DataCollatorWithPadding` and Scikit-Learn if the C++ module is not compiled.
  - **LibTorch C++ Inference Engine (`inference_engine.cpp`)**: Zero-Python standalone inference binary using exported TorchScript (`model.pt`).
- **End-to-End MLflow Experiment Tracking**:
  - Embedded SQLite tracking database (`sqlite:///mlflow.db`) and artifact store.
  - Automatic hyperparameter, step-loss, validation checkpoint, and test metric logging.
  - Dual model flavor export: standard Hugging Face directory format and MLflow `transformers` PyFunc flavor.
  - **Smart URI Resolution**: Seamlessly downloads and evaluates models using either friendly Run Names (`runs:/run-1/model`) or 32-character Run UUIDs (`runs:/<run_id>/model`).

---

## Benchmark & Evaluation Summary

Trained on the English subset of `cardiffnlp/tweet_sentiment_multilingual` (1,839 train, 324 validation, 870 test):

| Metric | Baseline Run (`run-1`, 3 Epochs) | Optimized Champion (`run-2`, 10 Epochs) | Status |
|---|---|---|---|
| **Test Accuracy** | 65.86% | **66.90%** | **+1.04%** |
| **Test Macro F1** | 64.79% | **66.51%** | **+1.72%** |
| **Test Weighted F1** | 64.79% | **66.51%** | **+1.72%** |
| **Final Train Loss** | 1.6539 | **0.7255** | **-56.1% (Strong convergence)** |
| **Negative F1 / Recall** | 72.75% / 87.93% | **72.98% / 82.41%** | Robust negative sentiment detection |
| **Positive F1 / Precision** | 73.00% / 79.35% | **73.70% / 79.60%** | High positive precision |
| **Peak GPU VRAM** | 1,456.0 MB (36.4%) | 1,456.0 MB (36.4%) | Zero OOM occurrences |
| **Training Speed** | 182.25 samples/sec | **202.97 samples/sec** | Sub-91s total training time |
| **Eval Throughput** | — | **1,237.31 samples/sec** | Sub-second test set eval |

*For complete confusion matrix charts and error analysis, see [Eval_Report.md](Eval_Report.md).*

---

## Project Structure

```
DistillBERT-Sentiment-Classifier/
├── config/
│   └── config.yaml                     # Central declarative YAML configuration
├── csrc/                               # Native C++ acceleration subsystem
│   ├── CMakeLists.txt                  # Standalone LibTorch C++ build configuration
│   ├── inference_engine.cpp            # Zero-Python LibTorch standalone inference runner
│   ├── include/
│   │   ├── fast_collator.h             # Dynamic batching header
│   │   └── fast_metrics.h              # C++ metrics calculation header
│   └── src/
│       ├── bindings.cpp                # Pybind11 Python extension bindings
│       ├── fast_collator.cpp           # Contiguous memory buffer allocation for batch collation
│       └── fast_metrics.cpp            # High-speed confusion matrix & multi-class F1
├── src/                                # Modular ML pipelines & components
│   ├── configuration_manager.py        # Strongly-typed configuration manager with frozen dataclasses
│   ├── components/
│   │   ├── data_ingestion.py           # HF Hub download, split validation, local parquet caching
│   │   ├── data_transformation.py      # Tokenization, dynamic padding, and C++ collator fallback
│   │   ├── model_trainer.py            # 4GB VRAM training loop, FP16, fused optimizer, TorchScript export
│   │   └── model_evaluation.py         # Test split evaluation, C++ metrics, plots, and JSON logging
│   └── pipeline/
│       ├── training_pipeline.py        # End-to-end training orchestration under MLflow run context
│       └── prediction_pipeline.py      # Real-time inference, batch prediction, and MLflow URI resolver
├── utils/
│   ├── custom_exception.py             # Custom sys-traced exception handling
│   ├── helpers.py                      # CUDA alloc setup, VRAM monitors, seeds, MLflow env
│   └── logger.py                       # Timestamped file & console logging
├── artifacts/                          # Versioned pipeline artifacts
│   ├── data/raw/                       # Downloaded train, validation, and test Parquet files
│   ├── data/transformed/               # Tokenized Hugging Face DatasetDict
│   ├── checkpoints/                    # Intermediate training checkpoints
│   ├── model/                          # Final fine-tuned model, tokenizer, and TorchScript export
│   └── metrics/                        # Confusion matrix heatmap and metrics JSON
├── logs/                               # Execution and inference log files
├── notebooks/
│   └── Distill_BERT_Finetuning.ipynb   # Exploratory data analysis and prototyping notebook
├── app.py                              # Interactive CLI console with confidence bar charts
├── main.py                             # Unified CLI entry point for training and prediction
├── Eval_Report.md                      # Comprehensive evaluation and benchmark report
├── System_Architecture.md              # Detailed system architecture with Mermaid diagrams
├── mlflow.db                           # SQLite database for MLflow experiment tracking
├── setup.py                            # C++ extension compiler with pybind11 and setuptools
└── requirements.txt                    # Project dependencies
```

---

## Quickstart Guide

### 1. Environment Setup (using `.venv`)

Activate your virtual environment and install the required dependencies:
```powershell
# Activate .venv (PowerShell)
.\.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA support and dependencies
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. (Optional) Compile Native C++ Acceleration Extension

Compile the C++ dynamic collator and metrics engine using MSVC or GCC:
```powershell
.\.venv\Scripts\python.exe setup.py build_ext --inplace
```
*(Note: If compilation is skipped, the system automatically falls back to PyTorch's native dynamic collator and Scikit-Learn without errors.)*

### 3. Run the Training Pipeline

Train DistilBERT with 4GB VRAM GPU optimizations and MLflow experiment tracking:
```powershell
.\.venv\Scripts\python.exe main.py --mode train --experiment_name "DistilBERT-Experiments" --run_name "run-1"
```

This will automatically:
1. Detect GPU hardware and activate 4GB VRAM safety settings.
2. Ingest, validate, and cache the CardiffNLP tweet sentiment dataset.
3. Tokenize sequences and apply dynamic padding collation.
4. Execute FP16 mixed-precision training with CUDA-fused AdamW.
5. Compute test metrics and confusion matrix via the C++ metrics engine.
6. Log all parameters, metrics, plots, and models to MLflow.

### 4. Launch the MLflow UI

View interactive run comparisons, metrics charts, confusion matrix plots, and model artifacts:
```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db
```
Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

### 5. Run Sentiment Predictions

#### Option A: Via CLI with Automatic MLflow Run-Name Resolution
Load models directly from MLflow using either the friendly **Run Name** or **Run ID**:
```powershell
# By Run Name
.\.venv\Scripts\python.exe main.py --mode predict --model_dir runs:/run-1/model --text "Amazing customer service!"

# By Run ID (UUID)
.\.venv\Scripts\python.exe main.py --mode predict --model_dir runs:/793752383148441bbe4980b1580a2dbd/model --text "Amazing customer service!"
```

#### Option B: Via CLI using Local Artifacts
```powershell
.\.venv\Scripts\python.exe main.py --mode predict --model_dir artifacts/model --text "This product exceeded all my expectations!"
```

#### Option C: Interactive Console Application
Launch the interactive terminal UI with Unicode confidence bars and sample benchmarks:
```powershell
.\.venv\Scripts\python.exe app.py
```

**Sample Output:**
```
============================================================
PREDICTION RESULT:
============================================================
Text:       "Amazing customer service!"
Sentiment:  POSITIVE (75.88%)
Probabilities:
  negative  :  8.30%
  neutral   : 15.82%
  positive  : 75.88%
============================================================
```

#### Option D: Direct Inference from Hugging Face Hub (Zero Setup)
Load and run inference with the champion model directly from Hugging Face in any Python environment:

```python
from transformers import pipeline

# Load pipeline directly from Hugging Face Hub
classifier = pipeline("sentiment-analysis", model="ZyroGod/distilbert-sentiment-classifier")

# Run inference
sample_text = "I absolutely love this product! Best purchase ever!"
prediction = classifier(sample_text)
print(prediction)
# Output: [{'label': 'positive', 'score': 0.7588}]
```

---

### 6. Published Hugging Face Model (Run-2 Champion)

The fine-tuned champion model is published and accessible live on the Hugging Face Hub:
👉 **[ZyroGod/distilbert-sentiment-classifier](https://huggingface.co/ZyroGod/distilbert-sentiment-classifier)**

It features complete model weights (`model.safetensors`), fast tokenizer configurations, exact class mappings (`negative`, `neutral`, `positive`), confusion matrix plots, and automated benchmark cards.

#### To Re-publish or Update via Unified CLI (`main.py`):
```powershell
# Standard push (uses HF_TOKEN from .env and auto-resolves username namespace)
.\.venv\Scripts\python.exe main.py --mode push_to_hf --repo_id "ZyroGod/distilbert-sentiment-classifier"

# Quick push with default repo name 'distilbert-sentiment-classifier'
.\.venv\Scripts\python.exe main.py --mode push_to_hf

# Specify custom token or private repo flag if needed
.\.venv\Scripts\python.exe main.py --mode push_to_hf --repo_id "ZyroGod/distilbert-sentiment-classifier" --hf_token "hf_xxx" --private
```

#### To Re-publish via Dedicated Script (`push_to_hf.py`):
```powershell
.\.venv\Scripts\python.exe push_to_hf.py --repo_id "ZyroGod/distilbert-sentiment-classifier"
```

---

## 4GB VRAM Memory Budget & Empirical Usage

| Component | Architecture Setting | Theoretical Budget | Measured Empirical Peak |
| :--- | :--- | :--- | :--- |
| **Model Weights** | FP16 DistilBERT (`distilbert-base-uncased`, 67M params) | ~134 MB | ~134 MB |
| **Optimizer States** | Fused AdamW (`adamw_torch_fused`, FP32 master copy) | ~536 MB | ~536 MB |
| **Micro-Batch Activations** | Micro-batch size 16, dynamic padding (avg ~35 tokens) | ~600 - 800 MB | ~112 MB |
| **PyTorch & CUDA Overhead** | Expandable segments active (`PYTORCH_CUDA_ALLOC_CONF`) | ~400 MB | ~674 MB |
| **Total VRAM Footprint** | **Safely within RTX 3050 4.0 GB limit** | **~1.8 - 2.2 GB** | **1,456.0 MB (~1.42 GB / 36.4%)** |

---

## License

This project is licensed under the MIT License.
