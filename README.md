# DistilBERT Sentiment Classifier: Modular Python & C++ Implementation

A production-grade, modular refactoring of the DistilBERT sentiment classification notebook, specifically engineered for **NVIDIA GeForce RTX 3050 Laptop GPUs (4GB VRAM)** and **CUDA** on Windows.

---

## Key Highlights

- **4GB VRAM Hardware Optimization**:
  - **FP16 Mixed Precision**: Uses Tensor Cores to cut activation memory in half and accelerate forward/backward passes.
  - **Micro-Batching + Gradient Accumulation**: Batch size of `16` with accumulation steps of `2` provides an effective batch size of `32` while maintaining peak VRAM under $\sim 2.2\text{ GB}$.
  - **Dynamic Padding**: Batches are dynamically padded to the longest sequence in that batch rather than statically to 128 tokens, saving $\sim 60\%$ compute and memory.
  - **Fused Optimizer**: Leverages PyTorch's `adamw_torch_fused` kernel for minimal kernel launch latency and reduced memory overhead.
  - **Allocator Guard**: Automatically configures `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to prevent virtual memory fragmentation.
- **C++ Acceleration Layer (`csrc/`)**:
  - Multi-threaded dynamic tensor batch collation and padding in C++ (`fast_collator.cpp`).
  - Zero-copy confusion matrix, precision, recall, and F1 calculations in C++ (`fast_metrics.cpp`).
  - Python bindings via `pybind11` (`setup.py`) with seamless, automatic pure-Python fallback.
  - Standalone LibTorch C++ inference engine (`csrc/inference_engine.cpp`).
- **Clean Modular Design**: Separated into Configuration, Utilities, Components (Ingestion, Transformation, Training, Evaluation), and Pipelines.

---

## Project Structure

```
DistillBERT-Sentiment-Classifier/
├── config/
│   └── config.yaml                     # Central parameters (model, dataset, training, hardware)
├── csrc/                               # C++ acceleration module
│   ├── CMakeLists.txt                  # CMake build configuration
│   ├── include/
│   │   ├── fast_collator.h             # C++ dynamic batching header
│   │   └── fast_metrics.h              # C++ metrics calculation header
│   ├── src/
│   │   ├── fast_collator.cpp           # Multi-threaded dynamic padding
│   │   ├── fast_metrics.cpp            # Fast metrics computation
│   │   └── bindings.cpp                # Pybind11 module bindings
│   └── inference_engine.cpp            # Standalone LibTorch C++ runner
├── notebooks/
│   └── Distill_BERT_Finetuning.ipynb   # Original reference notebook
├── src/
│   ├── configuration_manager.py        # Typed dataclass config loader
│   ├── components/
│   │   ├── data_ingestion.py           # Hugging Face dataset download & caching
│   │   ├── data_transformation.py      # Tokenization & dynamic collator
│   │   ├── model_trainer.py            # Fine-tuning with 4GB VRAM optimizations
│   │   └── model_evaluation.py         # Evaluation, classification report & heatmap
│   └── pipeline/
│       ├── training_pipeline.py        # End-to-end training orchestrator
│       └── prediction_pipeline.py      # Real-time & batch inference service
├── utils/
│   ├── custom_exception.py             # Traceback-aware exception handling
│   ├── helpers.py                      # Device detection & VRAM monitoring
│   └── logger.py                       # File & console logging
├── app.py                              # Interactive CLI demo application
├── main.py                             # CLI entrypoint (train, predict)
├── requirements.txt                    # Project dependencies
├── setup.py                            # Package & C++ extension installer
└── distilbert_modular_cpp_plan.md      # Implementation specification
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

### 2. (Optional) Build the C++ Acceleration Module

To compile the C++ dynamic collator and metrics engine using MSVC:
```powershell
.\.venv\Scripts\python.exe setup.py build_ext --inplace
```
*(Note: If C++ compilation is skipped, the system automatically falls back to PyTorch's native dynamic collator and scikit-learn without any errors.)*

### 3. Run the Training Pipeline (with MLflow Experiment Tracking)

Train DistilBERT with 4GB VRAM optimizations and automatic MLflow logging:
```powershell
.\.venv\Scripts\python.exe main.py --mode train
```

Optional CLI overrides for MLflow runs:
```powershell
.\.venv\Scripts\python.exe main.py --mode train --experiment_name "DistilBERT-Experiment" --run_name "fp16-fused-adamw"
```

This will automatically:
1. Log pipeline parameters, dataset split sizes, and hardware specs into MLflow.
2. Stream step-by-step training and validation loss curves via Hugging Face Trainer `report_to="mlflow"`.
3. Evaluate on the test set and log `test_accuracy`, `test_macro_f1`, `test_weighted_f1`, and per-class metrics.
4. Log evaluation artifacts to MLflow (`confusion_matrix.png` and `evaluation_metrics.json`).
5. Save and register model weights, tokenizer, and TorchScript model artifacts (`model.pt`) to MLflow.

### 4. Launch the MLflow UI

View interactive run comparisons, metrics charts, confusion matrix plots, and model artifacts:
```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db
```
Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

### 5. Run Predictions

**Via CLI (using local model artifacts):**
```powershell
.\.venv\Scripts\python.exe main.py --mode predict --text "I absolutely love this product! Best purchase ever!"
```

**Via CLI (using MLflow Run Artifacts directly):**
```powershell
.\.venv\Scripts\python.exe main.py --mode predict --model_dir runs:/<RUN_ID>/model --text "Amazing customer service!"
```

**Via Interactive Console:**
```powershell
.\.venv\Scripts\python.exe app.py
```

---

## 4GB VRAM Memory Budget

| Component | Setting | Peak VRAM |
| :--- | :--- | :--- |
| Model Weights | FP16 DistilBERT | ~134 MB |
| Optimizer States | Fused AdamW (FP16 master weights) | ~536 MB |
| Micro-Batch Activations | Batch size 16, dynamic padding (~35 tokens) | ~600 - 800 MB |
| PyTorch & CUDA Overhead | Expandable segments active | ~400 MB |
| **Total Peak VRAM** | **Safely within RTX 3050 4GB limit** | **~1.8 - 2.2 GB** |
