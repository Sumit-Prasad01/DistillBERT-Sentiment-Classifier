import os
import sys
from setuptools import setup, find_packages

# Try importing pybind11 for C++ extension build
try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    has_pybind11 = True
except ImportError:
    has_pybind11 = False

ext_modules = []
cmdclass = {}

if has_pybind11:
    extra_compile_args = []
    if sys.platform == "win32":
        extra_compile_args = ["/std:c++17", "/O2"]
    else:
        extra_compile_args = ["-std=c++17", "-O3"]

    ext_modules = [
        Pybind11Extension(
            "sentiment_cpp_accel",
            [
                "csrc/src/fast_collator.cpp",
                "csrc/src/fast_metrics.cpp",
                "csrc/src/bindings.cpp",
            ],
            include_dirs=["csrc/include"],
            extra_compile_args=extra_compile_args,
            language="c++",
        ),
    ]
    cmdclass = {"build_ext": build_ext}

setup(
    name="distilbert_sentiment_classifier",
    version="1.0.0",
    author="LLM Engineering Team",
    description="Modular DistilBERT Sentiment Classifier with Python & C++ acceleration for RTX 3050 4GB VRAM",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.10",
)
