#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/fast_collator.h"
#include "../include/fast_metrics.h"

namespace py = pybind11;

PYBIND11_MODULE(sentiment_cpp_accel, m) {
    m.doc() = "C++ acceleration module for DistilBERT dynamic collation and fast metrics";

    py::class_<BatchResult>(m, "BatchResult")
        .def_readonly("input_ids", &BatchResult::input_ids)
        .def_readonly("attention_mask", &BatchResult::attention_mask)
        .def_readonly("labels", &BatchResult::labels)
        .def_readonly("batch_size", &BatchResult::batch_size)
        .def_readonly("max_seq_len", &BatchResult::max_seq_len);

    py::class_<FastMetricsResult>(m, "FastMetricsResult")
        .def_readonly("accuracy", &FastMetricsResult::accuracy)
        .def_readonly("macro_f1", &FastMetricsResult::macro_f1)
        .def_readonly("weighted_f1", &FastMetricsResult::weighted_f1)
        .def_readonly("confusion_matrix", &FastMetricsResult::confusion_matrix)
        .def_readonly("class_precisions", &FastMetricsResult::class_precisions)
        .def_readonly("class_recalls", &FastMetricsResult::class_recalls)
        .def_readonly("class_f1s", &FastMetricsResult::class_f1s)
        .def_readonly("class_supports", &FastMetricsResult::class_supports);

    m.def("collate_batch_cpp", &collate_batch_cpp,
          "Collate and dynamically pad a batch of token sequences in C++",
          py::arg("token_ids_list"),
          py::arg("labels_list"),
          py::arg("pad_token_id") = 0,
          py::arg("max_allowed_len") = 128);

    m.def("compute_metrics_cpp", &compute_metrics_cpp,
          "Compute confusion matrix and classification metrics in C++",
          py::arg("predictions"),
          py::arg("references"),
          py::arg("num_classes") = 3);
}
