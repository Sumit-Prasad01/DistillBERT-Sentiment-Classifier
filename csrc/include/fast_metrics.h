#pragma once

#include <vector>
#include <cstdint>
#include <map>
#include <string>

struct FastMetricsResult {
    double accuracy;
    double macro_f1;
    double weighted_f1;
    std::vector<std::vector<int64_t>> confusion_matrix;
    std::vector<double> class_precisions;
    std::vector<double> class_recalls;
    std::vector<double> class_f1s;
    std::vector<int64_t> class_supports;
};

/**
 * High-speed computation of confusion matrix, per-class metrics, and weighted/macro F1.
 *
 * @param predictions Predicted class IDs.
 * @param references True class IDs.
 * @param num_classes Number of target classes (3).
 * @return FastMetricsResult containing all evaluation metrics.
 */
FastMetricsResult compute_metrics_cpp(
    const std::vector<int64_t>& predictions,
    const std::vector<int64_t>& references,
    int64_t num_classes = 3
);
