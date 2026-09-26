#include "../include/fast_metrics.h"
#include <stdexcept>
#include <numeric>

FastMetricsResult compute_metrics_cpp(
    const std::vector<int64_t>& predictions,
    const std::vector<int64_t>& references,
    int64_t num_classes
) {
    if (predictions.size() != references.size()) {
        throw std::invalid_argument("Predictions and references must have the same length.");
    }

    int64_t n = static_cast<int64_t>(predictions.size());
    FastMetricsResult result;
    result.confusion_matrix.resize(num_classes, std::vector<int64_t>(num_classes, 0));
    result.class_precisions.resize(num_classes, 0.0);
    result.class_recalls.resize(num_classes, 0.0);
    result.class_f1s.resize(num_classes, 0.0);
    result.class_supports.resize(num_classes, 0);

    if (n == 0) {
        result.accuracy = 0.0;
        result.macro_f1 = 0.0;
        result.weighted_f1 = 0.0;
        return result;
    }

    int64_t correct = 0;

    // 1. Build Confusion Matrix: [true_label][predicted_label]
    for (int64_t i = 0; i < n; ++i) {
        int64_t t = references[i];
        int64_t p = predictions[i];
        if (t >= 0 && t < num_classes && p >= 0 && p < num_classes) {
            result.confusion_matrix[t][p]++;
            if (t == p) {
                correct++;
            }
        }
    }

    result.accuracy = static_cast<double>(correct) / static_cast<double>(n);

    // 2. Compute Per-Class Precision, Recall, and F1
    double sum_f1 = 0.0;
    double weighted_f1_sum = 0.0;

    for (int64_t c = 0; c < num_classes; ++c) {
        int64_t tp = result.confusion_matrix[c][c];
        int64_t support = 0;
        for (int64_t p = 0; p < num_classes; ++p) {
            support += result.confusion_matrix[c][p];
        }
        result.class_supports[c] = support;

        int64_t predicted_c = 0;
        for (int64_t t = 0; t < num_classes; ++t) {
            predicted_c += result.confusion_matrix[t][c];
        }

        double precision = (predicted_c > 0) ? (static_cast<double>(tp) / static_cast<double>(predicted_c)) : 0.0;
        double recall = (support > 0) ? (static_cast<double>(tp) / static_cast<double>(support)) : 0.0;
        double f1 = (precision + recall > 0.0) ? (2.0 * precision * recall / (precision + recall)) : 0.0;

        result.class_precisions[c] = precision;
        result.class_recalls[c] = recall;
        result.class_f1s[c] = f1;

        sum_f1 += f1;
        weighted_f1_sum += f1 * static_cast<double>(support);
    }

    result.macro_f1 = (num_classes > 0) ? (sum_f1 / static_cast<double>(num_classes)) : 0.0;
    result.weighted_f1 = (n > 0) ? (weighted_f1_sum / static_cast<double>(n)) : 0.0;

    return result;
}
