#pragma once

#include <vector>
#include <cstdint>
#include <string>

struct BatchResult {
    std::vector<int64_t> input_ids;
    std::vector<int64_t> attention_mask;
    std::vector<int64_t> labels;
    int64_t batch_size;
    int64_t max_seq_len;
};

/**
 * Fast multi-threaded dynamic batch collation and padding in C++.
 *
 * @param token_ids_list List of variable-length token arrays for each sample.
 * @param labels_list Target class labels for each sample.
 * @param pad_token_id DistilBERT [PAD] token ID (0).
 * @param max_allowed_len Maximum sequence length clamp (128).
 * @return Contiguous flattened BatchResult tensors ready for PyTorch.
 */
BatchResult collate_batch_cpp(
    const std::vector<std::vector<int64_t>>& token_ids_list,
    const std::vector<int64_t>& labels_list,
    int64_t pad_token_id = 0,
    int64_t max_allowed_len = 128
);
