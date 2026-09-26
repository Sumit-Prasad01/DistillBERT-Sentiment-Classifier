#include "../include/fast_collator.h"
#include <algorithm>
#include <stdexcept>

BatchResult collate_batch_cpp(
    const std::vector<std::vector<int64_t>>& token_ids_list,
    const std::vector<int64_t>& labels_list,
    int64_t pad_token_id,
    int64_t max_allowed_len
) {
    int64_t batch_size = static_cast<int64_t>(token_ids_list.size());
    if (batch_size == 0) {
        return BatchResult{{}, {}, {}, 0, 0};
    }

    // 1. Find max sequence length in this batch (dynamic padding)
    int64_t max_len = 0;
    for (const auto& tokens : token_ids_list) {
        int64_t len = static_cast<int64_t>(tokens.size());
        if (len > max_len) {
            max_len = len;
        }
    }
    max_len = std::min(max_len, max_allowed_len);
    if (max_len == 0) {
        max_len = 1;
    }

    // 2. Pre-allocate contiguous memory buffers for input_ids and attention_mask
    int64_t total_elements = batch_size * max_len;
    BatchResult result;
    result.batch_size = batch_size;
    result.max_seq_len = max_len;
    result.input_ids.resize(total_elements, pad_token_id);
    result.attention_mask.resize(total_elements, 0);
    result.labels = labels_list;

    // 3. Fast parallel/sequential buffer population
    for (int64_t i = 0; i < batch_size; ++i) {
        const auto& tokens = token_ids_list[i];
        int64_t seq_len = std::min(static_cast<int64_t>(tokens.size()), max_len);
        int64_t row_offset = i * max_len;

        for (int64_t j = 0; j < seq_len; ++j) {
            result.input_ids[row_offset + j] = tokens[j];
            result.attention_mask[row_offset + j] = 1;
        }
    }

    return result;
}
