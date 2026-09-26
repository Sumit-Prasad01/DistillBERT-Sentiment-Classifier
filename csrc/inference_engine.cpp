#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>

// Note: To compile standalone with LibTorch:
// cmake -B build -DCMAKE_PREFIX_PATH=/path/to/libtorch
// cmake --build build --config Release

#ifdef USE_TORCH
#include <torch/script.h>
#include <torch/torch.h>

int run_torch_inference(const std::string& model_path) {
    try {
        std::cout << "Loading TorchScript model from: " << model_path << "..." << std::endl;
        torch::jit::script::Module module = torch::jit::load(model_path);
        module.eval();

        // Check CUDA availability
        torch::Device device = torch::kCPU;
        if (torch::cuda::is_available()) {
            device = torch::kCUDA;
            module.to(device);
            std::cout << "CUDA is available! Model moved to GPU." << std::endl;
        } else {
            std::cout << "Running on CPU." << std::endl;
        }

        // Create sample inputs (batch_size=1, seq_len=8)
        std::vector<int64_t> sample_tokens = {101, 2023, 2003, 1037, 2307, 2056, 999, 102};
        std::vector<int64_t> sample_mask = {1, 1, 1, 1, 1, 1, 1, 1};

        torch::Tensor input_ids = torch::tensor(sample_tokens).unsqueeze(0).to(device);
        torch::Tensor attention_mask = torch::tensor(sample_mask).unsqueeze(0).to(device);

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_ids);
        inputs.push_back(attention_mask);

        std::cout << "Executing forward pass..." << std::endl;
        auto output = module.forward(inputs).toGenericDict();
        auto logits = output.at("logits").toTensor();

        auto probs = torch::softmax(logits, -1);
        auto pred_class = torch::argmax(probs, -1).item<int64_t>();

        const std::vector<std::string> label_names = {"negative", "neutral", "positive"};
        std::cout << "Prediction: " << label_names[pred_class] << " (Class " << pred_class << ")" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Inference error: " << e.what() << std::endl;
        return 1;
    }
}
#endif

int main(int argc, char* argv[]) {
    std::cout << "=== DistilBERT C++ High-Speed Inference Engine ===" << std::endl;
    std::string model_path = (argc > 1) ? argv[1] : "artifacts/model/model.pt";

#ifdef USE_TORCH
    return run_torch_inference(model_path);
#else
    std::cout << "Compiled in standalone mode without LibTorch header links." << std::endl;
    std::cout << "Model path target: " << model_path << std::endl;
    std::cout << "To build with LibTorch, link against torch libraries using CMakeLists.txt." << std::endl;
    return 0;
#endif
}
