#include "rfdetr/ort_session.hpp"
#include "rfdetr/types.hpp"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <vector>
#include <dlfcn.h>

namespace rfdetr {

struct OrtSession::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "rfdetr"};
    Ort::SessionOptions options;
    Ort::Session session{nullptr};
    Ort::MemoryInfo memory{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    OrtConfig config;
    std::array<std::int64_t, 4> shape;
    std::vector<Ort::Value> outputs;
    bool profiling;

    explicit Impl(const OrtConfig& cfg) : config(cfg), shape{1, 3, cfg.height, cfg.width},
        profiling(!cfg.profile_prefix.empty()) {
        if (cfg.height < 1 || cfg.width < 1 || cfg.threads < 1 || cfg.input_name.empty() ||
            cfg.boxes_name.empty() || cfg.logits_name.empty() || cfg.boxes_name == cfg.logits_name)
            throw std::invalid_argument("Invalid ORT model contract");
        options.SetIntraOpNumThreads(cfg.threads);
        options.SetInterOpNumThreads(1);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.AddConfigEntry("session.intra_op.allow_spinning", "0");
        if (profiling) options.EnableProfiling(cfg.profile_prefix.c_str());
        session = Ort::Session(env, cfg.model.c_str(), options);
        if (session.GetInputCount() != 1) throw std::invalid_argument("Expected one image input");
        Ort::AllocatorWithDefaultOptions allocator;
        if (std::string(session.GetInputNameAllocated(0, allocator).get()) != cfg.input_name)
            throw std::invalid_argument("Configured input name absent from model");
        const auto type = session.GetInputTypeInfo(0);
        const auto input = type.GetTensorTypeAndShapeInfo();
        if (input.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::invalid_argument("Core requires explicit FP32 image input");
        const auto dims = input.GetShape();
        if (dims.size() != shape.size()) throw std::invalid_argument("Expected NCHW input");
        for (std::size_t i = 0; i < dims.size(); ++i)
            if (dims[i] >= 0 && dims[i] != shape[i]) throw std::invalid_argument("Input size differs from export");
        for (const auto& wanted : {cfg.boxes_name, cfg.logits_name}) {
            bool found = false;
            for (std::size_t i = 0; i < session.GetOutputCount(); ++i)
                if (std::string(session.GetOutputNameAllocated(i, allocator).get()) == wanted) {
                    const auto output_type = session.GetOutputTypeInfo(i);
                    if (output_type.GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                        throw std::invalid_argument("Core requires FP32 output tensors");
                    found = true;
                }
            if (!found) throw std::invalid_argument("Configured RF-DETR output name absent from model");
        }
    }
};

OrtSession::OrtSession(const OrtConfig& config) : impl_(std::make_unique<Impl>(config)) {}
OrtSession::~OrtSession() = default;

OutputView OrtSession::run(const float* nchw, std::size_t elements) {
    auto& state = *impl_;
    if (!nchw || elements != checked_product(checked_product(state.config.height, state.config.width), 3))
        throw std::invalid_argument("ORT input buffer size mismatch");
    // ORT's wrapper takes a mutable pointer; Session::Run consumes this tensor as input.
    auto input = Ort::Value::CreateTensor<float>(state.memory, const_cast<float*>(nchw), elements,
                                                state.shape.data(), state.shape.size());
    const char* input_names[]{state.config.input_name.c_str()};
    const char* output_names[]{state.config.boxes_name.c_str(), state.config.logits_name.c_str()};
    state.outputs = state.session.Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 2);
    const auto boxes = state.outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const auto logits = state.outputs[1].GetTensorTypeAndShapeInfo().GetShape();
    if (boxes.size() != 3 || logits.size() != 3 || boxes[0] != 1 || boxes[1] < 1 || boxes[2] != 4 ||
        logits[0] != 1 || logits[1] != boxes[1] || logits[2] < 1)
        throw std::invalid_argument("Expected boxes [1,Q,4] and logits [1,Q,C]");
    return {state.outputs[0].GetTensorData<float>(), state.outputs[1].GetTensorData<float>(),
        static_cast<std::size_t>(boxes[1]), static_cast<std::size_t>(logits[2])};
}

std::string OrtSession::end_profile() {
    if (!impl_->profiling) return {};
    Ort::AllocatorWithDefaultOptions allocator;
    auto path = impl_->session.EndProfilingAllocated(allocator);
    impl_->profiling = false;
    return path.get();
}

std::string OrtSession::version() { return OrtGetApiBase()->GetVersionString(); }

std::string OrtSession::library_path() {
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void*>(&OrtGetApiBase), &info) || !info.dli_fname)
        throw std::runtime_error("Cannot identify loaded ONNX Runtime library");
    return info.dli_fname;
}

}  // namespace rfdetr
