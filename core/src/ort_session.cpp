#include "rfdetr/ort_session.hpp"
#include "rfdetr/types.hpp"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <string>
#include <vector>
#include <dlfcn.h>

namespace rfdetr {

namespace {
std::string channel_mismatch(std::size_t classes, std::int64_t channels) {
    return "num_classes=" + std::to_string(classes) + " expects " + std::to_string(4 + classes) +
           " channels but model output has " + std::to_string(channels) + " (" +
           std::to_string(channels >= 4 ? channels - 4 : 0) + " classes)";
}
}  // namespace

struct OrtSession::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "rfdetr"};
    Ort::SessionOptions options;
    Ort::Session session{nullptr};
    Ort::MemoryInfo memory{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    OrtConfig config;
    std::array<std::int64_t, 4> shape;
    std::vector<Ort::Value> outputs;
    bool profiling;
    const bool yolo;

    explicit Impl(const OrtConfig& cfg) : config(cfg), shape{1, 3, cfg.height, cfg.width},
        profiling(!cfg.profile_prefix.empty()), yolo(cfg.model_type == "yolo11") {
        if (cfg.model_type != "rfdetr" && !yolo) throw std::invalid_argument("model_type must be rfdetr or yolo11");
        if (cfg.height < 1 || cfg.width < 1 || cfg.threads < 1 || cfg.input_name.empty() ||
            (yolo ? (cfg.output_name.empty() || cfg.classes == 0)
                  : (cfg.boxes_name.empty() || cfg.logits_name.empty() || cfg.boxes_name == cfg.logits_name)))
            throw std::invalid_argument("Invalid ORT model contract");
        options.SetIntraOpNumThreads(cfg.threads);
        options.SetInterOpNumThreads(1);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.AddConfigEntry("session.intra_op.allow_spinning", cfg.allow_spinning ? "1" : "0");
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
        const std::vector<std::string> wanted = yolo ? std::vector<std::string>{cfg.output_name}
                                                      : std::vector<std::string>{cfg.boxes_name, cfg.logits_name};
        for (const auto& name : wanted) {
            bool found = false;
            for (std::size_t i = 0; i < session.GetOutputCount(); ++i)
                if (std::string(session.GetOutputNameAllocated(i, allocator).get()) == name) {
                    const auto output_type = session.GetOutputTypeInfo(i);  // keep alive: output_info refers into it
                    const auto output_info = output_type.GetTensorTypeAndShapeInfo();
                    if (output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                        throw std::invalid_argument("Core requires FP32 output tensors");
                    if (yolo) {
                        const auto output_dims = output_info.GetShape();
                        if (output_dims.size() != 3) throw std::invalid_argument("Expected YOLO output [1, 4+C, N]");
                        if (output_dims[1] >= 0 && output_dims[1] != static_cast<std::int64_t>(4 + cfg.classes))
                            throw std::invalid_argument(channel_mismatch(cfg.classes, output_dims[1]));
                    }
                    found = true;
                }
            if (!found) throw std::invalid_argument("Configured model output name absent from model");
        }
    }

    void infer(const float* nchw, std::size_t elements, const char* const* output_names, std::size_t count) {
        if (!nchw || elements != checked_product(checked_product(config.height, config.width), 3))
            throw std::invalid_argument("ORT input buffer size mismatch");
        // ORT's wrapper takes a mutable pointer; Session::Run consumes this tensor as input.
        auto input = Ort::Value::CreateTensor<float>(memory, const_cast<float*>(nchw), elements,
                                                    shape.data(), shape.size());
        const char* input_names[]{config.input_name.c_str()};
        outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, count);
    }
};

OrtSession::OrtSession(const OrtConfig& config) : impl_(std::make_unique<Impl>(config)) {}
OrtSession::~OrtSession() = default;

OutputView OrtSession::run(const float* nchw, std::size_t elements) {
    auto& state = *impl_;
    if (state.yolo) throw std::logic_error("run() serves model_type rfdetr; use run_yolo()");
    const char* output_names[]{state.config.boxes_name.c_str(), state.config.logits_name.c_str()};
    state.infer(nchw, elements, output_names, 2);
    const auto boxes = state.outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const auto logits = state.outputs[1].GetTensorTypeAndShapeInfo().GetShape();
    if (boxes.size() != 3 || logits.size() != 3 || boxes[0] != 1 || boxes[1] < 1 || boxes[2] != 4 ||
        logits[0] != 1 || logits[1] != boxes[1] || logits[2] < 1)
        throw std::invalid_argument("Expected boxes [1,Q,4] and logits [1,Q,C]");
    return {state.outputs[0].GetTensorData<float>(), state.outputs[1].GetTensorData<float>(),
        static_cast<std::size_t>(boxes[1]), static_cast<std::size_t>(logits[2])};
}

YoloOutputView OrtSession::run_yolo(const float* nchw, std::size_t elements) {
    auto& state = *impl_;
    if (!state.yolo) throw std::logic_error("run_yolo() serves model_type yolo11; use run()");
    const char* output_names[]{state.config.output_name.c_str()};
    state.infer(nchw, elements, output_names, 1);
    const auto shape = state.outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 3 || shape[0] != 1 || shape[2] < 1) throw std::invalid_argument("Expected YOLO output [1, 4+C, N]");
    if (shape[1] != static_cast<std::int64_t>(4 + state.config.classes))
        throw std::invalid_argument(channel_mismatch(state.config.classes, shape[1]));
    return {state.outputs[0].GetTensorData<float>(), static_cast<std::size_t>(shape[1]),
        static_cast<std::size_t>(shape[2])};
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
