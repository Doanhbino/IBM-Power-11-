// ONNX Runtime custom operator com.rfdetr::VsxLayerNorm, a drop-in for LayerNormalization over the last axis.
//
// Why: in the RF-DETR profile ORT's own LayerNormalization kernel (ComputeJob<float,float,void>) took 14-16% of all
// cycles and does not use the vector units well. This operator calls rfdetr::layer_norm_rows (VSX on Power) and splits
// rows over ORT's intra-op thread pool, so it honours the session's thread count.
//
// Build: shared library, no link to libonnxruntime (the API table comes from the loader); see CMakeLists.txt.
// Load : SessionOptions::RegisterCustomOpsLibrary(path) in C++, register_custom_ops_library(path) in Python.
// Inputs X [..., N] float, scale [N] float, bias [N] float; attribute epsilon (float, default 1e-5); output Y like X.
#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#include "rfdetr/layernorm.hpp"

#include <cstdint>
#include <vector>

namespace {

struct Job {
    const float *x, *scale, *bias;
    float* y;
    std::size_t rows, n, rows_per_chunk;
    float epsilon;
};

void run_chunk(void* user, std::size_t chunk) {
    const auto& job = *static_cast<const Job*>(user);
    const std::size_t first = chunk * job.rows_per_chunk;
    const std::size_t count = std::min(job.rows_per_chunk, job.rows - first);
    rfdetr::layer_norm_rows(job.x + first * job.n, job.scale, job.bias, job.y + first * job.n, count, job.n, job.epsilon);
}

struct Kernel {
    Kernel(const OrtApi&, const OrtKernelInfo* info) {
        epsilon = Ort::ConstKernelInfo(info).GetAttribute<float>("epsilon");
    }

    void Compute(OrtKernelContext* context) const {
        Ort::KernelContext ctx(context);
        const auto x = ctx.GetInput(0), scale = ctx.GetInput(1), bias = ctx.GetInput(2);
        const auto shape = x.GetTensorTypeAndShapeInfo().GetShape();
        if (shape.empty() || shape.back() < 1) ORT_CXX_API_THROW("VsxLayerNorm needs a non-empty last axis", ORT_INVALID_ARGUMENT);
        const std::size_t n = static_cast<std::size_t>(shape.back());
        std::size_t rows = 1;
        for (std::size_t i = 0; i + 1 < shape.size(); ++i) {
            if (shape[i] < 0) ORT_CXX_API_THROW("VsxLayerNorm got a negative dimension", ORT_INVALID_ARGUMENT);
            rows *= static_cast<std::size_t>(shape[i]);
        }
        if (scale.GetTensorTypeAndShapeInfo().GetElementCount() != n || bias.GetTensorTypeAndShapeInfo().GetElementCount() != n)
            ORT_CXX_API_THROW("VsxLayerNorm scale and bias must have one value per last-axis element", ORT_INVALID_ARGUMENT);
        auto out = ctx.GetOutput(0, shape.data(), shape.size());
        Job job{x.GetTensorData<float>(), scale.GetTensorData<float>(), bias.GetTensorData<float>(),
                out.GetTensorMutableData<float>(), rows, n, 16, epsilon};
        if (rows == 0) return;
        const std::size_t chunks = (rows + job.rows_per_chunk - 1) / job.rows_per_chunk;
        ctx.ParallelFor(&run_chunk, chunks, 0, &job);
    }

    float epsilon = 1e-5f;
};

struct Op : Ort::CustomOpBase<Op, Kernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const { return new Kernel(api, info); }
    const char* GetName() const { return "VsxLayerNorm"; }
    const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
    size_t GetInputTypeCount() const { return 3; }
    ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
};

}  // namespace

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
    Ort::InitApi(api_base->GetApi(ORT_API_VERSION));
    try {
        static Op op;
        static Ort::CustomOpDomain domain{"com.rfdetr"};   // must outlive every session that uses it
        static bool added = false;
        if (!added) { domain.Add(&op); added = true; }
        Ort::UnownedSessionOptions session_options(options);
        session_options.Add(domain);
    } catch (const Ort::Exception& error) {
        return Ort::GetApi().CreateStatus(error.GetOrtErrorCode(), error.what());
    }
    return nullptr;
}
