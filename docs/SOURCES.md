# Nguồn chính thức đã đối chiếu

Ngày kiểm tra: 25-11-2026. Tất cả liên kết dưới đây là nguồn primary (nhà cung cấp/chủ dự án). Kiểm tra tài liệu không thay thế kiểm thử package trên Power11. Một số `latest` trả nội dung khác phiên bản cũ; báo cáo cần lưu version/commit thực tế.

| Chủ đề | Nguồn | Phạm vi đã xác nhận |
|---|---|---|
| CPU MMA | [IBM Power S1112 technical overview](https://www.redbooks.ibm.com/docs/MD260032/MD260032-04.html) | MMA trên chip Power10/11; khác adapter PCIe |
| Spyre Power | [IBM introduction](https://www.ibm.com/docs/en/ibm-spyre-for-power?topic=power-introduction-spyre-accelerator) | Drawer ENZ0, ppc64le, stack RHAIIS/RHOAI, RHEL 9.8 được nêu |
| Spyre stack | [Red Hat supported configurations](https://access.redhat.com/articles/rhaii-supported-configs) | vLLM Spyre ppc64le; không phải xác nhận RF-DETR |
| Python Power | [IBM Power-optimized wheels](https://www.ibm.com/docs/en/aiservices/2026.03.0?topic=matter-power-optimized-python-wheels) | Power10/11, ppc64le, Python 3.11/3.12 cho release này |
| Wheel availability | [IBM torch index](https://wheels.developerfirst.ibm.com/ppc64le/linux-v2026.09.0/torch) | Có danh sách torch; không xác minh đủ bộ torch/torchvision/RF-DETR/ORT cụ thể |
| ORT build | [Official build instructions](https://onnxruntime.ai/docs/build/inferencing.html) | Linux PPC64LE được liệt kê; build CPU/wheel/shared library |
| ORT POWER kernels | [MLAS main](https://github.com/microsoft/onnxruntime/blob/main/cmake/onnxruntime_mlas.cmake), [v1.23.2](https://github.com/microsoft/onnxruntime/blob/v1.23.2/cmake/onnxruntime_mlas.cmake) | Power10 compiler probes, MMA builtin, HWCAP runtime test, FP32/int8 files |
| ORT build pin candidate | [v1.25.0 build.py](https://github.com/microsoft/onnxruntime/blob/v1.25.0/tools/ci_build/build.py), [v1.25.0 package](https://pypi.org/project/onnxruntime/1.25.0/) | Source/package tồn tại; chưa chạy build trên server |
| MLAS tests | [MLAS README](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/mlas/README.md) | Có SGEMM kernel tests |
| RF-DETR dependencies | [Install](https://rfdetr.roboflow.com/latest/getting-started/install/) | Extra train/onnx; không cam kết dependency wheel ppc64le |
| RF-DETR train | [Train](https://rfdetr.roboflow.com/latest/learn/train/), [Parameters](https://rfdetr.roboflow.com/latest/learn/train/training-parameters/) | High-level fine-tune, COCO/YOLO; CPU batch số nguyên |
| RF-DETR export | [Basics](https://rfdetr.roboflow.com/latest/exports/basics/) | format API, output naming và metadata có thể khác bản cũ |
| RF-DETR tensor contract | [ONNX inference](https://rfdetr.roboflow.com/latest/exports/onnx/) | dets/labels, sigmoid, background mapping, tensor name bắt buộc |
| RTX 3060 | [NVIDIA CUDA GPU table](https://www.nvidia.com/content/cuda/cuda-gpus.html) | Compute capability 8.6 |
| TensorRT compatibility | [Support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html) | Kiểm tra theo version/platform/driver; engine portability có giới hạn |
| TensorRT 11 breaking changes | [Migration 11.5](https://docs.nvidia.com/deeplearning/tensorrt/11.5.0/api/migration/tensorrt-10x-to-11x-trtexec.html) | Không dùng --fp16; transfer/CUDA Graph/spin wait đổi mặc định |
| trtexec flags | [Benchmarking](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html), [official sample README](https://github.com/NVIDIA/TensorRT/blob/main/samples/trtexec/README.md) | Build/load engine, TF32, warm-up/duration, transfer, exportTimes/profile |

Không đưa link cache-miss hoặc trang release trả lỗi vào bằng chứng đã xác nhận. Phiên kiểm tra chưa chứng minh wheel ORT chính thức trên PyPI có ppc64le; đường source build là phương án có tài liệu. Không coi việc `pip install` resolve được là kiểm định numerical correctness hoặc MMA hot path.

TF32 control của training: [PyTorch CUDA environment](https://docs.pytorch.org/docs/main/cuda_environment_variables.html), [NVIDIA control](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/). Callback instrumentation: [RF-DETR customization](https://rfdetr.roboflow.com/latest/learn/train/customization/). Các script mới vẫn cần runtime test trên stack training đã pin.

Native TensorRT adapter (API xác minh, GPU chưa thử): [11.5 Python runtime](https://docs.nvidia.com/deeplearning/tensorrt/11.5.0/inference-library/python-api-docs.html), [10.x runtime](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/python-api-docs.html).
