# Hỗ trợ phần mềm và phần cứng đã kiểm chứng

Ngày đối chiếu tài liệu web: **25-11-2026**. Đây là xác nhận từ tài liệu/mã nguồn; chưa phải chứng nhận bộ binary chạy trên LPAR của người dùng. Các trang `latest`, nhánh `main` và chỉ mục wheel có thể thay đổi; mỗi lần dựng phải lưu phiên bản, commit, wheel và SHA256 vào manifest.

## Phân biệt đúng đường thực thi

| Đường | Phần cứng / runtime | Kết luận cho baseline |
|---|---|---|
| CPU Power11 | MMA nằm trong CPU; cùng đường CPUExecutionProvider | Đường FP32 bắt đầu. Không cần có Spyre mới dùng được MMA. |
| Spyre | Adapter PCIe rời, drawer ENZ0, driver/runtime riêng | Chỉ mở nhánh thực nghiệm sau khi xác nhận card, quyền gán PCIe cho LPAR và model support. |
| NVIDIA RTX 3060 | GPU Ampere CC 8.6, CUDA/TensorRT | Máy đối chứng riêng; cần ghi đầy đủ CPU host, RAM, OS, driver, CUDA, TensorRT, VRAM thực tế. |

IBM mô tả MMA là tăng tốc trên chip, còn Spyre dùng phần cứng PCIe và enablement stack riêng. [IBM Power S1112 technical overview](https://www.redbooks.ibm.com/docs/MD260032/MD260032-04.html), [IBM Spyre introduction](https://www.ibm.com/docs/en/ibm-spyre-for-power?topic=power-introduction-spyre-accelerator), [NVIDIA CUDA GPU table](https://www.nvidia.com/content/cuda/cuda-gpus.html).

**OpenBLAS và MMA không phải hai lựa chọn cùng tầng.** OpenBLAS/MLAS là thư viện kernel; MMA là tập lệnh/tài nguyên CPU mà kernel có thể dùng. ORT CPU baseline dùng MLAS. Cài thêm OpenBLAS không tự thay kernel ORT và không chứng minh RF-DETR đã dùng MMA. PyTorch training có thể đi qua thư viện BLAS khác với ORT inference.

## Ma trận quyết định

| Thành phần | Chứng cứ chính thức | Trạng thái thực tế cần kiểm tra |
|---|---|---|
| RHEL 9.8 + Spyre Power | IBM liệt kê RHEL 9.6, 9.8, 10.2 và ppc64le | Không suy ra LPAR đang có card hoặc RF-DETR được hỗ trợ. |
| ONNX Runtime CPU trên Linux ppc64le | Bảng kiến trúc build upstream ghi PPC64LE = YES | Phải build/test hoặc dùng wheel đúng ABI; bảng kiến trúc không đảm bảo wheel PyPI có sẵn. |
| MLAS POWER MMA | CMake upstream chứa POWER10 SGEMM, DGEMM, QGEMM; probe compiler MMA, runtime HWCAP MMA/ISA3.1 | Commit và binary đã dựng phải có kernel đó. CPUExecutionProvider không đảm bảo mọi operator chạy MMA. |
| PyTorch ppc64le | IBM công bố Power-optimized wheels cho Power10/11, Python 3.11/3.12 | Phải giải dependency đúng Python/glibc và kiểm tra torch/torchvision custom ops. |
| RF-DETR train/export | Roboflow có API train, ONNX export, inference | Không tìm thấy cam kết chính thức rằng toàn bộ RF-DETR đã được kiểm thử trên RHEL9.8 ppc64le. |
| Spyre RF-DETR | Stack đã công bố có vLLM, driver, runtime và danh sách model | **Chưa xác minh hỗ trợ RF-DETR**. Không chạy lệnh biên dịch Spyre suy đoán. |
| TensorRT trên RTX3060 | NVIDIA xác nhận CC8.6; TensorRT có support matrix theo release | Chốt release hợp driver/OS/GPU thực tế; build engine trên máy đích. Không dùng TensorRT trên CPU ppc64le. |

Nguồn cho ba dòng CPU: [ORT build](https://onnxruntime.ai/docs/build/inferencing.html), [MLAS CMake](https://github.com/microsoft/onnxruntime/blob/main/cmake/onnxruntime_mlas.cmake), [IBM wheels](https://www.ibm.com/docs/en/aiservices/2026.03.0?topic=matter-power-optimized-python-wheels). Nguồn Spyre: [IBM introduction](https://www.ibm.com/docs/en/ibm-spyre-for-power?topic=power-introduction-spyre-accelerator), [Red Hat supported configurations](https://access.redhat.com/articles/rhaii-supported-configs).

## Chọn package có thể tái lập

Tài liệu IBM dẫn repository `https://wheels.developerfirst.ibm.com/ppc64le/linux-v2026.03.31`; chỉ mục `linux-v2026.09.0` cũng trả về danh sách torch khi kiểm tra. Tên stage theo ngày **không được xem như lockfile bất biến**. Trong phiên này chưa xác minh được đầy đủ wheel torchvision/onnxruntime cho một bộ ABI cụ thể, nên baseline không khẳng định một tổ hợp pin sẵn đã chạy.

Quy trình trên máy đích: tạo venv trong thư mục dự án; dùng `python -m pip debug --verbose` kiểm tra supported tags; tải trước wheel vào wheelhouse; chỉ chấp nhận binary đúng ppc64le; kiểm tra SHA256; cài offline từ wheelhouse và chạy `pip check` cùng smoke test. Nếu thiếu wheel, dừng và dựng source ở prefix riêng hoặc dùng máy đối chứng cho train/export. Không đổi Python hệ thống, không `sudo pip`, không thay lib hệ thống, không dùng wheel x86_64 qua đổi tên file.

[IBM wheel index torch](https://wheels.developerfirst.ibm.com/ppc64le/linux-v2026.09.0/torch), [IBM wheel documentation](https://www.ibm.com/docs/en/aiservices/2026.03.0?topic=matter-power-optimized-python-wheels).

## RF-DETR API và giao ước tensor

Tài liệu hiện tại yêu cầu extra `train` khi training và `onnx` khi export. Đây là tên dependency của upstream; trên ppc64le vẫn phải kiểm tra khả năng giải dependency. CPU dùng batch số nguyên; không lấy auto-batch CUDA làm cấu hình Power. Ghi phiên bản package và chữ ký `train`/`export` trước khi chạy; khóa một release đã qua smoke test. [Install](https://rfdetr.roboflow.com/latest/getting-started/install/), [Train](https://rfdetr.roboflow.com/latest/learn/train/), [Training parameters](https://rfdetr.roboflow.com/latest/learn/train/training-parameters/).

ONNX trả tensor thô `dets` (box `cxcywh` chuẩn hóa) và `labels` (logits); phải ghép theo tên, không đoán theo shape. Logits dùng sigmoid. Background phụ thuộc mapping checkpoint: dữ liệu custom ID liên tục thường là slot cuối; pretrained COCO sparse-ID giữ tất cả slot. Decoder chính thức xếp hạng query/class toàn cục, nên một query có thể có nhiều class. Export hiện tại có thể đặt tên theo variant, không luôn là `inference_model.onnx`. Preprocess, category mapping, background, top-k và threshold phải cố định giữa máy. [ONNX guide](https://rfdetr.roboflow.com/latest/exports/onnx/), [Export basics](https://rfdetr.roboflow.com/latest/exports/basics/).

## Bằng chứng dùng MMA và cải tiến C++

1. Lưu commit ORT và build log. Kiểm tra compiler nhận `-mcpu=power10`, phép probe MMA thành công và compile các file `*POWER10*`/`qgemm_kernel_power10.cpp`.
2. Kiểm tra LPAR thực sự expose MMA/ISA cần thiết. Dùng profile call stack/symbol để tìm kernel MLAS trên model thật; disassembly chỉ chứng minh binary chứa lệnh, chưa chứng minh hot path thực thi.
3. ORT profile cho biết operator/provider; kết hợp `perf` khi có quyền để chứng minh hotspot. Không bịa tên PMU event; lấy event có sẵn bằng `perf list` trên đúng server.
4. Chỉ chuyển preprocess/decoder sang C++ sau khi profile cho thấy đáng làm. Giữ cùng input tensor và so sánh numerical/accuracy. Custom MMA kernel là nhánh cuối: cần kiểm correctness, alignment, tail, threading và NUMA trước đo tốc độ.

Đây là kế hoạch kiểm chứng của baseline, suy ra từ cấu trúc dispatch trong [MLAS source](https://github.com/microsoft/onnxruntime/blob/main/cmake/onnxruntime_mlas.cmake) và [MLAS test description](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/mlas/README.md); không phải công bố RF-DETR đã tối ưu hoàn toàn cho Power11.

## TensorRT đối chứng

Giữ một ONNX FP32 chung làm gốc; so sánh CPU FP32 với TensorRT FP32 đã kiểm precision/TF32; báo thêm TensorRT FP16 như một cấu hình riêng và chạy accuracy gate. Không gộp số model-only không transfer với số CPU end-to-end. Serialized engine có giới hạn tương thích platform/hardware/version; lưu engine cùng manifest và build trên GPU đích. Release TensorRT hiện tại phải đối chiếu ma trận của NVIDIA, không mặc định bản mới nhất hợp máy.

[NVIDIA support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html), [NVIDIA performance practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html).

## ONNX Runtime CPU source build ở prefix riêng

Upstream liệt kê Linux PPC64LE trong kiến trúc build được hỗ trợ. Bằng chứng kernel cụ thể đã đọc ở `v1.23.2` và `main`; `v1.25.0` là một **ứng viên pin để thử**, có trang package chính thức và source build script, chưa xác nhận tương thích toàn bộ RF-DETR trên RHEL 9.8. Dùng `git ls-remote --tags` xác nhận tag, clone vào vùng dự án, ghi SHA thực, đọc yêu cầu CMake/Python/compiler của chính tag, rồi chạy từ checkout đó:

```bash
./build.sh --help > build-help.txt
./build.sh --config Release --parallel 8 --build_wheel --build_shared_lib > build.log 2>&1
```

`8` chỉ giới hạn tải build ban đầu, không phải số thread inference tối ưu. Lệnh native mặc định chạy các pha update/build/test; không bỏ test trong hồ sơ nghiệm thu. Wheel CPU nằm dưới `build/Linux/Release/dist/` theo mặc định. Không thêm một cờ tưởng tượng như `--enable_mma`, `--use_openblas` hay `onnxruntime_USE_BLAS=ON` để khẳng định đường CPU MLAS. CMake upstream tự kiểm tra khả năng compiler và runtime Power10; kiểm tra build log và hot path. Không dùng `-mcpu=native` toàn cục trước khi xác định phạm vi máy triển khai. [Build instructions](https://onnxruntime.ai/docs/build/inferencing.html), [MLAS v1.23.2](https://github.com/microsoft/onnxruntime/blob/v1.23.2/cmake/onnxruntime_mlas.cmake), [build.py v1.25.0](https://github.com/microsoft/onnxruntime/blob/v1.25.0/tools/ci_build/build.py), [ORT package v1.25.0](https://pypi.org/project/onnxruntime/1.25.0/).

## Giới hạn của kết luận Spyre

Tài liệu IBM hiện tại nêu enablement qua RHAIIS/RHOAI và các RHEL được hỗ trợ gồm 9.8. Ma trận Red Hat có stack vLLM Spyre ppc64le. Điều này **không xác nhận** RF-DETR, ONNX CPU EP hoặc một model CV bất kỳ tự chạy trên Spyre. Báo cáo YOLOv5-face v1 là chứng cứ lịch sử cho model/stack/thời điểm trong báo cáo; không dùng để phủ định hoặc đảm bảo support RF-DETR hiện tại. Thử nghiệm Spyre cần lấy danh sách supported models/operators và hướng dẫn triển khai đúng release từ IBM/Red Hat, rồi ghi rõ full offload, partition/fallback, precision và accuracy. Không tự gán một PCI vendor/device ID chưa đối chiếu phần cứng. [IBM Spyre](https://www.ibm.com/docs/en/ibm-spyre-for-power?topic=power-introduction-spyre-accelerator), [Red Hat supported configurations](https://access.redhat.com/articles/rhaii-supported-configs).
