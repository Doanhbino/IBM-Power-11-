# Đối chứng RTX 3060 và TensorRT

Đối chiếu hướng dẫn NVIDIA ngày **25-11-2026**. Không có GPU được truy cập trong phiên dựng artifact; các lệnh dưới đây là workflow từ tài liệu chính thức, cần kiểm tra `trtexec --help` của binary đã cài. `trtexec` không đi kèm pip wheel TensorRT; dùng binary của distribution/container NVIDIA phù hợp. [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/).

## Tránh nhầm TensorRT 10.x và 11.x

| Hành vi | TensorRT 10.x | TensorRT 11.x |
|---|---|---|
| FP16 | `--fp16` cho phép tactic precision thấp | Cờ này đã bỏ; graph strongly typed, xuất/convert graph FP16 trước |
| H2D/D2H mặc định trong benchmark | Có | Không |
| `--noDataTransfers` | Loại transfer khỏi phép đo | Được chấp nhận nhưng không còn tác dụng |
| Bật transfer | Bỏ `--noDataTransfers` | `--includeDataTransfers` |
| CUDA Graph / spin wait | Chọn qua cờ | Mặc định bật; dùng `--noCudaGraph`, `--noSpinWait` để tắt |

Không copy lệnh `--fp16` từ bài hướng dẫn cũ sang TensorRT 11. [NVIDIA migration guide 11.5](https://docs.nvidia.com/deeplearning/tensorrt/11.5.0/api/migration/tensorrt-10x-to-11x-trtexec.html).

## Các lượt chạy bắt buộc

Lưu ONNX FP32 giống hệt file trên Power, SHA256, một tensor đã preprocess giống nhau, input/output name/shape. Ví dụ sau giả định static batch 1 đã export; không thêm `--shapes` nếu graph static không khớp. Nếu graph dynamic, lấy tên thật và thêm min/opt/max theo `--help`. Tạo thư mục kết quả trước.

```bash
mkdir -p results/nvidia
nvidia-smi -q > results/nvidia/gpu.txt
trtexec --help > results/nvidia/trtexec-help.txt
# Build strict-FP32 candidate. Không thêm --fp16; tắt TF32.
trtexec --onnx=models/rfdetr.onnx --saveEngine=results/nvidia/fp32.engine --noTF32 --skipInference > results/nvidia/build-fp32.log 2>&1
# TensorRT 11: graph-only device compute; bao gồm transfer ở lượt riêng bên dưới.
trtexec --loadEngine=results/nvidia/fp32.engine --warmUp=1000 --duration=60 --iterations=1000 --exportTimes=results/nvidia/device-times.json > results/nvidia/device.log 2>&1
trtexec --loadEngine=results/nvidia/fp32.engine --includeDataTransfers --warmUp=1000 --duration=60 --iterations=1000 --exportTimes=results/nvidia/transfer-times.json > results/nvidia/transfer.log 2>&1
```

[Nguồn cờ benchmark](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html), [nguồn exportTimes](https://github.com/NVIDIA/TensorRT/blob/main/samples/trtexec/README.md).

Trên TensorRT 10, lượt device compute thêm `--noDataTransfers`; lượt transfer bỏ `--includeDataTransfers`. Không chạy dòng 11.x trên 10.x nếu `--help` không có cờ. `--exportTimes` có trong sample trtexec upstream; kiểm tra binary thực trước khi chạy. Có thể dùng `--loadInputs=input:path.bin` sau khi đổi `input` thành tên thật và file raw đúng dtype/layout. [trtexec sample](https://github.com/NVIDIA/TensorRT/blob/main/samples/trtexec/README.md), [NVIDIA benchmarking](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html).

Để profile layer, build một engine riêng với `--profilingVerbosity=detailed`; chạy `--dumpProfile --exportProfile=results/nvidia/layers.json`. Chạy latency chính thức riêng, không dùng profile có instrumentation làm số chính. Với 11.x, precision nằm trong graph; tạo ONNX FP16 qua RF-DETR exporter đã kiểm chữ ký hoặc chuyển đổi đã validation, lưu hash riêng và build không dùng `--fp16`. Với 10.x, có thể build từ FP32 ONNX thêm `--fp16` để tạo cấu hình mixed precision; phải ghi đúng tên cấu hình. Engine có thể dùng nhiều precision nội bộ; kiểm tra layer info và accuracy.

## Quy tắc so sánh do baseline quy định

- CPU ORT wall time `session.run` trả output và GPU device time là hai phạm vi khác nhau. Báo cả host wall time, GPU compute và host latency có H2D/D2H; không dùng một cột speedup chung cho các phạm vi khác nhau.
- E2E dùng cùng camera/replay video, codec, độ phân giải, frame IDs, chính sách drop/queue, preprocess, postprocess, threshold và top-k. TensorRT native trong pipeline cần đo đồng bộ completion trước khi lấy timestamp; enqueue bất đồng bộ không phải latency.
- FP32 CPU vs FP32 TensorRT tắt TF32 là đối chứng accuracy ban đầu. TF32/FP16/INT8 là các cấu hình tối ưu riêng, qua gate mAP/recall trên test set bất biến; không nói FP16 tương đương FP32 chỉ vì cùng checkpoint.
- Trên máy NVIDIA ghi CPU host và chi phí decode/resize/transfer; Power ghi entitlement/physc/lparstat, CPU affinity và NUMA. Đổi thứ tự lượt chạy, lặp tối thiểu 3 lần, ghi nhiệt độ/clock/load; không khóa clock hay thay power limit trong script mặc định.
- `trtexec` dùng random data theo mặc định, chỉ kiểm hiệu năng graph. Kết quả đó không thay accuracy test và không phải camera benchmark. Mỗi engine cần kiểm đầu ra/detection trên cùng dữ liệu trước khi báo cáo.

RTX 3060 thuộc CC 8.6 theo [NVIDIA GPU table](https://www.nvidia.com/content/cuda/cuda-gpus.html). Chọn bộ driver/CUDA/TensorRT từ [support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html); engine serialized không mặc định di chuyển được sang platform/GPU/version khác.

## Adapter TensorRT native cho cùng pipeline

`configs/rtx3060-native.json` và `scripts/trt_native.py` cho phép dùng engine tĩnh do `trtexec` dựng trong cùng `benchmark.py`/`evaluate_coco.py`; không đi qua ORT và không có fallback CPU/CUDA EP. Cần Python TensorRT chuẩn 10/11 + CUDA PyTorch tương thích GPU, dùng PyTorch để giữ buffer/stream; không nhầm với package TensorRT-RTX. Adapter **chưa chạy GPU trong phiên bàn giao**; phải qua smoke/parity trước khi đo. Chỉ hỗ trợ một input NCHW, I/O FP32/FP16, shape cố định dương; shape tensor/dynamic/data-dependent output sẽ dừng.

```bash
# Sau khi build engine FP32 đúng GPU đích và cập nhật đường dẫn config:
python scripts/validate_onnx.py --model models/rfdetr.onnx --backend-config configs/rtx3060-native.json --reference runs/export/eager_reference.npz --output runs/native-parity.json
python scripts/benchmark.py --config configs/rtx3060-native.json --output runs/native-model-r1
# Đổi mode=camera/source/size ở bản sao config để chạy cùng replay.
# Accuracy dùng cùng script và category mapping đã chốt:
python scripts/evaluate_coco.py --config configs/rtx3060-native.json --annotations data/test/_annotations.coco.json --images data/test --output-dir runs/native-accuracy
```

Thêm `category_ids` thực trước lệnh accuracy; config mẫu chưa biết nhãn của dataset. Adapter preallocate device buffers, enqueue trên CUDA stream và đồng bộ completion; output đưa về host trước khi trả kết quả. Đây là **host-call latency có H2D/D2H/wrapper**, không phải device-only GPU time; số device-only vẫn lấy từ `trtexec` ở lượt riêng. Lưu SHA engine cùng hash ONNX/build log để chứng minh engine đúng nguồn. `--profile` ORT không áp dụng native adapter; dùng `trtexec`/Nsight. API tensor names, `set_tensor_address` và `execute_async_v3` đã đối chiếu [TensorRT Python API 11.5](https://docs.nvidia.com/deeplearning/tensorrt/11.5.0/inference-library/python-api-docs.html), [10.x](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/python-api-docs.html).
