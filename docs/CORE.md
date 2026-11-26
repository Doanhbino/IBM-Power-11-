# Core C++ và quy trình chạy trên server

Core nằm trong `core/`, dùng C++17. Python giữ nhiệm vụ điều phối thí nghiệm, hashing và tổng hợp báo cáo. Không có dependency ML trong thư viện `rfdetr_core` portable.

## Thành phần đã triển khai

| Thành phần | Trách nhiệm |
|---|---|
| `Preprocessor` | BGR uint8 có stride → RGB FP32 NCHW; resize half-pixel không antialias; tái sử dụng tensor và bảng tọa độ |
| `Decoder` | Validate output hữu hạn; sigmoid, global query/class top-k, background rõ ràng; tie theo thứ tự slot/query; clip box; không NMS |
| `BoundedQueue<T>` | Replay chờ khi đầy; live bỏ phần tử cũ nhất; close để drain, cancel để giải phóng producer; counters được khóa |
| `placement` | Lấy cpuset thực, kiểm mask, bind CPU và memory policy trước khi khởi tạo model; lỗi bind được trả về |
| `OrtSession` | Một input FP32 batch 1; kiểm tên, dtype và shape; ORT CPU/MLAS từ shared library đã chọn; giữ output tới lần Run tiếp theo |
| `camera_pipeline` | Một capture thread/stream, queue riêng, một inference consumer/worker, round-robin; signal/reconnect/timeout và join thread |
| `PackedRhs` / `GemmWorkspace` | FP32 C=A×B, logical strides, pack B tái sử dụng, pack A theo panel, tile 4×4 và tail |
| `gemm_mma.cpp` | Kernel thử nghiệm ISA 3.1 dùng MMA FP32; chỉ compile translation unit này với `-mcpu=power10 -mmma` |

Kernel GEMM riêng phục vụ nghiên cứu shape/packing/correctness. **Kernel này chưa được tích hợp vào graph ORT hoặc PyTorch và chưa được benchmark trên Power11.** Đường RF-DETR chính dùng backend MLAS/PyTorch Power có MMA đã được kiểm chứng. Chỉ đưa kernel riêng vào graph sau khi profile chứng minh khoảng trống và A/B chứng minh có lợi. Không thay backend ORT bằng việc cài OpenBLAS.

Tham chiếu API: [GCC MMA intrinsics](https://gcc.gnu.org/onlinedocs/gcc/PowerPC-Matrix-Multiply-Assist-Built-in-Functions.html), [ORT Session C++](https://onnxruntime.ai/docs/api/c/struct_ort_1_1_session.html). Cờ ISA phải qua compile probe; không đặt cờ Power10 cho toàn binary. Kernel chỉ được gọi khi build có kernel và Linux `AT_HWCAP2` expose `PPC_FEATURE2_MMA`. Yêu cầu backend `mma` sẽ lỗi nếu thiếu điều kiện, không âm thầm fallback.

## Build

Trên máy thường, chỉ cần CMake >=3.20 và compiler C++17 có sẵn:

```bash
bash scripts/build_core.sh portable build
python -m unittest discover -s tests -v
```

Trên Power11, sau khi xác nhận toolchain:

```bash
python3 scripts/inventory.py --output runs/preflight/inventory.json
bash scripts/build_core.sh power build
build/rfdetr_gemm --capability
```

Build camera cần **SDK C/C++** của đúng build ORT được chọn, với `include/onnxruntime_cxx_api.h`, các header phụ thuộc và `lib/libonnxruntime.so`; Python wheel đơn lẻ có thể thiếu header. Cần OpenCV >=4 với modules core/videoio, FFmpeg backend và dependency tương thích ppc64le. Script không cài package hoặc đổi toolchain hệ thống.

```bash
ONNXRUNTIME_ROOT=/home/models/stack/onnxruntime \
OpenCV_DIR=/path/to/opencv/lib/cmake/opencv4 \
BUILD_JOBS=4 bash scripts/build_core.sh power-camera build-camera
```

Giữ `configure.log`, `build.log`, `ctest.log`, `compile_commands.json` và `build-manifest.json`. CTest chạy GEMM reference trên mọi máy và bổ sung kiểm GEMM MMA với FP64 oracle khi MMA compiled+exposed. Compile probe/build thành công không chứng minh RF-DETR đã chạy MMA.

## Camera replay/live

1. Sao chép `configs/camera-core.json`, sửa model, SDK library, binary và nguồn video. Xác nhận input/output/background/category mapping đúng checkpoint. `cpus: []` trong template cố ý chưa hợp lệ.
2. Sinh placement **trên server**, không dùng CPU IDs của máy phát triển. Thêm các source vào template trước khi sinh plan; mỗi source cần ID riêng.
3. Chạy plan-only để kiểm mask/node, sau đó chạy chiến dịch thật với output mới.

```bash
python scripts/worker_plan.py --config configs/camera-core.json \
  --output configs/camera-local.json --threads 4 --smt-limit 1 --max-workers 4
python scripts/camera_service.py --config configs/camera-local.json \
  --output runs/camera-plan --plan-only
python scripts/camera_service.py --config configs/camera-local.json \
  --output runs/camera-replay-r1
```

Một process cho mỗi worker, tối đa một worker trên mỗi node trong generator. CPU mask được kiểm lại khi chạy; memory bind không thực hiện được sẽ làm worker thất bại. Model và tensor được khởi tạo sau placement. Số ORT thread không vượt mask. Capture/decode cũng tiêu thụ CPU trên mask này; cần profile số thread của codec/FFmpeg và dự trù ngân sách capture khi sweep. `cv::setNumThreads(1)` không tự khống chế mọi thread của codec.

Live: đặt `session.mode="live"`, `duration_s=3600` cho soak, `max_frame_age_ms` theo SLA, source RTSP thật, điều chỉnh `timeout_s` lớn hơn tổng session-init/warmup/soak/shutdown. Retry hữu hạn bằng `reconnect_attempts`, mỗi lần chờ `reconnect_delay_ms`; remote FFmpeg open/read có timeout. Thiết lập nguồn local/device có thể không hỗ trợ timeout giống RTSP. Replay không bỏ frame theo age; không đặt giới hạn duration/max_frames nếu muốn xử lý toàn video.

`run.json` ở parent có Run ID, hash model/runtime/binary/source replay, config đã che source, thống kê worker và throughput trong cửa sổ đồng thời. Worker ghi:

- `samples.csv`: camera/frame ID, PTS, timestamp monotonic, read/capture, queue age, preprocess, inference, postprocess, receive-to-result và read-to-result.
- `detections.jsonl`: box, confidence và **class slot thô**, chưa ánh xạ category của dataset.
- `result.json`: số nhận/xử lý/bỏ do queue, reject lúc shutdown, bỏ do age, lỗi/reconnect, high-watermark và version/library ORT thực được load.

Parent so hash library thực load với library khai báo. Tổng throughput được tính từ số frame hoàn thành/cửa sổ chạy chung và có thêm cửa sổ overlap; không cộng FPS của những lượt riêng. Camera benchmark gồm capture startup và output IO, chưa gọi là steady-state. Timestamp nhận và PTS không phải timestamp sensor đã đồng bộ.

Backend `VideoCapture.read()` có thể không phân biệt EOF sạch với video hỏng ở cuối. Có thể thêm `expected_frames` vào mỗi source replay; parent sẽ yêu cầu số nhận và số xử lý khớp số này và ghi `replay_frame_counts_verified`. Dùng thêm frame IDs/manifest để xác minh đúng dữ liệu. Nguồn decode chậm chỉ ảnh hưởng queue/camera đó. Khi inference hoặc capture lỗi nghiêm trọng, parent dừng các peer và lưu trạng thái thất bại. Worker config chứa credential được ghi mode 0600 và xóa khi kết thúc; dùng thư mục run chỉ tài khoản vận hành có quyền truy cập.

## Shape, kernel và bằng chứng MMA

Lượt thu bằng chứng dùng stack ứng viên, `require_mma=false`; vẫn ghi MMA unverified. Sau khi review bằng chứng, bật gate cho chiến dịch KPI. Quy trình này tránh đòi dossier trước khi thu được trace đầu tiên.

```bash
python scripts/benchmark.py --config configs/cpu.json --output runs/power-profile --profile
# Dùng profile_path thực trong result.json:
python scripts/profile_ops.py runs/power-profile/ort_TIMESTAMP.json \
  --model runs/power-profile/optimized.onnx \
  --output runs/operators.csv --shapes runs/gemm-shapes.json
python scripts/gemm_sweep.py --binary build/rfdetr_gemm --shapes runs/gemm-shapes.json \
  --output runs/gemm-ab --backends reference mma --repeats 3
```

`profile_ops.py` trích MatMul/mm/bmm/addmm/linear khi đủ shape, không đoán transpose của fused node. Với ORT, `--model` phải là optimized graph của chính session; nó khôi phục shape initializer bị prepack bỏ khỏi trace theo tên node duy nhất. Node thiếu tên hoặc không khớp không được đoán. Profile chỉ có logical dimensions; microbenchmark đo từng matrix, chưa mô phỏng batch/broadcast/layout/stride nguyên gốc. Timer ghi riêng B packing và phần A packing+kernel+copy; không gọi phần sau là kernel-only.

Sao chép `configs/mma-evidence.template.json`, điền **bằng chứng thật**. Dossier có inventory HWCAP2, revision/compiler/flags/build log, hash runtime binaries, profiled run, operator profile, và sampled stacks chứa symbol kernel; reviewer phải giải thích mapping node → kernel → instruction/dispatch. Dùng symbolized trace từ công cụ có quyền trên server; xác minh PMU events trên chính máy trước khi diễn giải.

```bash
python scripts/mma_evidence.py --evidence runs/evidence/inference.json \
  --workload inference --model models/rfdetr.onnx --output runs/evidence/inference-gate.json
```

Gate kiểm hash, Run ID, workload, model, inventory, profile gắn với run và symbol có trong trace. Gate kiểm **tính nhất quán của hồ sơ đã được người có chuyên môn review**, không tự đọc ngữ nghĩa assembly để chứng minh MMA. Không dùng kernel chứa sẵn trong binary hoặc `CPUExecutionProvider` làm bằng chứng thực thi.

Để bắt buộc hồ sơ khi benchmark, thêm vào config:

```json
{"require_mma": true, "mma_evidence": "runs/evidence/inference.json"}
```

Hồ sơ Python ORT phải bao phủ tập `runtime_binaries` mà harness đã ghi; bổ sung các thư viện backend thực liên kết vào `stack.binaries` và lưu linkage/trace trong hồ sơ. Với ONNX external data, sao chép cả `model_files` từ run manifest vào dossier để khóa tensor ngoài graph. Hồ sơ C++ camera gắn với shared library ORT thực load. Khi đổi binary/model, thu và review hồ sơ mới.

## A/B và nghiệm thu trên server

Chạy ít nhất ba lượt **không profiler** cho mỗi cấu hình; giữ đúng model/input/FP32 và ranh giới timer. Kết quả kernel A/B cần giữ nguyên CPU/thread; khi thay toàn stack hoặc phần cứng, báo đúng phạm vi cải thiện:

```bash
python scripts/compare_runs.py \
  --reference runs/r0-1 runs/r0-2 runs/r0-3 \
  --candidate runs/r1-1 runs/r1-2 runs/r1-3 \
  --comparison stack --output runs/ab.json
```

Công cụ từ chối run trùng, failed, profiled, synthetic smoke, khác model/input/shape/dtype/timer hoặc runtime/config bị đổi trong cùng nhóm repeat. Báo mean/stdev/range qua repeat; không suy statistical significance chỉ từ range. Numerical parity, COCO accuracy/recall, MMA evidence và SLA/soak là các gate độc lập.

## Trạng thái bàn giao

Core portable đã build/test trên x86; parity NumPy–C++, queue lifecycle và numerical GEMM được kiểm tại máy phát triển. Camera/OpenCV translation units được kiểm syntax với header có sẵn. **Chưa link/chạy camera C++ với SDK ORT tại máy này**, chưa chạy training/export, MMA instruction hoặc đo performance Power11/RTX. Việc tải SDK bổ sung chưa được thực hiện. Các script/config/build targets đã được chuẩn bị để build và kiểm chứng trên server.

Đối chiếu các bước P0–P5 trong [kế hoạch](POWER11_PLAN.md). Phần còn lại trên server là build/dependency compatibility, dữ liệu/checkpoint, trace MMA training/inference, sweep/NUMA, accuracy, camera soak và đối chứng RTX; chưa có kết quả nghiệm thu phần cứng.
