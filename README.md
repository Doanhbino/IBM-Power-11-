# Baseline nghiên cứu RF-DETR trên IBM Power11

**Core C++ đã được bổ sung**: preprocessing/decoder RF-DETR, queue camera hữu hạn, worker ORT, CPU/NUMA placement, GEMM FP32 reference và kernel MMA thử nghiệm, run/evidence gate và A/B. Xem [hướng dẫn core và build trên server](docs/CORE.md) cùng [kế hoạch triển khai](docs/POWER11_PLAN.md). Dùng `bash scripts/build_core.sh portable build` để kiểm core trên máy phát triển; `power-camera` để build trên server với SDK ORT/OpenCV đã chọn. Kernel riêng chưa được tích hợp vào ORT/PyTorch; đường RF-DETR tiếp tục dùng backend thư viện đã kiểm chứng.

Bộ harness có thể mang lên server để kiểm chứng, với model ONNX RF-DETR đã có sẵn (repo không train/export), **accuracy → inference → camera → thread/NUMA/multi-worker → báo cáo**. Đường chuẩn là **CPU FP32 / ONNX Runtime CPUExecutionProvider / MLAS**. Spyre PCIe có cổng kiểm chứng riêng; chưa có adapter RF-DETR/Spyre được xác nhận. Bộ này chưa được chạy trên Power11 của bạn, không chứa số FPS dự đoán.

Ngày đối chiếu: **25/11/2026**. Máy mục tiêu theo yêu cầu: RHEL 9.8 ppc64le, PowerVM, 240 logical CPU ở SMT8, 4 NUMA, RAM 954 GiB, `/home/models` 745G. 240 logical CPU không đồng nghĩa 240 core, và cũng không xác định entitlement của shared LPAR. Chưa lấy được ảnh/đầu ra cấu hình gốc từ chat cũ; xem [chứng cứ và phần còn thiếu](docs/EVIDENCE.md).

Đã đọc bản DOCX gốc “Báo cáo YOLOv5-face trên IBM Spyre v1.docx”. Báo cáo đó đo YOLOv5-face qua Spyre rời và RTX 3060, không phải RF-DETR chạy CPU MMA. Đừng dùng 47/61 FPS trong báo cáo cũ làm mục tiêu hoặc kết quả của baseline này. [Phân tích tài liệu](docs/EVIDENCE.md), [hỗ trợ đã xác minh](docs/SUPPORT.md), [nguồn chính thức](docs/SOURCES.md).

## Bắt đầu trên Power11

Giải nén repo vào thư mục riêng có quyền ghi, ví dụ `/home/models/power11-rfdetr-baseline`; chạy các lệnh từ gốc repo. Không cần root. Repo không tự đổi SMT/NUMA host, entitlement, driver, Python hệ thống hoặc cài RPM. Dùng thư mục `runs` riêng cho mỗi lần đo; lưu khoảng 15–20% đĩa trống là mục tiêu vận hành, kiểm dung lượng thực trước khi tải dữ liệu/checkpoint.

```bash
python3 scripts/inventory.py --output runs/preflight/inventory.json
```

Đọc inventory trước: kiến trúc/OS, cpuset/cgroup, CPU→core→sibling→NUMA, RAM/disk, lparstat/entitlement và PCI/driver. Thiếu lệnh hay quyền đọc được ghi là thiếu. Spyre chỉ được xác nhận sau khi định danh PCIe + driver + vendor runtime health, không suy từ tên Power11. Xem [hướng dẫn hệ thống](docs/SYSTEM.md).

Tách venv CPU inference và NVIDIA đối chứng. **requirements.in chỉ là danh sách, chưa là lock ppc64le đã được thử.** Dùng Python 3.11/3.12 đã có trên máy theo nguồn IBM phù hợp. Kiểm từng wheel/version/Python ABI; không mặc định `pip install torch` hay `onnxruntime` có wheel ppc64le.

Quy trình binary ưu tiên: tạo thư mục wheelhouse rỗng, resolve/download từ index đã chọn, lưu log nguồn; review wheel rồi sinh lock/hash; cài offline trong venv mới. Ví dụ dưới kiểm PyPI cho inference; nếu thiếu binary, dừng và đọc [support](docs/SUPPORT.md), [source build ORT](docs/BUILD_ORT.md), không tự build dependency hoặc thay toolchain hệ thống.

```bash
mkdir -p wheelhouse runs/setup
# Đây là phép dò trên đúng máy/ABI; có thể thất bại trên ppc64le và đó là kết quả cần giữ.
python3 -m pip download --only-binary=:all: --dest wheelhouse -r requirements.in > runs/setup/download.log 2>&1
# Sau khi kiểm nguồn/version/tag của TẤT CẢ wheel:
python3 scripts/wheel_lock.py wheelhouse --output requirements.cpu.lock
bash scripts/setup_env.sh probe
bash scripts/setup_env.sh install
source .venv-cpu/bin/activate
python -m pip check
python -m unittest discover -s tests -v
```

Các test tích hợp cần thêm OpenCV và `pycocotools`; resolve `requirements.camera.in` thay cho `requirements.in`, rồi tạo lock đầy đủ trước khi chạy. Nếu chỉ cần model-only, OpenCV và `pycocotools` không phải dependency của `benchmark.py`. Không cài đồng thời `onnxruntime` và `onnxruntime-gpu` vào cùng venv. `wheel_lock.py` chỉ cần Python chuẩn; hash không thay thế kiểm provenance/compatibility.

IBM wheel index có tài liệu riêng; cần xác nhận đúng package/version/ABI, không ghép nhiều index không kiểm soát. Model ONNX được tạo ở nơi khác (ví dụ máy GPU) rồi đưa sang Power; repo này không benchmark training.

## Điểm xuất phát

**Cần có model ONNX RF-DETR** (repo không train/export; tạo model ở nơi khác): kiểm tên/shape/dtype bằng ONNX, đi thẳng vào model-only rồi camera. `configs/cpu.json` và `camera.json` mặc định 640×640 để phù hợp model Small 7 lớp cục bộ đã kiểm checker. Mẫu background=-1 là giả định cho custom contiguous-ID, cần xác nhận checkpoint. Mapping class phải thiết lập theo dataset, không suy từ tên “7cls”.

Đối chiếu số học ONNX với PyTorch eager (tùy chọn) và đo accuracy COCO được mô tả trong [ACCURACY.md](docs/ACCURACY.md). Size trong mọi config inference/accuracy phải khớp với shape đã export (ví dụ không chạy camera 640 với model export 384).

## Benchmark model-only và camera

Sửa `model`, `size`, input/output names, mean/std, background, top-k/threshold trong config. CPU baseline FP32; input dtype sai sẽ dừng, không tự cast. Chạy cùng tensor trên hai máy để tách compute khỏi decode/resize. `prepare_input.py` stretch RGB, tensor bilinear half-pixel không antialias, normalize NCHW; cần qua preprocessing parity với revision RF-DETR thực dùng trước nghiệm thu accuracy.

```bash
mkdir -p data runs
# Đặt model thật ở models/rfdetr.onnx; sửa model/size/decode config theo graph.
python scripts/inspect_model.py models/rfdetr.onnx --check --output runs/model-info.json
# Đổi input thành tên thật từ model-info.json nếu khác; cùng ảnh nguồn cho hai máy.
python scripts/prepare_input.py --config configs/cpu.json --image data/sample.jpg --input-name input --output data/input.npz
python scripts/benchmark.py --config configs/cpu.json --output runs/cpu-t1-r1
# Profile riêng, không dùng số có instrumentation làm KPI.
python scripts/benchmark.py --config configs/cpu.json --output runs/cpu-profile --profile
# Thay ort_TIMESTAMP.json bằng profile_path trong result.json.
python scripts/profile_ops.py runs/cpu-profile/ort_TIMESTAMP.json --output runs/cpu-profile/operators.csv
python scripts/benchmark.py --config configs/camera.json --output runs/camera-r1
python scripts/report.py runs --output runs/report

```

`result.json` lưu model/input hash, external ONNX data hash, phiên bản/build/providers, affinity, session-init/first-call, p50/p95/p99/mean và FPS; `samples.csv` lưu từng lần đo và capture/preprocess/inference/postprocess. Warmup không vào KPI. Model-only là synchronous host NumPy→host NumPy, bao gồm transfers trên GPU EP. Camera sequential đo từ trước `read()` đến decoded detection, ghi frame index/PTS và replay hash; chưa đo sensor-to-result, source drop hay service queue latency. RTSP source được che khỏi manifest; dùng source_id không chứa thông tin đăng nhập. Xem [phạm vi đo](docs/RESEARCH.md).

RF-DETR detection decoder chuẩn dùng sigmoid và global query/class top-k. Không thêm NMS từ YOLOv5-face sang RF-DETR theo thói quen. Profile ORT tổng thời gian kernel có thể overlap và gồm warmup, không phải tổng wall; provider CPU không chứng minh MMA đã chạy.

## NUMA, thread, worker và accuracy

```bash
python scripts/sweep.py --config configs/sweep.cpu.json --output runs/sweep-plan --plan-only
# Đọc mask và thu hẹp grid trước khi chạy; case không đủ CPU sẽ không tạo.
python scripts/sweep.py --config configs/sweep.cpu.json --output runs/sweep-r1
python scripts/telemetry.py --output runs/telemetry-r1 --seconds 600 --interval 1
python scripts/dataset_manifest.py /home/models/datasets/custom-coco --output runs/dataset-files.json
```

Chạy telemetry song song ở terminal khác; có thể thêm `--pid`. Sweep chỉ chọn subset sibling từ sysfs/cpuset, không đổi SMT toàn hệ thống. Mask worker không chồng nhau. Tổng throughput là ảnh thực chia **parent makespan chạy đồng thời**, gồm startup/warmup; không cộng FPS độc lập. Muốn steady-state phục vụ dài hạn cần cửa sổ đồng bộ và queue harness nâng cấp. Xem [SYSTEM](docs/SYSTEM.md).

Accuracy sinh COCO predictions bằng cùng adapter hoặc đọc predictions từ eager/native TensorRT. Mẫu `accuracy-custom-example.json` là Nano một lớp để minh họa mapping, phải sửa cho model/dataset thực.

```bash
python scripts/evaluate_coco.py --config configs/accuracy-custom-example.json --annotations /home/models/datasets/custom-coco/test/_annotations.coco.json --images /home/models/datasets/custom-coco/test --output-dir runs/accuracy-cpu
# Dùng AP thực 0..1 khi gọi gate; xem tham số:
python scripts/accuracy_gate.py --help
```

Không dùng WIDER FACE AP của YOLO làm COCO AP RF-DETR. FP32 numerical parity và mAP/recall trên test set bất biến là hai gate riêng. Lưu category mapping, image IDs, annotation/image hash, seed và precision. [Nghiệm thu và thiết kế nghiên cứu](docs/RESEARCH.md).

Reference eager (nếu cần) phải được sinh độc lập ở nơi khác; xem [ACCURACY.md](docs/ACCURACY.md). Phải xác nhận resolution/num_select/mapping của checkpoint cùng cấu hình ONNX.

## RTX 3060 và bước tối ưu tiếp theo

[NVIDIA.md](docs/NVIDIA.md) có đối chứng TensorRT FP32 tắt TF32, FP16 riêng, tách device compute/transfer/host/E2E và cờ TensorRT 10/11. `configs/rtx3060-native.json` chạy engine TensorRT native qua buffer/stream CUDA với completion đồng bộ; chưa thử GPU trong phiên bàn giao. `configs/rtx3060-trt.json` dùng **ORT TensorRT EP có CUDA/CPU fallback**, cần profile provider assignment; không gọi là TensorRT native hay strict FP32 nếu chưa xác nhận precision/TF32 của runtime. Chạy accuracy trên kết quả thực của đúng backend. Engine cache/serialized engine phải gắn GPU/version/hash.

[Lộ trình C++/MMA](docs/RESEARCH.md): profile hotspot → C++ preprocess/decoder + buffer reuse + worker-local NUMA → kiểm MLAS MMA dispatch → tối ưu shapes/packing/precision → chỉ viết custom kernel khi microbenchmark và E2E chứng minh có lợi. Python harness đo và xác định nơi cần “code core”; nó không giả lập đã tối ưu nhất server.

## Kiểm chứng bộ bàn giao

Xem [VERIFICATION.md](docs/VERIFICATION.md) để biết script/test nào đã chạy trên x86, model cục bộ nào đã smoke, và phần nào còn phải chạy trên Power11/RTX. Không kèm DOCX gốc, model, dataset, khóa truy cập hoặc kết quả phần cứng giả. Tất cả nguồn hỗ trợ được liên kết trong [SOURCES](docs/SOURCES.md).
