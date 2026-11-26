# Những gì đã kiểm thử khi bàn giao

## Bổ sung core C++ — 25/11/2026

Bản cập nhật có CMake/C++17 core, adapter parity bridge, camera worker/launcher, CPU/NUMA planner, kernel FP32 reference/MMA thử nghiệm, shape extraction, run manifests, evidence gate và A/B. Xem [CORE.md](CORE.md) để build trên server.

- Release CMake build và CTest portable: đạt; gồm queue block/drop/close/cancel, decoder, 160 tổ hợp GEMM có tail/transpose B so FP64 oracle, transpose A, shared packed B với workspace riêng và từ chối MMA khi không có ISA/kernel.
- 29 Python unittest: đạt; gồm 20 tổ hợp ảnh/resize có stride đệm so NumPy–C++, decoder random/tie/background, profile ORT thật với prepacked initializer → optimized graph → shape → ba lần benchmark C++ reference, evidence tampering/workload/binary checks và placement/A/B gates. Log cập nhật ở `reports/core-python-tests.txt`.
- AddressSanitizer + UndefinedBehaviorSanitizer core tests: đạt với `ASAN_OPTIONS=detect_leaks=0`. LeakSanitizer không chạy được trong môi trường có ptrace; chưa kiểm memory leak bằng LSan.
- Camera app và pipeline: kiểm syntax với OpenCV header 4.6 có sẵn. **Chưa link/chạy worker C++ với ONNX Runtime SDK**. Adapter `ort_session.cpp` cần compile/link với SDK trên server; không có SDK C++ đầy đủ tại máy phát triển.
- Chưa chạy instruction MMA trên Power, training/export RF-DETR, dataset thật, camera soak hoặc RTX. Kernel 4×4 MMA còn thử nghiệm và chưa tích hợp vào graph ORT/PyTorch. Không dùng các test portable này làm số đo Power11.

Log/metadata mới: `reports/core-python-tests.txt`, `reports/core-ctest.txt`, `reports/core-sanitizer.txt`, `reports/core-camera-syntax.txt`, `reports/core-verification.json`. Bảng bên dưới là hồ sơ harness trước khi bổ sung core.

Ngày 25-11-2026. Host thực thi là **x86_64 Linux / Python 3.12.3**, không phải server Power11 hoặc RTX 3060. Các dependency tải vào venv riêng trong workspace; không sửa Python hệ thống. Dependency x86 dùng để thử harness: numpy 2.5.5, onnx 1.23.2, onnxruntime 1.32.0, Pillow 13.0.0, OpenCV headless 5.1.0.95, pycocotools 2.0.11. **Không dùng danh sách này làm lockfile ppc64le.**

| Kiểm tra thực hiện | Kết quả / giới hạn |
|---|---|
| Python compile và CLI help; bash syntax setup | Đạt; không chứng minh wheel ppc64le cài được |
| 14 unittest | Đạt; log `reports/tests.txt` |
| Adapter RGB normalization, half-pixel resize, multi-label/background, collision 3 class + background | Đạt trên fixture nhỏ; chưa đối chiếu toàn pipeline với PyTorch RF-DETR |
| TensorRT EP báo có nhưng session fallback về CPU | Harness từ chối kết quả; test mock, chưa chạy GPU |
| ONNX fixture → ORT → profile → CSV/report | Chạy thực, có CPU node events |
| Same-input numerical validator + reference sai cố ý | Positive đạt; reference sai trả exit 2 đúng yêu cầu |
| Video MJPEG tổng hợp → capture/preprocess/infer/decode | Chạy thực; kiểm frame count và output |
| COCO prediction/evaluator trên fixture | Chạy thực; không phải model accuracy trên dữ liệu người dùng |
| ONNX external tensor hash | Đạt với fixture một external file |
| Hai worker đồng thời, affinity không chồng, tổng ảnh/makespan | Đạt trên fixture; không suy ra scaling PowerVM |
| Parse topology/cpuset không liên tục | Đạt; không tự tạo CPU ID ngoài allowed set |
| Inventory, telemetry, sweep plan trên host hiện tại | Chạy thực, missing Power commands được ghi thiếu |
| Training `--check-only` với fixture đường dẫn/annotation | Đạt; không chạy forward/backward |
| Model ONNX người dùng có sẵn | Checker thực + ORT CPU smoke 3 measured calls, thêm profile riêng; output tensors đúng interface |

Model smoke: `rfdetr_small_7cls_static_640x640_new1665.onnx`, SHA256 `073b767a0caed9bcccd995d121ee3a39a46224fb977695f7badbf1c70bb03052`, input `input` FP32 `[1,3,640,640]`, outputs `dets=[1,300,4]`, `labels=[1,300,8]`, opset 17. Không có metadata suy category/background. Dùng tensor zero tổng hợp; không chạy ảnh thực hay mAP, không được lấy FPS/p99 của ba mẫu này làm số đánh giá sản phẩm. Lượt profile có 776 node groups với CPUExecutionProvider; không phải bằng chứng MMA vì host x86. Các report ví dụ là số đo thật có nhãn x86/synthetic/profiled và đã che đường dẫn cá nhân, không chứa weights.

Chưa kiểm chứng: ảnh cấu hình gốc, Power11 LPAR thật, wheel/build ppc64le, torch/torchvision/RF-DETR training hoặc export, eager prediction generator, training probe callbacks/profiler, numerical parity RF-DETR thật, dataset test thực, card/driver/runtime Spyre, RF-DETR trên Spyre, TensorRT native adapter/GPU/RTX accuracy hoặc telemetry công suất. Những phần này có workflow và gate trong repo, cần chạy trên đúng máy có dữ liệu/quyền cần thiết.

Các test có thể chạy lại trong môi trường đã giải dependency:

```bash
python -m unittest discover -s tests -v
python -m compileall -q scripts
bash -n scripts/setup_env.sh
```

Fixture test tự sinh mô hình ONNX nhỏ và video tạm, không cần model/dataset người dùng. Tests không chứng nhận tương thích chính thức của nhà cung cấp. Đọc `EVIDENCE.md`, `SUPPORT.md`, `RESEARCH.md` trước khi báo kết luận nghiên cứu.
