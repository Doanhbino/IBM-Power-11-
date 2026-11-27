# Multi-model worker (RF-DETR / YOLO11) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mỗi worker camera chọn `model_type` = `rfdetr` hoặc `yolo11`, chạy cùng pipeline và cùng cách đo để so sánh hiệu năng hai model giao thông trên Power11.

**Architecture:** Bản tham chiếu NumPy (`scripts/common.py`) định nghĩa tiền xử lý letterbox, giải mã YOLO và NMS; harness Python (`benchmark.py`, `camera_service.py`) chọn adapter theo `model_type`. Lõi C++ thêm `LetterboxPreprocessor` + `YoloDecoder` và được kiểm bằng parity với bản NumPy. `OrtSession` và `camera_pipeline.cpp` rẽ nhánh theo `model_type`.

**Tech Stack:** Python 3.12 + NumPy + onnxruntime 1.27.0 (venv `.venv-cpu`), C++17 (`rfdetr_core` không phụ thuộc ML/OpenCV), CMake, ORT C++ SDK + OpenCV 4 chỉ cho target camera.

**Spec:** `docs/superpowers/specs/2026-11-26-multi-model-worker-design.md` (đã được duyệt, cách A).

## Global Constraints

- ORT CPU FP32, batch 1, `CPUExecutionProvider`; không fallback âm thầm giữa hai model.
- Hai model có 4 lớp cùng thứ tự: `Car`, `Motorcycle`, `Primitive_Vehicle`, `person` (slot 0..3). RF-DETR có thêm slot background cuối (`background_index: -1`, logits 5 chiều).
- Không đoán tên/shape output: worker kiểm hợp đồng output của ONNX lúc khởi tạo và dừng khi lệch.
- Ngưỡng YOLO11 (`conf_threshold`, `iou_threshold`, `max_det`) phải đặt rõ trong cấu hình, không có mặc định ẩn; mẫu dùng `0.25` / `0.7` / `300`. Đệm letterbox `114`.
- NMS theo từng lớp, mỗi anchor chỉ giữ lớp có điểm cao nhất, IoU so sánh `>` (nghiêm ngặt), hòa điểm chọn index nhỏ hơn, kết quả sắp theo điểm giảm dần.
- Làm tròn letterbox theo Python `round` (ties-to-even); trong C++ dùng `std::nearbyint`.
- Không đưa lại code training/export vào repo.
- `rfdetr_core` portable, C++17, không OpenCV/ORT; chỉ target camera mới dùng ORT + OpenCV 4 (`find_package(OpenCV 4 ...)` không chấp nhận OpenCV 5).
- Mọi thay đổi mã nguồn đi qua công cụ Review Hub; chỉ commit khi người dùng yêu cầu.
- Máy phát triển thiếu `onnx`/`cv2`: test Python cần hai thư viện này chạy trên server bằng `.venv-cpu`; test chỉ cần NumPy chạy được ở cả hai nơi.

## Review Focus

1. Ảnh nhỏ hơn/lớn hơn/không vuông so với input model (kể cả 1×1): letterbox đệm đúng hai trục, box ánh xạ ngược nằm trong khung gốc (Task 1, 4).
2. Không anchor nào vượt `conf_threshold`: kết quả rỗng, không lỗi. Output có NaN/Inf: ném lỗi như decoder RF-DETR (Task 1, 4).
3. Hai box khác lớp chồng nhau phải cùng sống; hai box cùng lớp có IoU đúng bằng ngưỡng không bị loại (Task 1, 4).
4. Cấu hình cũ không có `model_type` phải chạy y hệt như trước (RF-DETR) (Task 2, 3, 6).
5. `num_classes` khai báo khác `channels-4` của ONNX: lỗi khởi tạo nêu cả hai con số (Task 1, 6).
6. Chiến dịch có hai loại model mà `require_mma=true` bị từ chối, vì bằng chứng MMA là theo từng model (Task 3).

---

### Task 1: Tham chiếu NumPy cho letterbox, NMS, giải mã YOLO

**Files:**
- Modify: `scripts/common.py` (thêm các hàm mới sau `decode`)
- Create: `tests/test_yolo_adapter.py`

**Interfaces:**
- Produces (mọi task sau dùng):
  - `MODEL_TYPES = ('rfdetr', 'yolo11')`
  - `model_type(cfg: dict) -> str`: `cfg.get('model_type', 'rfdetr')`; `ValueError` nếu ngoài `MODEL_TYPES`.
  - `preprocess_letterbox(frame: np.ndarray, h: int, w: int, pad: int = 114) -> tuple[np.ndarray, dict]`: `frame` BGR uint8 HxWx3; trả tensor `(1,3,h,w)` float32 (RGB, `/255`, đệm = `pad/255`) và `meta = {'scale': float, 'pad_x': int, 'pad_y': int}`. `scale = min(h/ih, w/iw)`; `new_w = round(iw*scale)`, `new_h = round(ih*scale)`; `dw=(w-new_w)/2`, `dh=(h-new_h)/2`; `pad_x = round(dw-0.1)`, `pad_y = round(dh-0.1)` (quy tắc LetterBox của Ultralytics); nội dung resize bilinear nửa-pixel giống `preprocess_bgr`.
  - `nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]`: greedy, thứ tự điểm giảm dần, hòa điểm index nhỏ; loại box có `IoU > iou`.
  - `decode_yolo(outputs: dict, cfg: dict, original_hw: tuple[int,int], meta: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]`: trả `(xyxy float32 [K,4], scores [K], class_slots int [K])`. Đọc `outputs[cfg['output_name']]` dạng `[1, 4+C, N]`; yêu cầu khóa `output_name`, `num_classes`, `conf_threshold`, `iou_threshold`, `max_det` (thiếu → `ValueError` nêu tên khóa); `4+num_classes != channels` → `ValueError` nêu cả hai số.

- [ ] **Step 1: Viết test thất bại `tests/test_yolo_adapter.py`** (mẫu `tests/test_adapter.py`, chỉ cần NumPy). Các test và khẳng định:
  - `test_letterbox_wide_image`: ảnh 100×200 vào 640×640 → `scale == 3.2`, `pad_x == 0`, `pad_y == 160`; hàng 0 của tensor bằng `114/255`.
  - `test_letterbox_tall_and_tiny`: ảnh 1×1 và ảnh 200×100 vào 64×64; tensor shape `(1,3,64,64)`, toàn giá trị trong [0,1].
  - `test_letterbox_round_trip`: ánh xạ ngược `(x-pad_x)/scale` của góc nội dung về đúng `(0,0)` và `(iw,ih)` (sai số ≤ 1 pixel) cho 5 cỡ ảnh khác nhau.
  - `test_nms_same_class_and_threshold_equality`: hai box IoU đúng `0.5` với `iou=0.5` → cả hai sống; với `iou=0.49` → chỉ box điểm cao sống.
  - `test_nms_tie_prefers_lower_index`.
  - `test_decode_per_class_nms_keeps_other_class`: hai box trùng nhau khác lớp → 2 kết quả; cùng lớp → 1.
  - `test_decode_best_class_per_anchor`: một anchor có điểm 0.9 ở lớp 1 và 0.8 ở lớp 2 → chỉ sinh lớp 1.
  - `test_decode_empty_below_threshold` và `test_decode_nonfinite_raises`, `test_decode_max_det`.
  - `test_decode_maps_back_and_clips`: box ngoài biên được kẹp vào `[0,w]×[0,h]` của ảnh gốc.
  - `test_channel_mismatch_names_both_numbers`: `num_classes=4` với 84 kênh → `ValueError`, thông điệp chứa `4` và `80`.
  - `test_model_type_default_and_unknown`: không khóa → `'rfdetr'`; `'yolo9'` → `ValueError`.
- [ ] **Step 2: Chạy** `python3 -m unittest discover -s tests -p 'test_yolo_adapter.py' -v` → FAIL (`ImportError`).
- [ ] **Step 3: Cài đặt** các hàm trên trong `scripts/common.py`, dùng lại phép resize nửa-pixel của `preprocess_bgr` (tách hàm phụ dùng chung nếu tiện, không đổi hành vi `preprocess_bgr`). NMS theo lớp bằng cách gọi `nms` từng lớp. Trả kết quả sắp theo điểm giảm dần, cắt `max_det`.
- [ ] **Step 4: Chạy lại** cùng lệnh, và `python3 -m unittest discover -s tests -p 'test_adapter.py' -v` → PASS (không hồi quy RF-DETR).
- [ ] **Step 5: Đối chiếu letterbox với Ultralytics trên ảnh thật** (máy phát triển, venv `/home/doanh/export-traffic/venv-yolo`): lấy một khung từ một video giao thông, so `preprocess_letterbox` với `ultralytics.data.augment.LetterBox((640,640), auto=False)`; đồng nhất `scale/pad_x/pad_y`, sai khác pixel ≤ 2/255 trên vùng nội dung (khác biệt do cv2 dùng số nguyên). Ghi kết quả vào `docs/CORE.md` (Task 7). Nếu lệch hơn, dừng và báo người dùng.

### Task 2: `benchmark.py` chọn adapter theo `model_type`

**Files:**
- Modify: `scripts/common.py` (thêm `adapter_for`), `scripts/benchmark.py` (chế độ `camera`, ghi `model_type`)
- Test: `tests/test_harness.py` (thêm test), `tests/test_yolo_adapter.py`

**Interfaces:**
- Consumes: `model_type`, `preprocess_bgr`, `preprocess_letterbox`, `decode`, `decode_yolo` (Task 1).
- Produces: `adapter_for(cfg) -> Adapter` với `Adapter = namedtuple('Adapter', 'preprocess decode')`, trong đó `preprocess(frame, h, w, cfg) -> (tensor, meta_or_None)` và `decode(outputs_dict, cfg, original_hw, meta) -> (xyxy, scores, slots)`; `result.json` có thêm khóa `model_type`.

- [ ] **Step 1: Viết test thất bại** trong `tests/test_harness.py`: `test_yolo11_camera_mode`. Dựng fixture ONNX nhỏ (mẫu `setUp`) có input `images [1,3,16,16]` và output hằng `output0 [1,6,4]` (4 anchor, 2 lớp) trong đó đúng một anchor có điểm 0.9; cấu hình `model_type='yolo11'`, `output_name='output0'`, `num_classes=2`, `conf_threshold=0.25`, `iou_threshold=0.7`, `max_det=300`, `mode='camera'`. Khẳng định `result.json['model_type']=='yolo11'` và cột `detections` trong `samples.csv` bằng 1 ở mọi hàng. Thêm `test_default_model_type_is_rfdetr` kiểm `test_camera_and_coco` hiện có vẫn qua và `result.json['model_type']=='rfdetr'`.
- [ ] **Step 2: Chạy** `python3 -m unittest tests.test_harness -v` (trên server, `.venv-cpu`) → FAIL.
- [ ] **Step 3: Cài đặt** `adapter_for` và dùng nó trong `benchmark._run` ở chế độ `camera` thay cho các lời gọi `preprocess_bgr`/`decode` cứng; với `yolo11` kiểm `num_classes` với shape output thật của session ngay trước vòng đo. Ghi `model_type` vào `result`. Chế độ `model` không đổi.
- [ ] **Step 4: Chạy lại** `python3 -m unittest tests.test_harness tests.test_adapter -v` → PASS.

### Task 3: `camera_service.py` và `worker_plan.py` nhận `model_type` theo worker

**Files:**
- Modify: `scripts/camera_service.py` (`validate_config`, `run`), `scripts/worker_plan.py` (`place` giữ các khóa theo worker)
- Create: `tests/test_camera_service.py`

**Interfaces:**
- Consumes: `model_type`, `MODEL_TYPES` (Task 1).
- Produces: mỗi phần tử trong `config['workers']` có thể có `model_type`, `model` và các khóa YOLO (`output_name`, `num_classes`, `conf_threshold`, `iou_threshold`, `max_det`, `pad_value`); khóa thiếu lấy từ `session`. `validate_config(config, rows)` trả danh sách cfg theo worker đã gộp, mỗi cfg chứa `model_type` và `model`. `run.json` có `models: [{worker: int, model_type: str, path: str, sha256: str}]`; `model_sha256` chỉ đặt khi mọi worker dùng cùng một file.

- [ ] **Step 1: Viết test thất bại `tests/test_camera_service.py`** (hàm thuần, `rows=[{'cpu': i, 'node': 0} for i in range(8)]`, file replay tạm): `test_legacy_config_defaults_to_rfdetr`, `test_mixed_workers_merge_per_worker_keys`, `test_yolo_requires_explicit_thresholds` (thiếu `iou_threshold` → `ValueError` nêu tên khóa), `test_unknown_model_type_rejected`, `test_mixed_models_refuse_require_mma` (cấu hình `require_mma=true` với hai model khác nhau → `ValueError`), `test_worker_plan_preserves_model_keys`.
- [ ] **Step 2: Chạy** `python3 -m unittest tests.test_camera_service -v` → FAIL.
- [ ] **Step 3: Cài đặt**: tách kiểm tra khóa bắt buộc theo loại (`rfdetr`: `boxes_output`, `logits_output`, `background_index`; `yolo11`: `output_name`, `num_classes`, `conf_threshold`, `iou_threshold`, `max_det`); gộp khóa worker đè lên `session`; trong `run` tính hash model theo từng worker và truyền `model` riêng cho mỗi worker khi ghi file cấu hình riêng. `require_evidence` chỉ gọi khi tất cả worker dùng cùng một model, ngược lại ném `ValueError` nếu `require_mma` bật.
- [ ] **Step 4: Chạy lại** `python3 -m unittest tests.test_camera_service tests.test_topology -v` → PASS.

### Task 4: C++ `LetterboxPreprocessor` và `YoloDecoder`

**Files:**
- Create: `core/include/rfdetr/yolo.hpp`, `core/src/yolo.cpp`, `tests/test_yolo_core.py`
- Modify: `CMakeLists.txt` (thêm `core/src/yolo.cpp` vào `rfdetr_core`), `core/apps/adapter.cpp` (hai chế độ), `core/tests/core_tests.cpp`

**Interfaces:**
- Consumes: `ImageView`, `Detection`, `checked_product` từ `types.hpp`.
- Produces:
  - `struct LetterboxMeta { float scale; int pad_x; int pad_y; int source_height; int source_width; };`
  - `class LetterboxPreprocessor { LetterboxPreprocessor(int height, int width, std::uint8_t pad_value = 114); const std::vector<float>& run(ImageView image); const LetterboxMeta& meta() const noexcept; };` output RGB NCHW FP32 `/255`, đệm `pad_value/255`, cùng quy tắc làm tròn như Task 1.
  - `struct YoloConfig { std::size_t classes; float conf_threshold; float iou_threshold; std::size_t max_det; };`
  - `class YoloDecoder { explicit YoloDecoder(YoloConfig); const std::vector<Detection>& run(const float* output, std::size_t channels, std::size_t anchors, const LetterboxMeta& meta); };` `Detection::query` là chỉ số anchor; ném `std::invalid_argument` khi `channels != 4 + classes`, không hữu hạn, hoặc ngưỡng ngoài [0,1].
  - `adapter letterbox <image.bin> IH IW STRIDE H W PAD <tensor.bin> <meta.txt>` (meta dạng `scale,pad_x,pad_y`); `adapter yolo-decode <output.bin> CHANNELS ANCHORS IH IW SCALE PAD_X PAD_Y CLASSES CONF IOU MAXDET <out.csv>` (mỗi dòng `x1,y1,x2,y2,score,class_slot,anchor` như lệnh `decode`).

- [ ] **Step 1: Viết test thất bại.** Trong `core_tests.cpp` thêm hàm `yolo()` (gọi từ `main`) với: letterbox ảnh 1×1 vào 4×4 cho meta đúng; decode một anchor hợp lệ cho box và slot đúng; hai box cùng lớp chồng nhau chỉ còn một; khác lớp còn hai; `throws` cho `channels` sai, NaN, `conf_threshold>1`. Trong `tests/test_yolo_core.py` (mẫu `tests/test_core.py`, `skipUnless` có `build/rfdetr_adapter`) so parity với `preprocess_letterbox` và `decode_yolo` (Task 1) cho các cỡ ảnh `(1,1) (1,9) (11,1) (3,7) (23,19) (100,200) (200,100)`, target `(8,8) (19,2) (64,64)`, có stride đệm, và cho output ngẫu nhiên `C in {1,4}`, `N in {1,17,300}` có hòa điểm; dung sai `atol=3e-5` cho box, `1e-7` cho điểm, slot bằng nhau.
- [ ] **Step 2: Chạy** `bash scripts/build_core.sh portable build` → FAIL biên dịch (chưa có `yolo.hpp`).
- [ ] **Step 3: Cài đặt** hai lớp trong `yolo.cpp`: letterbox tái dùng công thức trục nửa-pixel của `Preprocessor::axis`, tái dùng buffer giữa các lần gọi; decoder giữ sẵn bộ đệm ứng viên để không cấp phát mỗi khung, NMS theo lớp bằng sắp xếp điểm giảm dần (hòa điểm index nhỏ) rồi loại theo IoU `>`. Thêm hai chế độ vào `adapter.cpp`.
- [ ] **Step 4: Chạy** `bash scripts/build_core.sh portable build` (ctest phải qua) rồi `python3 -m unittest tests.test_yolo_core tests.test_core -v` → PASS. Build với `POWER11_SANITIZE=ON` và chạy lại `core_tests` (`ASAN_OPTIONS=detect_leaks=0`) → PASS.

### Task 5: Chuẩn bị ORT C++ SDK và OpenCV 4 trên server (điều kiện cho Task 6)

**Files:** không sửa repo; tạo `/home/models/stack/onnxruntime-1.27.0/{include,lib}` và ghi `runs/setup/ort-sdk.txt`.

- [ ] **Step 1: Header ORT** cùng tag `v1.27.0`: `git clone --depth 1 --branch v1.27.0 --filter=blob:none --sparse https://github.com/microsoft/onnxruntime /home/models/stack/ort-src && cd /home/models/stack/ort-src && git sparse-checkout set include/onnxruntime/core/session`; sao chép các `onnxruntime_*.h` vào `.../onnxruntime-1.27.0/include/`.
- [ ] **Step 2: Thư viện**: sao chép `libonnxruntime.so.1.27.0` từ wheel (`.venv-cpu/.../onnxruntime/capi/`) vào `.../lib/`, tạo liên kết `libonnxruntime.so`; ghi SHA256 vào `runs/setup/ort-sdk.txt`. Kiểm phiên bản header và thư viện khớp bằng `OrtGetApiBase()->GetVersionString()` in `1.27.0` (Task 6 ghi vào `result.json`).
- [ ] **Step 3: OpenCV 4 C++** có `videoio` + FFmpeg: kiểm `pkg-config --modversion opencv4` và `dnf list installed | grep -i opencv`. Nếu thiếu, **hỏi người dùng** chọn một trong: nhờ admin cài `opencv-devel`, hoặc tạo env conda-forge riêng (`libopencv`), không đổi hệ thống. Wheel Python OpenCV 5 không dùng được.
- [ ] **Step 4: Kiểm**: `ONNXRUNTIME_ROOT=/home/models/stack/onnxruntime-1.27.0 OpenCV_DIR=<đường dẫn> bash scripts/build_core.sh camera build-camera` phải cấu hình thành công (trước khi sửa code Task 6, với mã hiện tại).

### Task 6: `OrtSession` và `camera_pipeline.cpp` chọn model theo worker

**Files:**
- Modify: `core/include/rfdetr/ort_session.hpp`, `core/src/ort_session.cpp`, `core/src/camera_pipeline.cpp`

**Interfaces:**
- Consumes: `LetterboxPreprocessor`, `YoloDecoder`, `YoloConfig`, `LetterboxMeta` (Task 4); khóa cấu hình worker từ Task 3.
- Produces: `OrtConfig` thêm `std::string model_type{"rfdetr"}` và `std::string output_name`; `struct YoloOutputView { const float* data; std::size_t channels; std::size_t anchors; };` và `YoloOutputView OrtSession::run_yolo(const float* nchw, std::size_t elements);` `run` chỉ dùng cho `rfdetr`, `run_yolo` chỉ cho `yolo11`; gọi sai loại ném `std::logic_error`. Worker JSON nhận `model_type` (mặc định `rfdetr`), các khóa YOLO như Task 3; `result.json` thêm `model_type`, `input_height`, `input_width`; `samples.csv` và `detections.jsonl` giữ nguyên cột/định dạng.

- [ ] **Step 1: Cài đặt hợp đồng YOLO trong `OrtSession::Impl`**: một input FP32 `NCHW`, một output FP32 3 chiều `[1, C, N]` với tên `output_name`; `C` phải bằng `4 + num_classes` (truyền qua `OrtConfig`), lệch thì ném `std::invalid_argument` nêu cả hai số. Nhánh `rfdetr` không đổi.
- [ ] **Step 2: Trong `camera_pipeline.cpp`** thay các biến `preprocessor/decoder/session` bằng một đối tượng cục bộ theo `model_type` với ba bước tách biệt để giữ ba cột thời gian: `preprocess(ImageView)`, `infer(const float*, std::size_t)`, `postprocess(int h, int w)`; warmup dùng khung đen cùng cách. Giữ nguyên quy tắc lỗi hiện có (không nuốt ngoại lệ).
- [ ] **Step 3: Kiểm cú pháp/biên dịch** với SDK ở Task 5: `bash scripts/build_core.sh camera build-camera` phải xong không cảnh báo mới (`-Wall -Wextra -Wpedantic`), `ctest` qua.
- [ ] **Step 4: Kiểm chạy trên server** với video thay thế nhỏ: một worker `rfdetr` (cấu hình cũ không có `model_type`, hồi quy) và một worker `yolo11`; `result.json` có `status: ok`, `model_type` đúng, `ort_version` là `1.27.0`. Thử `num_classes` sai → worker thoát mã 2 với thông báo khởi tạo.

### Task 7: Cấu hình mẫu và tài liệu

**Files:**
- Create: `configs/camera-core-rfdetr-traffic.json`, `configs/camera-core-yolo11-traffic.json`
- Modify: `docs/CORE.md`, `README.md`, `MANIFEST.sha256`

- [ ] **Step 1:** `camera-core-rfdetr-traffic.json`: model `/home/models/models/rfdetr_medium_traffic_576.onnx`, `input_name: input`, `height=width=576`, `boxes_output: dets`, `logits_output: labels`, `background_index: -1`, `top_k: 300`, `threshold: 0.25`, mean/std ImageNet. `camera-core-yolo11-traffic.json`: model `/home/models/models/yolo11m_traffic.onnx`, `input_name: images`, `height=width=640`, `model_type: yolo11`, `output_name: output0`, `num_classes: 4`, `conf_threshold: 0.25`, `iou_threshold: 0.7`, `max_det: 300`, `pad_value: 114`. Cả hai `cpus: []` (bắt buộc sinh bằng `worker_plan.py`).
- [ ] **Step 2:** `docs/CORE.md`: mục "Hai loại model" (quy ước letterbox, NMS, phạm vi thời gian, khác biệt input 576/640, kết quả đối chiếu Ultralytics ở Task 1 bước 5) và câu lệnh chạy hai loại worker; `README.md` trỏ tới mục đó.
- [ ] **Step 3: Kiểm** `python3 scripts/worker_plan.py` và `python3 scripts/camera_service.py --plan-only` trên server với hai cấu hình thành công; tạo lại `MANIFEST.sha256` bằng cùng quy tắc cũ (toàn bộ file tracked trừ chính nó và `.vscode`, sắp xếp theo thứ tự C) và `sha256sum -c MANIFEST.sha256` phải sạch.

### Task 8: Chiến dịch đo trên server (cần video giao thông)

**Files:** không sửa repo; kết quả vào `runs/`.

- [ ] **Step 1: Video replay.** Người dùng chọn video giao thông (máy phát triển có sẵn các file `.mp4` trong `Downloads`) rồi chuyển sang `/home/models/data/`; ghi `expected_frames`.
- [ ] **Step 2: Chạy sạch** từng model một, cùng mask CPU (một worker trên NUMA node 2, 1/4/7 luồng), ít nhất 3 lượt, xen kẽ thứ tự; rồi một lượt đồng thời hai loại worker trên hai mask không chồng nhau. Không ghim vào lõi mà dịch vụ vLLM đang dùng, không đổi cấu hình tiến trình khác.
- [ ] **Step 3: Báo cáo** bằng `scripts/compare_runs.py` và `scripts/report.py`; ghi rõ input 576 so với 640, NMS nằm trong hậu xử lý, và đây là thời gian capture-to-detection chưa phải sensor-to-result.
- [ ] **Step 4: Bằng chứng MMA cho từng model** (tách biệt): `perf record -g` trên worker thật, đọc symbol đứng đầu; chỉ gắn nhãn "MMA" khi thấy `MlasSgemmKernelPOWER10` trong hot path của model đó.
