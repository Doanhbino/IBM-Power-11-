# Thiết kế: chọn model theo worker (RF-DETR / YOLO11) để đo performance giao thông

Trạng thái: **đã được người dùng duyệt (26/11/2026), chọn cách A**. Giá trị mẫu NMS/ngưỡng/đệm letterbox trong mục "Quyết định của người lập" vẫn là quyết định của người lập và có thể đổi.

## Mục đích và lý do
Đo và so sánh **hiệu năng suy luận** của hai model giao thông trên Power11: RF-DETR Medium và YOLO11 Medium (cùng 4 lớp, cùng thứ tự: `Car`, `Motorcycle`, `Primitive_Vehicle`, `person`). Mỗi worker camera chọn một trong hai model, chạy cùng pipeline, cùng cách đo. Lý do chọn model/quy mô camera cụ thể đang để ngỏ; người dùng chưa nêu.

## Mục tiêu
- Trường `model_type` (`rfdetr` hoặc `yolo11`) cho từng worker trong cấu hình camera.
- Một binary C++, một pipeline: capture → tiền xử lý → ORT → hậu xử lý. NMS của YOLO11 tính vào thời gian hậu xử lý.
- Bản Python tương ứng cho đo model-only (`benchmark.py`) dùng cùng quy ước.
- Mỗi worker ghi `model_type`, hash model và shape vào kết quả.
- Chạy tách biệt (cùng CPU mask, lần lượt) là cách so sánh sạch; chạy đồng thời hai loại worker trên mask không chồng nhau cũng được hỗ trợ (đo tải hỗn hợp).

## Ngoài phạm vi
- Training và export model. Export làm ở nơi khác rồi chuyển ONNX sang (đã chốt với người dùng).
- Đo độ chính xác mAP cho YOLO11 (có thể làm sau bằng `evaluate_coco.py`).
- Spyre, GPU, INT8/BF16, các phiên bản YOLO khác.

## Điều không được xảy ra
- Fallback âm thầm giữa hai model, hay lấy thời gian hậu xử lý của model này so với model kia khi phạm vi đo khác nhau.
- Đoán tên output/shape. Worker phải kiểm hợp đồng output của ONNX lúc khởi tạo và dừng nếu lệch.

## Ràng buộc
- ORT CPU FP32, batch 1, dùng MLAS (đã có bằng chứng kernel POWER10 MMA cho SGEMM trên MatMul tổng hợp). Bằng chứng MMA phải thu riêng cho từng model.
- Chưa có file ONNX nào; `imgsz`, tên input/output lấy từ ONNX sau khi export.
- `Primitive_Vehicle` không có trong COCO, nên chỉ dùng checkpoint đã train 4 lớp.

## Các cách đã cân nhắc
- **A (đề xuất):** `model_type` theo worker, thêm letterbox + giải mã YOLO + NMS vào C++. Một pipeline, công bằng, tốn công sửa core.
- **B:** export YOLO11 kèm NMS trong ONNX. Ít code, nhưng NMS lọt vào thời gian inference nên không tách được và khó so với RF-DETR.
- **C:** chỉ Python. Nhanh nhất nhưng không phản ánh queue camera/NUMA worker.
- Chọn: **A** (người dùng chưa xác nhận rõ; chờ duyệt).

## Thiết kế (cho cách A)
**Cấu hình.** Mỗi worker có `model_type` và đường dẫn `model` riêng. Khóa theo loại:
- `rfdetr`: giữ nguyên (`boxes_output`, `logits_output`, `background_index`, `mean`, `std`, `top_k`, `threshold`).
- `yolo11`: `output_name`, `conf_threshold`, `iou_threshold`, `max_det`, giá trị đệm letterbox (mặc định 114). Không có `mean/std` (chia 255).

**C++ (`core/`).**
- Tiền xử lý: giữ `Preprocessor` (resize kéo giãn + mean/std) cho RF-DETR; thêm `LetterboxPreprocessor` cho YOLO11 (giữ tỉ lệ, đệm, chia 255) và trả thêm tỉ lệ/đệm để ánh xạ box về khung gốc.
- Giải mã: giữ `Decoder`; thêm `YoloDecoder` đọc output `[1, 4+C, N]` (box xywh theo pixel, điểm lớp đã qua sigmoid trong model), lọc theo `conf_threshold`, NMS theo từng lớp, tối đa `max_det`, rồi ánh xạ về khung gốc.
- `OrtSession`: hợp đồng output theo `model_type`. RF-DETR: hai output như hiện tại. YOLO11: một output 3 chiều, kiểm `4+C` với số lớp khai báo.
- Pipeline camera: chọn bộ tiền xử lý/giải mã theo `model_type`, các cột `samples.csv` giữ nguyên để kết quả so sánh được.

**Python.** `common.py` thêm `preprocess_letterbox` và `decode_yolo` (tham chiếu NumPy, cũng dùng làm oracle cho test C++). `benchmark.py`, `camera_service.py`, `worker_plan.py` nhận `model_type`, hash model theo worker.

**Thứ tự thực hiện.** (1) Python + test trên fixture, kiểm với ONNX thật khi có. (2) C++ + test parity với NumPy. (3) Config mẫu `camera-yolo11.json` và tài liệu.

**Kiểm thử.** Fixture output YOLO tổng hợp; NMS đối chiếu với tham chiếu NumPy (tie, IoU biên, đa lớp); letterbox và ánh xạ ngược khớp ở nhiều cỡ ảnh; parity NumPy–C++ như `test_adapter.py`.

## Quyết định của người lập (đánh dấu để người dùng sửa)
- NMS theo từng lớp (không class-agnostic); ngưỡng mẫu `conf=0.25`, `iou=0.7`. Giá trị phải đặt rõ trong cấu hình, không có mặc định ẩn. Cần xác nhận với cài đặt lúc đánh giá model.
- Giá trị đệm letterbox 114 theo mặc định của Ultralytics; phải khớp với cách model được train.
- Phần đo thời gian phân tách sẵn capture / preprocess / inference / postprocess như hiện tại.

## Để lại cho người xây
- Tên trường chính xác trong JSON, cách đặt tên file mẫu, và việc dùng thuật toán NMS nào miễn cho ra cùng kết quả với tham chiếu.
