# Parity số học và accuracy (phía inference)

Repo này không train hay export model. ONNX và reference đến từ nơi khác; tài liệu này chỉ mô tả cách kiểm model đã có.

## Parity ONNX so với PyTorch eager

`scripts/validate_onnx.py` chạy `onnx.checker` đầy đủ rồi so output ORT (CPU 1 thread, hoặc backend khác qua `--backend-config`) với reference. Reference là NPZ do PyTorch eager FP32 tạo **độc lập ở nơi khác**; không bao giờ lấy từ ORT. NPZ chứa các khóa `input_{i}` (tensor NCHW FP32 đã preprocess), `dets_{i}` và `labels_{i}` cho mỗi case `i`.

```bash
python scripts/validate_onnx.py --model models/rfdetr.onnx \
  --reference <eager_reference.npz> --output runs/parity.json
```

Mặc định `atol=1e-4`, `rtol=1e-3`: đây là ngưỡng khởi đầu, không là cam kết của mọi model. NaN, Inf, shape sai hoặc lệch vượt ngưỡng đều thoát với mã 2. Một input ngẫu nhiên chỉ là smoke; nghiệm thu cần ít nhất 100 ảnh thực đã qua đúng pipeline preprocess. Chạy cùng NPZ trên Power và trên máy đối chứng, báo max absolute error.

Nếu model dùng external ONNX data, chuyển toàn bộ file cùng thư mục và ghi hash từng file; hash `.onnx` riêng chưa chứng thực tensor external. Kiểm graph nhận float32 NCHW, output đúng tên và không chứa custom domain thiếu kernel.

## Đo COCO AP với decoder dùng chung

`evaluate_coco.py` dùng cùng `common.py` với benchmark. `background_index` và `category_ids` phải phản ánh checkpoint thật: raw logit slot ánh xạ sang COCO category ID, không mặc định cộng 1. Với sparse COCO-pretrained head, không xóa slot cuối nếu đó là foreground. Decoder sigmoid/top-K cặp query-class, không softmax; giữ đúng chính sách NMS của mô hình. Preprocessing phải so với eager trên ảnh thật trước khi coi accuracy đã nghiệm thu.

```bash
python scripts/evaluate_coco.py \
  --config configs/accuracy-custom-example.json \
  --annotations /home/models/datasets/custom-coco/test/_annotations.coco.json \
  --images /home/models/datasets/custom-coco/test \
  --output-dir /home/models/runs/accuracy-cpu
# Đánh giá file predictions COCO từ eager/TensorRT qua cùng evaluator:
python scripts/evaluate_coco.py \
  --annotations /home/models/datasets/custom-coco/test/_annotations.coco.json \
  --predictions /home/models/runs/reference/predictions.coco.json \
  --output-dir /home/models/runs/accuracy-reference
```

Sửa `configs/accuracy-custom-example.json`: ví dụ một lớp đặt logit slot `0` thành COCO category `1`; thay mapping theo dataset thực. Dùng cùng thông số input/output với bản benchmark. Script override confidence về `accuracy_threshold` (mặc định 0.001) và `accuracy_top_k` (mặc định 300) để không cắt cụt đường precision-recall; COCOeval vẫn dùng maxDets 1/10/100 chuẩn và ghi trong report. Ghi AP@[.50:.95], AP50, AP75, AP theo kích thước và AR. So AP theo điểm phần trăm: giảm 0.005 đơn vị 0–1 là 0.5 điểm AP. Giữ cùng image IDs, annotations, maxDets và mapping giữa các runtime. Camera thực cần thêm test theo khoảng cách, ánh sáng, che khuất, vật nhỏ và density; AP tổng không thay thế recall ở ngưỡng vận hành.

`pycocotools` là dependency của bước này; trên ppc64le không có wheel PyPI, cần build từ sdist và ghi nguồn gốc "tự build".
