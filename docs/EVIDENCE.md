# Chứng cứ người dùng và giới hạn truy xuất

Đã đọc trực tiếp bản gốc **Báo cáo YOLOv5-face trên IBM Spyre v1.docx** trong thư mục Downloads/Telegram Desktop; không lấy kết luận từ câu trả lời cũ của ChatGPT. SHA256 file gốc: `80341ff79d7f6f511eb27dd165768e1f2dc511354ea4450ba8dbcdc4cf430b31`. Tiêu đề trong file là “Báo cáo so sánh: Nhận diện khuôn mặt trên hệ thống hiện tại và server IBM Power 11 Spyre”, ngày 23-10-2026. Tài liệu không có ảnh nhúng trong `word/media`; bảng được đọc trực tiếp từ OOXML. Repo không phát tán lại tài liệu/weights/dataset của người dùng.

| Nội dung trong tài liệu | Diễn giải đúng cho nghiên cứu mới |
|---|---|
| Model `yolov5s-face`, 50 ảnh 640×640, batch 1, FP16 hai bên | Không phải RF-DETR; không phải phép đo CPU ORT FP32 |
| Một Spyre so một RTX 3060; máy chủ thử có 8 card | Số latency là một card; không nhân FPS lên tám lần |
| Model stage 10.1 ms Spyre / 12.7 ms NVIDIA | Khoảng 20% giảm latency stage; stage NVIDIA gộp cả decode output nên ranh giới chưa hoàn toàn giống nhau |
| Preprocess 5.5/2.4 ms; decode Spyre 1.6 ms; NMS 3.9/1.4 ms | Chi phí CPU và ranh giới stage giải thích E2E; không suy ra kernel CPU MMA chậm |
| Tổng bảng stage 21.2/16.5 ms; headline làm tròn 21 ms, 47/61 ảnh/s | Giữ cả số gốc và làm tròn; 61/47 ≈ 1.30, không gọi đó là 30% giảm latency |
| p99 27.7/28.0 ms trên bài đo 50 ảnh | Mẫu quá nhỏ để dùng p99 làm SLA camera dài hạn |
| WIDER FACE 3,226 ảnh: Easy 94.42/94.39, Medium 92.70/92.67, Hard 83.84/83.78 | Chỉ accuracy YOLO face trên split đó; COCO AP RF-DETR là phép đánh giá khác |
| PyTorch 2.7.1 + `torch_sendnn` ở Spyre; NVIDIA PyTorch 2.6.0 + CUDA 12.4 | Không có bằng chứng trong file rằng bên NVIDIA dùng TensorRT; không gắn nhãn TensorRT cho số lịch sử |
| Tách Detect để chạy một phần CPU, fixed 640×640, compile 8–15 giây | Đường hybrid riêng của thử nghiệm đó; không tự áp dụng sửa Detect/NMS cho RF-DETR |

RF-DETR detection baseline dùng sigmoid/top-k trên box transformer, không mặc định thêm NMS của YOLO. Chỉ thêm NMS nếu checkpoint/deployment contract yêu cầu và đánh giá accuracy lại.

Đã gọi `read_thread` cho cuộc trò chuyện `6affc77d-d2c0-83ec-b837-776dad19b0c6`, 10 turns; công cụ trả 5 turns, `hasMore=false`, `attachments=[]`. Các câu trả lời cũ chỉ là mã tham chiếu nội dung, không truy xuất được nội dung đầy đủ. **Chưa truy xuất được các ảnh/đầu ra cấu hình gốc Power11.** Vì thế RHEL 9.8, PowerVM, 240 logical CPU SMT8, 4 NUMA, RAM 954 GiB, `/home/models` 745G là thông tin do người dùng cung cấp trong yêu cầu, chưa được xác nhận bằng ảnh/lệnh gốc. Không bịa CPU mask, entitlement, driver hay số Spyre được gán cho LPAR. `inventory.py` là bước đầu tiên để xác nhận lại.

Ngoài ra tìm được ONNX cục bộ `rfdetr_small_7cls_static_640x640_new1665.onnx`, SHA256 `073b767a0caed9bcccd995d121ee3a39a46224fb977695f7badbf1c70bb03052`. Đọc thực tế: opset 17, input `input` float32 `[1,3,640,640]`, output `dets` `[1,300,4]`, `labels` `[1,300,8]`, không có metadata. Đã chạy checker và smoke inference CPU x86 trên tensor tổng hợp; không suy ra background/category mapping từ shape, không coi đây là accuracy test hay performance của Power11. Xem `VERIFICATION.md`.
