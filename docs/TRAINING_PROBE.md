# Đo training theo batch, epoch và operator

`training_probe.py` là nhánh instrumentation tùy chọn, chưa chạy RF-DETR thật trong phiên bàn giao. Chạy training thường trước để xác nhận dependency/forward/backward. Probe giữ vòng training của RF-DETR, thêm callback bằng `extra_callbacks` qua factory `rfdetr.training.build_trainer` đã đối chiếu [API customization](https://rfdetr.roboflow.com/latest/learn/train/customization/). Nó dùng wrapper tạm trong tiến trình, không sửa package trên đĩa và không thay các callback mặc định. API này cần pin và xác minh lại nếu đổi RF-DETR release; script dừng nếu factory không có tham số hoặc không thấy batch.

Sao chép config CPU smoke, đổi output_dir sang thư mục mới, dùng subset train nhỏ có validation/test hợp lệ. Đo training thường trong một run riêng để lấy thời gian tự nhiên. Probe chỉ hỗ trợ single-process training; không lấy số này làm distributed scaling.

```bash
python scripts/training_probe.py --config configs/train-cpu-smoke.json --profile
# Trên máy RTX, output_dir của config phải là thư mục mới:
NVIDIA_TF32_OVERRIDE=0 python scripts/training_probe.py --config configs/train-rtx3060-fp32.json --sync-cuda --profile
```

Output ở `train.output_dir/probe`: `batches.csv`, `epochs.csv`, `summary.json`; profile tùy chọn tạo `trace-step-N.json`, `operators.txt`. Mặc định bỏ 1 batch, warmup profiler 1 batch, capture 3 batch; chỉnh `--wait/--warmup/--active`. Phải chạy đủ batch để trace hoàn chỉnh. Operator profile chạy riêng, không dùng latency có instrumentation làm KPI.

Timer batch bắt đầu ở callback sau transfer dữ liệu: gồm forward/backward và optimizer khi accumulation đủ, không gồm data loader wait/transfer. `global_optimizer_step` giúp phân biệt microbatch và update step. Timer epoch là khoảng callback của PTL, có thể gồm validation/checkpoint theo thứ tự loop; số images/s được đặt tên theo phạm vi đó. Không coi đây là forward/backward thuần hoặc throughput tối đa phần cứng. Nếu không đọc được batch size, images/FPS để null, không tự suy từ config.

`--sync-cuda` đồng bộ completion tại hai biên batch để host timer phản ánh công việc GPU đã xong, nhưng làm thay đổi overlap/tốc độ. Khi không bật, host time có thể bất đồng bộ; CUDA trace hữu ích hơn cho operator. Các timer này phục vụ chẩn đoán, thời gian train wall của run tự nhiên vẫn là số chính để so time-to-accuracy. Profile ngắn có thể còn chi phí warmup và validation; kiểm trace trước khi diễn giải.
