# Training, export và accuracy

Đây là workflow nghiên cứu; các con số hiệu năng phải đo tại máy. Việc RF-DETR có tùy chọn CPU không xác nhận mọi dependency có bản ppc64le. Chỉ bắt đầu sau khi import torch, torchvision, rfdetr và toán tử cần dùng vượt qua smoke test trong môi trường riêng. Đường đối chứng GPU giúp tạo checkpoint/export khi Power11 chưa dựng được training stack; vẫn có thể benchmark ONNX CPU trên Power11 độc lập.

## Chốt thí nghiệm trước khi chạy

Wrapper hỗ trợ trực tiếp YOLO detection với `data.yaml`/`data.yml` và nhãn `.txt`, hoặc COCO kiểu Roboflow với `_annotations.coco.json`. Dùng phiên bản RF-DETR có YOLO loader; khả năng train thực phải qua smoke test trong môi trường server.

Với dataset YOLO giao thông, sửa `dataset_dir` trong `configs/train-traffic-smoke.json` và `configs/train-traffic-cpu.json` thành thư mục gốc chứa YAML. Hai config đặt `dataset_file="yolo"`, Small 512, FP32 và `run_test=false`. Test split chưa cần cho giai đoạn fine-tune. Cấu trúc mẫu:

```text
traffic-yolo/
  data.yaml
  train/images/*.jpg
  train/labels/*.txt
  valid/images/*.jpg
  valid/labels/*.txt
```

YAML khai báo `train: train/images`, `val: valid/images`, `nc` và `names` đúng thứ tự class ID trong nhãn. Các layout `images/train` + `labels/train`, `images/val` + `labels/val` cũng được hỗ trợ khi YAML trỏ đúng đường dẫn. Wrapper hiện nhận split dạng thư mục, chưa nhận danh sách ảnh `.txt` hoặc list nhiều đường dẫn. Đường dẫn phải nằm trong `dataset_dir`.

Mỗi dòng nhãn detection gồm `class_id cx cy width height`; class ID liên tục từ 0, tọa độ chuẩn hóa [0,1], width/height dương. Mỗi ảnh cần nhãn cùng tên, khác phần mở rộng. Quy tắc preflight của harness yêu cầu ảnh negative có `.txt` rỗng để phân biệt với file nhãn thất lạc. Polygon/pose bị từ chối bởi wrapper detection này.

```bash
python scripts/train.py --config configs/train-traffic-smoke.json --check-only
python scripts/train.py --config configs/train-traffic-smoke.json
# Sau khi 1 epoch hoàn thành và đã xem thời gian/loss/validation:
python scripts/train.py --config configs/train-traffic-cpu.json
```

YOLO preflight cần PyYAML, không import torch/RF-DETR khi `--check-only`. Nó kiểm tên lớp, đường dẫn, nhãn, box hữu hạn và việc dùng chung file ảnh giữa các split; chưa decode ảnh hoặc tìm ảnh trùng nội dung dưới tên khác. Manifest training lưu class mapping, thống kê các split đã kiểm và SHA256 của YAML/các nhãn sử dụng. Các script `evaluate_coco.py`/`predict_eager.py` vẫn nhận COCO annotations; dùng evaluator của RF-DETR trong training hoặc chuẩn bị COCO annotations cho bước đối chiếu độc lập sau này.

Nguồn: [RF-DETR YOLO dataset format](https://rfdetr.roboflow.com/latest/learn/train/dataset-formats/#yolo-format).

Chọn Nano 384×384 để mở đường; model thực tế khác phải dùng đúng class, resolution, checkpoint, số lớp, mean/std, mapping nhãn và kiến trúc đã train. Không hạ resolution riêng cho Power rồi so trực tiếp với GPU. Thư mục COCO:

```text
/home/models/datasets/custom-coco/
  train/_annotations.coco.json + ảnh train
  valid/_annotations.coco.json + ảnh validation
  test/_annotations.coco.json  + ảnh test
```

Chốt một manifest ảnh bằng SHA-256; kiểm tra ảnh trùng giữa ba split, nguồn camera và khoảng thời gian tách biệt. Giữ test kín; lựa chọn tham số bằng valid. Dùng cùng checkpoint khởi tạo và tập dữ liệu trên hai máy. Seed 42 cho smoke; nghiên cứu chính lặp 3 seed (42, 43, 44), đổi output_dir mỗi lượt, báo mean/std của AP và thời gian.

Các JSON `expected_versions` đang để trống vì chưa có stack được kiểm chứng trên server. Sau smoke thành công, điền chính xác các phiên bản vào object này và lưu thêm pip-freeze/commit nguồn/build flags. Script kiểm tra mọi phiên bản đã khai báo trước khi chạy; đây chưa phải lockfile đã nghiệm thu. Không cài thêm dependency trực tiếp vào môi trường đang dùng cho benchmark.

## Fine-tune FP32 có đối chứng

Trong môi trường Python riêng đã chuẩn bị theo README, từ gốc repo:

```bash
python scripts/train.py --config configs/train-cpu-smoke.json --check-only
python scripts/train.py --config configs/train-cpu-smoke.json
python scripts/train.py --config configs/train-cpu.json
# Trên x86_64 + RTX 3060, trong môi trường CUDA riêng:
NVIDIA_TF32_OVERRIDE=0 python scripts/train.py --config configs/train-rtx3060-fp32.json
```

Sửa tất cả đường dẫn và class trước. Check-only chỉ kiểm tra file/config; không kiểm tra khả năng chạy toán tử. CPU bắt đầu 1 epoch, batch 1, accumulation 4, FP32. Đối chứng training dùng cùng microbatch và accumulation để giảm sai khác; bài đo tận dụng tối đa từng phần cứng là thí nghiệm riêng. Tăng torch_threads/num_workers theo CPU affinity thực tế; không dùng ngay 240 luồng. RAM lớn không thay thế băng thông hay năng lực compute.

`amp_dtype=null` khóa FP32; CUDA TF32 được tắt để làm đối chứng FP32. Seed không bảo đảm bitwise đồng nhất giữa kiến trúc. `deterministic_algorithms=true` là một thí nghiệm riêng có thể dừng nếu toán tử thiếu implementation xác định. BF16/AMP chỉ mở sau khi có đường FP32 và phải báo accuracy, tốc độ và precision thực thi thực tế.

`baseline_manifest.json` ghi hash weights/annotation, phiên bản, CPU affinity, thời gian dựng model, train wall và RSS tối đa. Train wall bao gồm data loader, train, validation và lưu checkpoint; không được gọi nó là forward/backward thuần. Giữ log upstream để phân tích loss, AP theo epoch và thời gian ổn định; bỏ epoch đầu khi ước tính tốc độ steady-state nhưng vẫn báo tổng thời gian đến AP mục tiêu. Để phân rã train-step sâu hơn, dùng torch.profiler trên đoạn ngắn trong training stack đã pin, đánh dấu forward/backward/optimizer/data-wait và không dùng lượt profile làm lượt báo tốc độ.

Checkpoint best `.pth` có thể chỉ chứa weights/trạng thái rút gọn; full `.ckpt` mới thích hợp tiếp tục optimizer/scheduler. Đọc log upstream khi resume. Chọn checkpoint theo valid trên cùng quy tắc; chạy accuracy test sau export và sau từng thay đổi precision/runtime.

## Export và parity số học

RF-DETR hiện tài liệu hóa ONNX opset 17 và shape cố định hoặc dynamic batch. Baseline batch 1 cố định giảm biến số. Sửa `configs/export-nano.json` tới checkpoint đã chọn. Với checkpoint cũ thiếu metadata, dùng đúng class + kwargs và `load_from_checkpoint=false`; không để tự suy đoán kiến trúc.

```bash
python scripts/export_model.py --config configs/export-nano.json
python scripts/validate_onnx.py \
  --model /home/models/runs/export-nano-fp32/inference_model.onnx \
  --reference /home/models/runs/export-nano-fp32/eager_reference.npz \
  --output /home/models/runs/export-nano-fp32/validation.json
```

Exporter tạo tensor FP32 chuẩn hóa, chạy eager trước khi gọi export và lưu `eager_reference.npz`; validator chạy ONNX checker rồi so cùng input với CPU ORT. Eager adapter dùng `model.model.model` và `pred_boxes`/`pred_logits`, là API nội bộ nên dừng khi schema đổi. Không dùng output ORT làm reference cho chính ORT. Default một input ngẫu nhiên chỉ là smoke; acceptance cần `reference_inputs` trỏ file `.npy` float32 NCHW gồm ít nhất 100 ảnh thực đã xử lý đúng pipeline, chia hết cho batch. Chạy cả CPU Power và GPU-host CPU ORT với cùng NPZ. Báo max absolute error; mặc định atol=1e-4, rtol=1e-3 là ngưỡng ban đầu, không là cam kết của mọi model. Bất kỳ NaN, Inf, shape sai hoặc lệch vượt ngưỡng đều exit 2.

Nếu model dùng external ONNX data, chuyển toàn bộ file cùng thư mục và ghi hash từng file; hash .onnx riêng chưa chứng thực tensor external. Luôn kiểm tra graph có nhận float32 NCHW, output đúng tên và graph không chứa custom domain thiếu kernel.

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

Sửa `configs/accuracy-custom-example.json`: ví dụ một lớp đặt logit slot `0` thành COCO category `1`; thay mapping theo dataset thực. Dùng cùng thông số input/output với bản benchmark. Script override confidence về `accuracy_threshold` (mặc định 0.001), `accuracy_top_k` (mặc định 300) để không cắt cụt đường precision-recall; COCOeval vẫn dùng maxDets 1/10/100 chuẩn và ghi trong report. Ghi AP@[.50:.95], AP50, AP75, AP theo kích thước và AR. So AP theo điểm phần trăm: giảm 0.005 đơn vị 0–1 là 0.5 điểm AP. Giữ cùng image IDs, annotations, maxDets và mapping giữa các runtime. Camera thực cần thêm test theo khoảng cách, ánh sáng, che khuất, vật nhỏ và density; AP tổng không thay thế recall ở ngưỡng vận hành.

## Nguồn API chính thức đã đọc ngày 25-11-2026

- [Training parameters](https://rfdetr.roboflow.com/latest/learn/train/training-parameters/): device, seed, batch/accumulation, FP32, checkpoint/resume. Tài liệu này không xác nhận binary ppc64le.
- [Advanced export](https://rfdetr.roboflow.com/latest/exports/advanced/): tham số shape/batch/opset/format.
- [ONNX](https://rfdetr.roboflow.com/latest/exports/onnx/): output `dets`/`labels`, normalization, background mapping và postprocessing.
- [RF-DETR detector implementation](https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/detr.py) và [LWDETR](https://github.com/roboflow/rf-detr/blob/develop/src/rfdetr/models/lwdetr.py): đối chiếu adapter eager và from_checkpoint. `develop` là tham chiếu đang di chuyển; lưu commit thực tế dùng trong thí nghiệm.

## Sinh predictions eager để so accuracy độc lập

```bash
python scripts/predict_eager.py --model-config configs/export-nano.json --decode-config configs/accuracy-custom-example.json --annotations /home/models/datasets/custom-coco/test/_annotations.coco.json --images /home/models/datasets/custom-coco/test --output-dir runs/eager-reference
python scripts/evaluate_coco.py --annotations /home/models/datasets/custom-coco/test/_annotations.coco.json --predictions runs/eager-reference/predictions.coco.json --output-dir runs/accuracy-eager
```

Script dùng module eager `model.model.model`, cùng adapter fail-fast như export; preprocess/decoder được viết độc lập bằng PyTorch FP32 (interpolate, sigmoid, sort), không gọi lại NumPy decoder của ONNX. Nó chưa được chạy trong phiên bàn giao vì chưa cài stack training. Xác nhận resolution, background và category mapping theo checkpoint. Nội bộ RF-DETR có thể đổi giữa release; phải pin/version và smoke adapter trước. Dùng cùng `--image-ids` cho eager và ONNX khi đo subset.

## Khóa TF32 cho đối chứng CUDA FP32

Phải đặt `NVIDIA_TF32_OVERRIDE=0` ngay khi khởi chạy tiến trình như lệnh ở trên; script từ chối chạy CUDA baseline nếu thiếu. RF-DETR hiện có thể đặt lại `torch.set_float32_matmul_precision("high")` trong import/trainer, nên chỉ đặt `allow_tf32=False` trước khi gọi `model.train` chưa đủ. Biến môi trường này chỉ áp dụng cho tiến trình được khởi chạy, không thay cấu hình hệ thống. Lưu giá trị vào manifest và kiểm runtime/profiler trên GPU. Xem [PyTorch CUDA environment](https://docs.pytorch.org/docs/main/cuda_environment_variables.html), [NVIDIA global control](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/).
