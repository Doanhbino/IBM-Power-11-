# Thiết kế nghiên cứu, nghiệm thu và cải tiến

## Các câu hỏi và phép đối chứng

Đường A đo fine-tune CPU Power11 để xác định tính khả thi, thời gian đến accuracy mục tiêu và tài nguyên; đường B fine-tune RTX 3060 cùng model/init weights/split/seed/FP32, cùng microbatch và accumulation. Thực nghiệm tối đa throughput riêng cho mỗi máy được báo riêng vì thay batch ảnh hưởng trajectory tối ưu. Dùng seed 42/43/44 cho kết luận hội tụ, không kết luận từ một seed. Chọn model theo checkpoint và use case đang dùng; Nano 384 chỉ là bài smoke nhẹ, không thay thế RF-DETR Small 640 đang có.

Đường C xuất một checkpoint đã chọn, kiểm FP32 eager→ONNX, dùng chính ONNX đó trên Power CPU/MLAS và máy đối chứng. Việc train trên NVIDIA rồi deploy Power là workflow production hợp lệ, nhưng không phải benchmark training Power. Ghi thời gian tải model, compile/engine build, first-call, warmup, steady-state và tổng time-to-accuracy tách biệt.

| Thí nghiệm | Biến thay | Giữ cố định | Đầu ra |
|---|---|---|---|
| CPU single worker | threads 1/2/4/8/… theo core/cpuset | shape/batch/input/precision | p50/p95/p99, FPS, RSS, latency stage |
| Affinity/NUMA | node-local/all/spread; sibling limit 1/2/4/8 | thread budget, entitlement ổn định | CPU mask, memory bind, throughput |
| Nhiều worker | 1/2/4/… tới giới hạn CPU/RAM | tổng ngân sách CPU đã nêu, không overlap mask | tổng frame / common makespan; RSS mỗi worker |
| Camera replay | codec, resolution, số camera | video hash/frame IDs/preprocess/decoder | read/decode + preprocess + inference + postprocess + total |
| Live camera | số stream, input FPS, load | cùng camera/network/time window | output FPS, lỗi, backlog/age nếu instrument được |
| RTX TensorRT | FP32 no-TF32 rồi FP16 riêng | cùng checkpoint/data/shape | device compute, transfer, host wall, accuracy |

Tách warm/cold cache, serial latency và saturated throughput. Batch 1 là latency baseline; throughput batch >1 là chiến dịch riêng với export đúng batch. Không tự tăng batch từ shape tĩnh. Chạy ≥3 repeat/case, warmup 30–100 sau first-call, ≥1,000 measured calls ban đầu; chọn thêm duration/iterations để có ≥10,000 mẫu cho tail latency. Với CPU chậm, cần dự toán thời gian trước khi mở sweep rộng. Báo số mẫu và biến thiên giữa repeat; p99 là thống kê thực nghiệm, không phải chứng minh SLA.

`benchmark.py` mode model replay cùng một tensor để cô lập compute. Dùng danh sách ảnh thật và nhiều lượt tensor nếu cần phản ánh data-dependent workload. Camera mode hiện sequential đọc backend, đo từ trước `read()` đến detection; có local frame index/PTS, hash replay, không biết sensor timestamp, số frame camera drop hay độ sâu queue backend. Replay không throttle theo FPS video: đây là khả năng xử lý offline. Để nghiệm thu live realtime cần timestamp nguồn/RTCP đồng bộ và queue bounded/latest-frame/drop counter; chưa coi camera mode hiện tại là service multi-camera production.

## Cổng nghiệm thu đề xuất

Các ngưỡng dưới đây là chính sách nghiên cứu đề xuất, cần chốt theo bài toán; không phải hiệu năng đã đạt.

| Cổng | Điều kiện qua |
|---|---|
| Nguồn chứng cứ | Inventory xác nhận OS/arch/NUMA/cpuset/entitlement; Spyre status = xác nhận hoặc unknown, không suy từ tên server |
| Môi trường | Wheel/source provenance + version/hash; pip check; torch/torchvision operator smoke; không fallback CPU khi yêu cầu CUDA |
| Training | CPU smoke 1 epoch có checkpoint, loss hữu hạn, valid eval; đối chứng cùng cấu hình chạy được; wall/RSS/log lưu đầy đủ |
| Export | ONNX checker; ≥100 ảnh đại diện same-input eager/ORT, finite/shape; FP32 atol 1e-4, rtol 1e-3 khởi đầu, điều tra thay vì nới ngưỡng tùy tiện |
| Preprocess/decoder | Tensor normalized đối chiếu eager trên ảnh portrait/landscape, up/downscale; cùng background/category mapping, threshold/top-k; không chỉ so shape |
| Accuracy | Bản FP32 giảm AP@[.50:.95] ≤0.001 (0.1 điểm AP); bản FP16/INT8 có budget riêng gợi ý ≤0.005; recall lớp quan trọng không giảm quá ngưỡng nghiệp vụ |
| Benchmark | ≥3 repeat; unprofiled; sample count và ranh giới timer rõ; profile/provider/MMA claim có bằng chứng; không mix số synthetic với data thật |
| Camera SLA | Chốt số camera C, input FPS F, p95 latency L và tuổi frame A; soak ≥1 giờ, không reconnect/lỗi âm thầm; chưa chốt C/F/L/A thì không kết luận đạt realtime |
| Reproducibility | Config + hardware + data/model/wheel hash + raw CSV/JSON + telemetry + logs; rerun có sai lệch median ≤5% hoặc giải thích drift/PowerVM sharing |

`accuracy_gate.py` kiểm AP từ hai báo cáo; không tự chứng minh category mapping hoặc phép đo là công bằng. WIDER FACE cần evaluator/split chính thức nếu bài toán face; COCO script không tạo Easy/Medium/Hard WIDER score. Model detection 7 class không dùng landmark/segmentation adapter.

## Lộ trình C++ và MMA dựa trên profile

1. **B0: đường FP32 đúng.** Chạy ORT CPU, một NUMA, threads nhỏ. Chốt accuracy, lưu ORT revision/compiler và hotspot operator. Với PowerVM kiểm physc/%entc/steal và cgroup quota; host rảnh chưa chắc LPAR có capacity tương ứng 240 logical CPU.
2. **B1: tối ưu layout.** Sweep thread/sibling/NUMA và nhiều process. Khởi tạo model bên trong affinity/memory bind để first-touch đúng node. So riêng latency tối thiểu và tổng throughput tối đa. Xác minh operator MatMul/Gemm/Conv cùng transpose/softmax/layernorm/resize để biết compute hay bandwidth/scheduling chiếm ưu thế.
3. **B2: C++ pipeline dùng ORT C++ API.** Giữ shared library cùng build/MLAS. Chuyển decoder/video/preprocess có profile cao sang C++; buffer tái sử dụng, resize half-pixel cùng RGB/mean/std, không allocation mỗi frame; pin pool tránh oversubscription. So tensor sai số và cùng COCO output trước khi đo. Xuất timer stage đúng ranh giới B0. Đưa queue bounded/drop metrics/PTS vào service camera, không chỉ viết vòng `Run()` nhanh hơn.
4. **B3: chứng minh MMA hot path.** Xem build log có Power10/MMA probes và kernel thực sự compile. `perf list` trên máy trước khi chọn event; `perf record`/stack nếu có quyền, symbol/disassembly để phân biệt kernel MLAS Power10 với fallback. CPU EP trong JSON chưa chứng minh MMA. So build có/không đường tối ưu trong checkout riêng với cùng graph/input/thread; không dùng phép so OpenBLAS vs MMA như hai thiết bị.
5. **B4: chỉ viết kernel riêng khi có lợi.** Đo microbenchmark đúng M/N/K/batch/dtype/stride RF-DETR phát ra; tính packing/copy/dispatch/tails/alignment/NUMA. Tận dụng MLAS/BLAS đã tối ưu trước. Custom ORT op hoặc contribution MLAS cần test shapes nhỏ, odd tails, NaN/Inf, multithread và numerical accumulation. ISA dispatch phải có fallback; không build Power-only binary cho host khác. FP16/BF16/INT8 không tự nhanh trên CPU chỉ vì model từng chạy FP16 Spyre.
6. **B5: nhánh Spyre.** Xác nhận card được gán cho LPAR, driver/container/runtime và operator/model compatibility với IBM. Prototype RF-DETR trên stack được hỗ trợ trước; ghi compile time, partition graph, CPU fallback, transfers và E2E. Không reuse API `torch_sendnn` lịch sử nếu chưa có hướng dẫn cho phiên bản hiện dùng.

Ưu tiên theo Amdahl: nếu infer chiếm 80% wall và tối ưu CPU preprocess chiếm 10% được 2×, tổng chỉ nhanh khoảng 1/(0.9+0.1/2)=1.053×. Chọn việc theo tỷ trọng đã đo; RAM lớn/240 logical CPU không bảo đảm scaling tuyến tính.
