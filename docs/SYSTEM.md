# Kiểm kê LPAR, môi trường và sweep

Chạy từ thư mục gốc repo bằng tài khoản thường. Không có script nào đổi SMT,
firmware, driver, entitlement, mount, gói RHEL hoặc Python hệ thống. Các script
kiểm kê/telemetry chỉ cần thư viện chuẩn Python 3 trên Linux; inference cần môi
trường đã qua cổng hỗ trợ trong `SUPPORT.md`.

## 1. Chốt cấu hình thực thấy trong LPAR

```bash
python3 scripts/inventory.py --output runs/preflight/inventory.json
```

File ghi phiên bản OS/kernel, affinity được cấp cho tiến trình, CPU→core→sibling→NUMA,
`lparstat -i`, `/proc/ppc64/lparcfg`, `lscpu -e`, `numactl --hardware`, PCI/driver,
gói RPM và dung lượng đĩa theo byte. Lệnh thiếu/quyền đọc thiếu được ghi rõ;
script không tự cài thêm. Lưu bản inventory trước và sau chiến dịch đo.

240 logical CPU ở SMT8 chỉ tương ứng 30 core nếu toàn bộ core/sibling đều được
LPAR cấp và online. Entitlement của shared LPAR, uncapped/capped và độ tranh chấp
processor pool là các biến riêng. Không dùng 240 làm số core vật lý. Đọc
`partition_entitled_capacity`, output `lparstat -i`, giới hạn cpuset/cgroup và
affinity; nếu thiếu thì ghi “chưa xác minh”. Đo lại khi LPAR thay đổi cấu hình.

Kiểm tra 4 NUMA bằng output thực tế; không suy CPU node 0 là `0-59`.
Mẫu cấu hình người dùng là RHEL 9.8, ppc64le, RAM 954 GiB và `/home/models` 745G;
inventory mới là chứng cứ tại thời điểm đo. Chừa dung lượng cho checkpoint,
dataset, wheelhouse và log, giữ ít nhất 15–20% trống là mục tiêu vận hành đề xuất.

## 2. Spyre PCIe và MMA CPU

MMA là khả năng tính ma trận trong CPU, không xuất hiện như card Spyre trong
`lspci`. PCIe có thiết bị IBM cũng chưa đủ để kết luận đó là Spyre hoặc LPAR có
quyền dùng accelerator. Đối chiếu tên/ID thiết bị với hồ sơ IBM được quản trị viên
cung cấp, assignment vào LPAR, driver đã bind, runtime được IBM hỗ trợ và health
check của đúng phiên bản runtime. Repo chỉ thu thập chứng cứ, không bịa PCI ID,
command driver hay một ONNX Runtime Spyre provider. Mọi kết quả CPU ở đây dùng
`CPUExecutionProvider`; không ghi “Spyre FPS”.

Không suy “đã dùng MMA” từ tên POWER11, `-mcpu`, OpenBLAS được cài, hoặc FPS tăng.
Muốn gán tốc độ cho MMA cần chốt ORT build/commit, đường kernel thực chạy, profile
operator, disassembly/counter phù hợp và thử A/B cùng graph/threads/precision.

## 3. Cài tách biệt, dừng nếu không có wheel đúng nền tảng

Admin cung cấp một Python được phép dùng có `venv`/`pip`; không sửa Python RHEL.
Tạo manifest **đúng phiên bản và SHA256 của mọi dependency bắc cầu** sau khi xác
minh nguồn phân phối và các wheel ppc64le. Repo không phát hành một file lock
ppc64le giả khi chưa kiểm tra được các binary trên server. `requirements*.in`
(nếu có) là danh sách nhu cầu, không phải lock để tái lập.

```bash
PYTHON_BIN=/duong-dan/python-da-duyet REQ_FILE=/duong-dan/cpu.lock WHEEL_DIR="$PWD/wheelhouse" bash scripts/setup_env.sh probe
PYTHON_BIN=/duong-dan/python-da-duyet REQ_FILE=/duong-dan/cpu.lock WHEEL_DIR="$PWD/wheelhouse" VENV_DIR="$PWD/.venv-cpu" bash scripts/setup_env.sh install
```

Thay đường dẫn mẫu trước khi chạy. `probe` dùng index hiện cấu hình của pip,
không đổi index, chỉ tải wheel tương thích máy đang chạy; chạy nó **trên Power11**
để kiểm tra tag ppc64le/Python/ABI. Lưu `wheelhouse/pip-platform.txt`. Không chạy
probe trên x86 rồi coi là bằng chứng ppc64le. `--require-hashes` chặn dependency
không pin/hash; `--only-binary=:all:` chặn fallback build source âm thầm.
`install` chạy offline trong venv mới và `pip check`. Nếu cài lỗi, venv chưa dùng
được; chọn thư mục venv mới sau khi sửa nguyên nhân.

Nếu wheel thiếu, dừng cổng binary. Chọn nguồn binary được IBM hỗ trợ hoặc xây từ
source trong môi trường riêng theo tài liệu chính thức, lưu commit/toolchain/
build log và qua smoke test; không tự thay gcc/glibc/driver của server. Environment
training và CPU inference nên tách riêng; môi trường RTX/TensorRT ở máy NVIDIA.
Lưu `pip freeze`, wheel SHA256, `pip check`, ORT providers/build info và checkpoint
hash cùng run. Không coi import thành công là đã qua accuracy/performance gate.

## 4. Telemetry

```bash
python3 scripts/telemetry.py --output runs/session01/telemetry --seconds 600 --interval 1
```

Có thể thêm `--pid PID_CUA_BENCHMARK` để lấy RSS/peak RSS/số thread/I/O của tiến
trình đó. Khởi động collector song song từ terminal khác trước benchmark; dừng
bằng Ctrl-C để kết thúc sớm. Output gồm metadata, JSONL và CSV. CPU busy/steal
từ procfs là toàn LPAR/host hiện thấy, không phải riêng model. `/proc/ppc64/lparcfg`
ghi các counter PowerVM đọc được; PURR/SPURR là counter thô, không tự biến thành
core entitlement sử dụng nếu chưa có timebase/cách diễn giải phù hợp. RSS của một
PID không cộng các worker con.

Collector không đo công suất, nhiệt, memory bandwidth, GPU utilization hay MMA
instructions. `metadata.json` ghi các tool ngoài có/không nhưng không tự chạy
chúng; admin có thể thu riêng `lparstat`, `numastat`, `iostat`, `perf` và telemetry
NVIDIA theo quyền/tool đã cài. Lưu command thực dùng, sampling interval, output
gốc, version; báo N/A khi thiếu. Không so energy/frame nếu chỉ một máy có wattmeter
hoặc phạm vi công suất khác nhau. Pin collector vào CPU ngoài mask model nếu cần
đo độ nhiễu; ghi cách pin và cùng overhead trên mọi lần so sánh.

## 5. Affinity, NUMA và nhiều worker

```bash
python3 scripts/sweep.py --config configs/sweep.cpu.json --output runs/sweep-plan --plan-only
python3 scripts/sweep.py --config configs/sweep.cpu.json --output runs/sweep01
```

Đọc `plan.json` trước khi chạy. Mẫu rộng có thể tạo hàng nghìn case; bắt đầu với
threads `[1,4,8]`, smt_limits `[1,8]`, workers `[1]`, repeats `3`, rồi mở rộng quanh
điểm tốt nhất. Mỗi thư mục output của lần chạy thực phải mới, tránh lẫn kết quả.
`base_config` tương đối tính từ thư mục chứa config sweep. Đường dẫn model/input
trong base config tuân theo benchmark; nên dùng đường dẫn tuyệt đối khi chạy ở
nơi khác. Chạy từ gốc repo với các mẫu hiện có.

`smt_limits` chỉ lấy tối đa N logical sibling trong mỗi core từ sysfs và cpuset
được cấp, không thay SMT của OS. CPU ID không cần liên tục. Các mask worker tách
biệt và số logical CPU trong mỗi mask bằng số ORT intra threads; inter threads=1,
OpenBLAS/MKL=1 để tránh nhiều tầng song song. ORT MLAS không mặc nhiên dùng
OpenBLAS; đây là giới hạn tránh oversubscription cho thư viện khác nếu có.

`numa-local` chạy từng node với `numactl --physcpubind=… --membind=…`; thiếu tool
hoặc không được phép bind thì báo lỗi, không tự đổi chính sách. `all` chọn theo
CPU ID, `spread` xen kẽ CPU từ các NUMA; cả hai dùng `taskset` và memory policy mặc
định (first touch), **không** tuyên bố interleave memory. So thêm interleave bằng
một chiến dịch riêng nếu cần và lưu chính sách chính xác. `smt_limit=8` cùng
threads nhỏ có thể lấy nhiều sibling của cùng core vì ID thực; xem mask để hiểu
layout. Chỉ số smt_limit là giới hạn chọn, không đảm bảo mọi core đang chạy N
thread trong mọi case.

Các worker chạy đồng thời, mỗi worker có session/model riêng. Throughput tổng là
`tổng images đã đo / thời gian parent từ trước launch đến sau join`, gồm startup,
warmup và teardown; vì vậy được đặt tên `throughput_makespan_fps`. Nó là số đo
bảo thủ của job hữu hạn, không phải steady-state phục vụ. Tăng iterations đủ dài
để startup <1% makespan, hoặc dùng harness có cửa sổ đồng bộ chung cho SLA dài
hạn. Không cộng FPS của những lần chạy worker riêng lẻ. Nếu worker lỗi hoặc thiếu
result, summary đánh `valid=false`, FPS tổng để null. Xem log từng worker và không
đưa case lỗi vào biểu đồ. `results.csv/json` cùng `summary.json` mỗi case giữ CPU
mask, số thread/worker, node, repeat và công thức đo.

Giữ dataset/input đã cache và page-cache policy nhất quán. Chạy 3–5 repeat, đảo
thứ tự chiến dịch hoặc lặp lại theo thứ tự đảo để tìm drift (harness hiện chạy
theo thứ tự cố định). Không rút kết luận SMT8/NUMA tốt nhất từ một repeat. Ghi
entitlement, pool sharing, tải nền, governor/công suất nếu đo được. Chốt best
latency và best total throughput như hai cấu hình riêng vì mục tiêu khác nhau.

Inventory hiện đọc quota cgroup hiện tại và root; nếu chạy trong container có cgroup lồng nhau, cần kiểm quota của mọi cgroup tổ tiên. Affinity/cpuset không thay thế CPU quota hoặc entitlement PowerVM; sweep chưa tự giới hạn số threads theo quota tổ tiên.
