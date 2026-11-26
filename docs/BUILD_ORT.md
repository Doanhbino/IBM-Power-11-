# Dựng ORT CPU khi không có wheel ppc64le

Đây là đường source build có tham số đối chiếu upstream, **chưa build trên Power11**. Không mặc định RHEL 9.8 có đủ toolchain. Chọn release sau khi kiểm support; `v1.25.0` chỉ là ứng viên đã xác nhận có mã nguồn, không phải pin tối ưu được chứng nhận. Nếu dùng release khác, kiểm lại yêu cầu và cờ của đúng checkout.

Dùng tài khoản thường, thư mục dự án, Python build venv riêng, compiler/CMake đã được quản trị viên cấp hoặc user-local. Tài liệu hiện tại nêu Python ≥3.10 và CMake ≥3.28; không chạy cài system package theo hướng dẫn Ubuntu trên RHEL. Xem [hướng dẫn build chính thức](https://onnxruntime.ai/docs/build/inferencing.html) và [script của candidate](https://github.com/microsoft/onnxruntime/blob/v1.25.0/tools/ci_build/build.py). Không tự thay gcc/glibc/driver của host.

```bash
# Chạy trong thư mục source dành riêng của dự án, build venv đã kích hoạt.
# Thư mục đích phải chưa có. Phải có quyền network hoặc mirror/offline source hợp lệ.
git clone --branch v1.25.0 --recursive https://github.com/microsoft/onnxruntime.git ort-v1.25.0
cd ort-v1.25.0
git rev-parse HEAD > source-commit.txt
git submodule status --recursive > source-submodules.txt
python --version
cmake --version
gcc --version
./build.sh --help > build-help.txt
# Giới hạn parallel khởi đầu 8; điều chỉnh theo entitlement/build RAM, không dùng 240 mặc định.
# Các package build phải có sẵn trong venv; thiếu dependency thì dừng để giải quyết riêng.
./build.sh --config Release --build_shared_lib --build_wheel --parallel 8 > build.log 2>&1
```

Lưu log kể cả khi thất bại, chạy các test upstream sinh ra và kiểm tra wheel `build/Linux/Release/dist` thực tế. Không thêm `--skip_tests` để biến build lỗi thành kết quả đạt. Đưa wheel tự build cùng các dependency đã kiểm vào wheelhouse mới, tạo lock/hash, cài vào **venv inference mới**, `pip check`, test ONNX thật, accuracy và sweep. Cần lưu cả compiler/CMake/dependency provenance; wheel tự build không mang nghĩa hỗ trợ thương mại từ IBM/Microsoft.

Không thêm cờ giả định `USE_OPENBLAS`, không thay MLAS bằng thư viện khác chỉ vì đã cài OpenBLAS. Mã [MLAS CMake](https://github.com/microsoft/onnxruntime/blob/v1.23.2/cmake/onnxruntime_mlas.cmake) có probe compiler/ISA Power10; đối chiếu lại file cùng tag build. Kiểm log cho các source POWER/MMA và profile runtime như `SUPPORT.md`. Build thành công hoặc CPU EP hiện diện chưa chứng minh model RF-DETR thực sự chạy kernel MMA.
