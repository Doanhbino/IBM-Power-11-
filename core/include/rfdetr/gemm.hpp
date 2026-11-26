#pragma once

#include <cstddef>
#include <vector>

namespace rfdetr {

enum class GemmBackend { reference, mma };
struct MmaCapability { bool compiled; bool exposed; };
MmaCapability mma_capability() noexcept;
const char* backend_name(GemmBackend backend) noexcept;

// Logical matrix with positive element strides; transpose by swapping strides.
struct MatrixView {
    const float* data;
    std::size_t rows, cols, row_stride, col_stride;
};

// Immutable packed B, reusable for constant weights. Never caches changing inputs.
class PackedRhs {
public:
    explicit PackedRhs(MatrixView rhs);
    std::size_t k() const noexcept { return k_; }
    std::size_t n() const noexcept { return n_; }
    const float* panel(std::size_t column) const noexcept;
private:
    std::size_t k_, n_;
    std::vector<float> values_;
};

// One workspace per caller/worker. C is contiguous row-major, C = A @ B, FP32.
// Output may not alias A. No hidden thread pool or implicit precision conversion.
class GemmWorkspace {
public:
    void run(MatrixView lhs, const PackedRhs& rhs, float* output, GemmBackend backend);
private:
    std::vector<float> lhs_panel_;
};

}  // namespace rfdetr
