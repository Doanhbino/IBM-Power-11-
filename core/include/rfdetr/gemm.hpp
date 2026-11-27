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

// Float storage whose data() is 64-byte aligned, so packed panels can be read with aligned vector loads.
class AlignedBuffer {
public:
    AlignedBuffer() = default;
    AlignedBuffer(const AlignedBuffer& other) { assign_from(other); }
    AlignedBuffer& operator=(const AlignedBuffer& other) { if (this != &other) assign_from(other); return *this; }
    AlignedBuffer(AlignedBuffer&&) noexcept = default;  // vector move keeps the heap block, so the offset stays valid
    AlignedBuffer& operator=(AlignedBuffer&&) noexcept = default;
    void resize(std::size_t count);  // zero-filled
    float* data() noexcept { return storage_.data() + offset_; }
    const float* data() const noexcept { return storage_.data() + offset_; }
    std::size_t size() const noexcept { return size_; }
private:
    void assign_from(const AlignedBuffer& other);
    std::vector<float> storage_;
    std::size_t offset_ = 0, size_ = 0;
};

// Immutable packed B, reusable for constant weights. Never caches changing inputs.
class PackedRhs {
public:
    explicit PackedRhs(MatrixView rhs);
    std::size_t k() const noexcept { return k_; }
    std::size_t n() const noexcept { return n_; }
    // `column` must be a multiple of the 16-column tile; the panel holds [k][16] floats, 64-byte aligned.
    const float* panel(std::size_t column) const noexcept;
private:
    std::size_t k_, n_;
    AlignedBuffer values_;
};

// One workspace per caller/worker. C is contiguous row-major, C = A @ B, FP32.
// Output may not alias A. No hidden thread pool or implicit precision conversion.
class GemmWorkspace {
public:
    void run(MatrixView lhs, const PackedRhs& rhs, float* output, GemmBackend backend);
private:
    AlignedBuffer lhs_panels_;
};

}  // namespace rfdetr
