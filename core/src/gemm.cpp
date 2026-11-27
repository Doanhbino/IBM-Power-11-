#include "rfdetr/gemm.hpp"
#include "rfdetr/types.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#if defined(__linux__) && defined(__powerpc64__)
#include <sys/auxv.h>
#include <asm/cputable.h>
#endif

namespace rfdetr {
#ifdef POWER11_HAS_MMA_KERNEL
void mma_tile_8x16(const float* a, const float* b, std::size_t k, float* tile);
#endif
namespace {
// Micro-kernel geometry from REDP-5612 section 4.1: eight 4x4 accumulators form an 8x16 FP32 tile, and K is
// processed in blocks so one A block (8x256x4 = 8 KiB) plus one B block (16x256x4 = 16 KiB) fit in a 32 KiB L1.
constexpr std::size_t kTileM = 8, kTileN = 16, kBlockK = 256;

void validate(MatrixView matrix) {
    if (!matrix.data || !matrix.rows || !matrix.cols || !matrix.row_stride || !matrix.col_stride)
        throw std::invalid_argument("Expected nonempty matrix with positive strides");
    const auto last_row = checked_product(matrix.rows - 1, matrix.row_stride);
    const auto last_col = checked_product(matrix.cols - 1, matrix.col_stride);
    if (last_row > std::numeric_limits<std::size_t>::max() - last_col)
        throw std::overflow_error("Matrix offset overflow");
}
void reference_tile(const float* a, const float* b, std::size_t k, float* out) {
    std::fill(out, out + kTileM * kTileN, 0.f);
    for (std::size_t p = 0; p < k; ++p)
        for (std::size_t i = 0; i < kTileM; ++i)
            for (std::size_t j = 0; j < kTileN; ++j)
                out[i * kTileN + j] += a[p * kTileM + i] * b[p * kTileN + j];
}
}

MmaCapability mma_capability() noexcept {
    bool exposed = false;
#if defined(__linux__) && defined(__powerpc64__) && defined(PPC_FEATURE2_MMA)
    exposed = (getauxval(AT_HWCAP2) & PPC_FEATURE2_MMA) != 0;
#endif
#ifdef POWER11_HAS_MMA_KERNEL
    return {true, exposed};
#else
    return {false, exposed};
#endif
}

const char* backend_name(GemmBackend backend) noexcept {
    return backend == GemmBackend::mma ? "mma-fp32-8x16" : "reference-fp32-8x16";
}

void AlignedBuffer::resize(std::size_t count) {
    constexpr std::size_t alignment = 64, spare = alignment / sizeof(float);
    storage_.assign(count + spare, 0.f);
    const auto address = reinterpret_cast<std::uintptr_t>(storage_.data());
    offset_ = ((alignment - address % alignment) % alignment) / sizeof(float);
    size_ = count;
}

void AlignedBuffer::assign_from(const AlignedBuffer& other) {
    resize(other.size_);
    std::copy_n(other.data(), other.size_, data());
}

PackedRhs::PackedRhs(MatrixView rhs) : k_(rhs.rows), n_(rhs.cols) {
    validate(rhs);
    const auto panels = n_ / kTileN + (n_ % kTileN != 0);
    values_.resize(checked_product(checked_product(panels, k_), kTileN));
    float* out = values_.data();
    for (std::size_t j = 0; j < n_; ++j)
        for (std::size_t p = 0; p < k_; ++p)
            out[(j / kTileN * k_ + p) * kTileN + j % kTileN] = rhs.data[p * rhs.row_stride + j * rhs.col_stride];
}

const float* PackedRhs::panel(std::size_t column) const noexcept {
    return values_.data() + column / kTileN * k_ * kTileN;
}

void GemmWorkspace::run(MatrixView lhs, const PackedRhs& rhs, float* output, GemmBackend backend) {
    validate(lhs);
    if (!output || lhs.cols != rhs.k()) throw std::invalid_argument("GEMM shape/output mismatch");
    checked_product(lhs.rows, rhs.n());
    auto kernel = reference_tile;
    if (backend == GemmBackend::mma) {
        const auto capability = mma_capability();
        if (!capability.compiled || !capability.exposed)
            throw std::runtime_error("MMA requested but kernel/LPAR HWCAP2 unavailable; refusing fallback");
#ifdef POWER11_HAS_MMA_KERNEL
        kernel = mma_tile_8x16;
#endif
    }
    const std::size_t k = lhs.cols, m = lhs.rows, n = rhs.n();
    const std::size_t row_panels = m / kTileM + (m % kTileM != 0);
    // Pack every 8-row panel of A once as [panel][k][8]; tails are zero padded.
    lhs_panels_.resize(checked_product(checked_product(row_panels, k), kTileM));
    float* packed = lhs_panels_.data();
    for (std::size_t panel = 0; panel < row_panels; ++panel) {
        const auto rows = std::min(kTileM, m - panel * kTileM);
        float* dst = packed + panel * k * kTileM;
        for (std::size_t p = 0; p < k; ++p)
            for (std::size_t r = 0; r < kTileM; ++r)
                dst[p * kTileM + r] = r < rows ? lhs.data[(panel * kTileM + r) * lhs.row_stride + p * lhs.col_stride] : 0.f;
    }
    // Loop order from REDP-5612 Example 4-6: keep the larger B block resident in L1 (j, then K block) and stream
    // the smaller A blocks past it (i). Partial sums from successive K blocks are added into the output.
    alignas(64) float tile[kTileM * kTileN];
    for (std::size_t j = 0; j < n; j += kTileN) {
        const auto columns = std::min(kTileN, n - j);
        for (std::size_t p0 = 0; p0 < k; p0 += kBlockK) {
            const auto depth = std::min(kBlockK, k - p0);
            const float* b_block = rhs.panel(j) + p0 * kTileN;
            for (std::size_t panel = 0; panel < row_panels; ++panel) {
                kernel(packed + panel * k * kTileM + p0 * kTileM, b_block, depth, tile);
                const auto rows = std::min(kTileM, m - panel * kTileM);
                for (std::size_t r = 0; r < rows; ++r) {
                    float* dst = output + (panel * kTileM + r) * n + j;
                    const float* src = tile + r * kTileN;
                    if (p0 == 0) std::copy_n(src, columns, dst);
                    else for (std::size_t c = 0; c < columns; ++c) dst[c] += src[c];
                }
            }
        }
    }
}

}  // namespace rfdetr
