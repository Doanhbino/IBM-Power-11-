#include "rfdetr/gemm.hpp"
#include "rfdetr/types.hpp"
#include <algorithm>
#include <limits>
#if defined(__linux__) && defined(__powerpc64__)
#include <sys/auxv.h>
#include <asm/cputable.h>
#endif

namespace rfdetr {
#ifdef POWER11_HAS_MMA_KERNEL
void mma_tile(const float* a, const float* b, std::size_t k, float* tile);
#endif
namespace {
void validate(MatrixView matrix) {
    if (!matrix.data || !matrix.rows || !matrix.cols || !matrix.row_stride || !matrix.col_stride)
        throw std::invalid_argument("Expected nonempty matrix with positive strides");
    const auto last_row = checked_product(matrix.rows - 1, matrix.row_stride);
    const auto last_col = checked_product(matrix.cols - 1, matrix.col_stride);
    if (last_row > std::numeric_limits<std::size_t>::max() - last_col)
        throw std::overflow_error("Matrix offset overflow");
}
void reference_tile(const float* a, const float* b, std::size_t k, float* out) {
    std::fill(out, out + 16, 0.f);
    for (std::size_t p = 0; p < k; ++p)
        for (std::size_t i = 0; i < 4; ++i)
            for (std::size_t j = 0; j < 4; ++j)
                out[i * 4 + j] += a[p * 4 + i] * b[p * 4 + j];
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
    return backend == GemmBackend::mma ? "mma-fp32-4x4" : "reference-fp32-4x4";
}

PackedRhs::PackedRhs(MatrixView rhs) : k_(rhs.rows), n_(rhs.cols) {
    validate(rhs);
    const auto panels = n_ / 4 + (n_ % 4 != 0);
    values_.resize(checked_product(checked_product(panels, k_), 4), 0.f);
    for (std::size_t j = 0; j < n_; ++j)
        for (std::size_t p = 0; p < k_; ++p)
            values_[(j / 4 * k_ + p) * 4 + j % 4] = rhs.data[p * rhs.row_stride + j * rhs.col_stride];
}

const float* PackedRhs::panel(std::size_t column) const noexcept {
    return values_.data() + column / 4 * k_ * 4;
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
        kernel = mma_tile;
#endif
    }
    lhs_panel_.resize(checked_product(lhs.cols, 4));
    for (std::size_t i = 0; i < lhs.rows; i += 4) {
        const auto rows = std::min<std::size_t>(4, lhs.rows - i);
        for (std::size_t p = 0; p < lhs.cols; ++p)
            for (std::size_t r = 0; r < 4; ++r)
                lhs_panel_[p * 4 + r] = r < rows ? lhs.data[(i + r) * lhs.row_stride + p * lhs.col_stride] : 0.f;
        for (std::size_t j = 0; j < rhs.n(); j += 4) {
            alignas(16) float tile[16];
            kernel(lhs_panel_.data(), rhs.panel(j), lhs.cols, tile);
            const auto columns = std::min<std::size_t>(4, rhs.n() - j);
            for (std::size_t r = 0; r < rows; ++r)
                std::copy_n(tile + r * 4, columns, output + (i + r) * rhs.n() + j);
        }
    }
}

}  // namespace rfdetr
