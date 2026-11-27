// Only this translation unit is compiled for ISA 3.1. Generic dispatch stays portable.
//
// 8x16 FP32 micro-kernel after IBM REDP-5612 section 4.1 ("Multiple accumulators SGEMM for load value reuse"):
// all eight 512-bit accumulators hold one 4x4 block each, so every k step issues eight xvf32gerpp from six
// 128-bit loads (two from A, four from B) instead of two loads per gerpp. No VSR is touched while an accumulator
// is primed, which avoids the accumulator/VSR transfers REDP-5612 section 2.2 warns about.
//
// Inputs are packed by gemm.cpp and 64-byte aligned:  a = [k][8] (rows of C),  b = [k][16] (columns of C).
// Output tile is row-major [8][16].
#include <altivec.h>
#include <cstddef>

namespace rfdetr {

namespace {
typedef __vector unsigned char vec_t;

// Accumulator (row block I, column block J) holds C[4I..4I+3][4J..4J+3]. Row r of the accumulator is C row 4I+r.
inline void store_block(float* tile, __vector_quad* acc, int row_block, int column_block) {
    __vector float rows[4];
    __builtin_mma_disassemble_acc(rows, acc);
    for (int r = 0; r < 4; ++r)
        *reinterpret_cast<__vector float*>(tile + (row_block * 4 + r) * 16 + column_block * 4) = rows[r];
}
}  // namespace

__attribute__((noinline)) void mma_tile_8x16(const float* a, const float* b, std::size_t k, float* tile) {
    __vector_quad acc00, acc01, acc02, acc03, acc10, acc11, acc12, acc13;
    __builtin_mma_xxsetaccz(&acc00);
    __builtin_mma_xxsetaccz(&acc01);
    __builtin_mma_xxsetaccz(&acc02);
    __builtin_mma_xxsetaccz(&acc03);
    __builtin_mma_xxsetaccz(&acc10);
    __builtin_mma_xxsetaccz(&acc11);
    __builtin_mma_xxsetaccz(&acc12);
    __builtin_mma_xxsetaccz(&acc13);
    for (std::size_t p = 0; p < k; ++p) {
        const vec_t* pa = reinterpret_cast<const vec_t*>(a + p * 8);
        const vec_t* pb = reinterpret_cast<const vec_t*>(b + p * 16);
        vec_t a0 = pa[0], a1 = pa[1];
        vec_t b0 = pb[0], b1 = pb[1], b2 = pb[2], b3 = pb[3];
        __builtin_mma_xvf32gerpp(&acc00, a0, b0);
        __builtin_mma_xvf32gerpp(&acc01, a0, b1);
        __builtin_mma_xvf32gerpp(&acc02, a0, b2);
        __builtin_mma_xvf32gerpp(&acc03, a0, b3);
        __builtin_mma_xvf32gerpp(&acc10, a1, b0);
        __builtin_mma_xvf32gerpp(&acc11, a1, b1);
        __builtin_mma_xvf32gerpp(&acc12, a1, b2);
        __builtin_mma_xvf32gerpp(&acc13, a1, b3);
    }
    store_block(tile, &acc00, 0, 0);
    store_block(tile, &acc01, 0, 1);
    store_block(tile, &acc02, 0, 2);
    store_block(tile, &acc03, 0, 3);
    store_block(tile, &acc10, 1, 0);
    store_block(tile, &acc11, 1, 1);
    store_block(tile, &acc12, 1, 2);
    store_block(tile, &acc13, 1, 3);
}

}  // namespace rfdetr
