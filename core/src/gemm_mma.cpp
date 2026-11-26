// Only this translation unit is compiled for ISA 3.1. Generic dispatch stays portable.
#include <altivec.h>
#include <cstddef>
#include <cstring>

namespace rfdetr {

__attribute__((noinline)) void mma_tile(const float* a, const float* b, std::size_t k, float* tile) {
    __vector_quad accumulator;
    __builtin_mma_xxsetaccz(&accumulator);
    for (std::size_t p = 0; p < k; ++p) {
        __vector unsigned char left, right;
        std::memcpy(&left, a + p * 4, sizeof(left));
        std::memcpy(&right, b + p * 4, sizeof(right));
        __builtin_mma_xvf32gerpp(&accumulator, left, right);
    }
    __vector float rows[4];
    __builtin_mma_disassemble_acc(rows, &accumulator);
    std::memcpy(tile, rows, sizeof(rows));
}

}  // namespace rfdetr
