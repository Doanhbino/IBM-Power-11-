#pragma once

#include <cstddef>

namespace rfdetr {

// LayerNorm over the last dimension of `rows` rows of length `n` (FP32):
//   y = (x - mean) / sqrt(var + epsilon) * scale + bias,  `bias` may be null.
// Written with GCC vector extensions so the compiler emits VSX on Power and SSE/NEON elsewhere. Mean and variance use
// a corrected two-pass algorithm, so rows with a large mean (typical of transformer activations) stay accurate.
// y may alias x: a row is fully read before it is written. Throws std::invalid_argument if n == 0.
void layer_norm_rows(const float* x, const float* scale, const float* bias, float* y,
                     std::size_t rows, std::size_t n, float epsilon);

}  // namespace rfdetr
