#include "rfdetr/layernorm.hpp"
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace rfdetr {
namespace {
typedef float V4 __attribute__((vector_size(16)));

inline V4 load(const float* p) { V4 v; std::memcpy(&v, p, sizeof v); return v; }
inline void store(float* p, V4 v) { std::memcpy(p, &v, sizeof v); }
inline V4 splat(float f) { return V4{f, f, f, f}; }
inline float horizontal_sum(V4 v) { return (v[0] + v[1]) + (v[2] + v[3]); }
}  // namespace

void layer_norm_rows(const float* x, const float* scale, const float* bias, float* y,
                     std::size_t rows, std::size_t n, float epsilon) {
    if (n == 0) throw std::invalid_argument("LayerNorm row length must be positive");
    const float count = static_cast<float>(n);
    for (std::size_t row = 0; row < rows; ++row) {
        const float* in = x + row * n;
        float* out = y + row * n;
        std::size_t i = 0;
        // Pass 1: mean.
        V4 acc0 = splat(0.f), acc1 = splat(0.f);
        for (; i + 8 <= n; i += 8) { acc0 += load(in + i); acc1 += load(in + i + 4); }
        for (; i + 4 <= n; i += 4) acc0 += load(in + i);
        float sum = horizontal_sum(acc0 + acc1);
        for (; i < n; ++i) sum += in[i];
        const float mean = sum / count;
        // Pass 2: sum and sum of squares of the deviations. Folding the sum back into the mean removes the rounding
        // error of pass 1 (corrected two-pass): var = (sum(d^2) - sum(d)^2 / n) / n.
        const V4 centre = splat(mean);
        V4 deviation = splat(0.f), square = splat(0.f);
        for (i = 0; i + 4 <= n; i += 4) {
            const V4 d = load(in + i) - centre;
            deviation += d;
            square += d * d;
        }
        float dsum = horizontal_sum(deviation), dsq = horizontal_sum(square);
        for (; i < n; ++i) { const float d = in[i] - mean; dsum += d; dsq += d * d; }
        const float correction = dsum / count;
        float variance = (dsq - dsum * correction) / count;
        if (variance < 0.f) variance = 0.f;
        const float inverse = 1.f / std::sqrt(variance + epsilon);
        const V4 shift = splat(mean + correction), factor = splat(inverse);
        // Pass 3: normalise, scale, shift.
        for (i = 0; i + 4 <= n; i += 4) {
            V4 v = (load(in + i) - shift) * factor * load(scale + i);
            if (bias) v += load(bias + i);
            store(out + i, v);
        }
        for (; i < n; ++i) {
            float v = (in[i] - (mean + correction)) * inverse * scale[i];
            out[i] = bias ? v + bias[i] : v;
        }
    }
}

}  // namespace rfdetr
