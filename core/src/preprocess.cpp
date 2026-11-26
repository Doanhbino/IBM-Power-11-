#include "rfdetr/preprocess.hpp"
#include <algorithm>
#include <cmath>

namespace rfdetr {

Preprocessor::Preprocessor(int height, int width, Normalization normalization)
    : height_(height), width_(width), normalization_(normalization) {
    if (height < 1 || width < 1) throw std::invalid_argument("Invalid input size");
    for (std::size_t c = 0; c < 3; ++c) {
        if (!std::isfinite(normalization.mean[c]) || !std::isfinite(normalization.std[c]) ||
            normalization.std[c] <= 0)
            throw std::invalid_argument("Normalization must be finite with positive std");
    }
    output_.resize(checked_product(checked_product(height, width), 3));
}

std::vector<Preprocessor::Axis> Preprocessor::axis(int source, int target) {
    std::vector<Axis> result(target);
    for (int i = 0; i < target; ++i) {
        const double coordinate = std::max((i + .5) * source / target - .5, 0.0);
        const int lo = static_cast<int>(std::floor(coordinate));
        result[i] = {lo, std::min(lo + 1, source - 1), static_cast<float>(coordinate - lo)};
    }
    return result;
}

const std::vector<float>& Preprocessor::run(ImageView image) {
    if (!image.data || image.height < 1 || image.width < 1 ||
        image.stride_bytes < checked_product(image.width, 3))
        throw std::invalid_argument("Expected nonempty BGR uint8 image with valid stride");
    checked_product(image.height, image.stride_bytes);
    if (image.height != source_height_ || image.width != source_width_) {
        ys_ = axis(image.height, height_);
        xs_ = axis(image.width, width_);
        source_height_ = image.height;
        source_width_ = image.width;
    }
    const std::size_t plane = checked_product(height_, width_);
    for (int y = 0; y < height_; ++y) {
        const auto ay = ys_[y];
        const auto* row0 = image.data + ay.lo * image.stride_bytes;
        const auto* row1 = image.data + ay.hi * image.stride_bytes;
        for (int x = 0; x < width_; ++x) {
            const auto ax = xs_[x];
            for (std::size_t c = 0; c < 3; ++c) {
                const auto bgr = 2 - c;
                const float a = static_cast<float>(row0[3 * ax.lo + bgr]) / 255.f;
                const float b = static_cast<float>(row0[3 * ax.hi + bgr]) / 255.f;
                const float d = static_cast<float>(row1[3 * ax.lo + bgr]) / 255.f;
                const float e = static_cast<float>(row1[3 * ax.hi + bgr]) / 255.f;
                const float top = a * (1.f - ax.weight) + b * ax.weight;
                const float bottom = d * (1.f - ax.weight) + e * ax.weight;
                const float value = top * (1.f - ay.weight) + bottom * ay.weight;
                output_[c * plane + static_cast<std::size_t>(y) * width_ + x] =
                    (value - normalization_.mean[c]) / normalization_.std[c];
            }
        }
    }
    return output_;
}

}  // namespace rfdetr
