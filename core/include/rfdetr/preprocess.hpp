#pragma once

#include "rfdetr/types.hpp"
#include <vector>

namespace rfdetr {

// Worker-local workspace. Output remains valid until the next run().
// Stretch BGR uint8 -> RGB float32 NCHW, half-pixel bilinear, no antialias.
class Preprocessor {
public:
    Preprocessor(int height, int width, Normalization normalization = {});
    const std::vector<float>& run(ImageView image);
    int height() const noexcept { return height_; }
    int width() const noexcept { return width_; }

private:
    struct Axis { int lo, hi; float weight; };
    static std::vector<Axis> axis(int source, int target);
    int height_, width_, source_height_{0}, source_width_{0};
    Normalization normalization_;
    std::vector<Axis> ys_, xs_;
    std::vector<float> output_;
};

}  // namespace rfdetr
