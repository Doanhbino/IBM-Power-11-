#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace rfdetr {

inline std::size_t checked_product(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error("Tensor size overflow");
    return a * b;
}

struct ImageView {
    const std::uint8_t* data;
    int height;
    int width;
    std::size_t stride_bytes;
};

struct Normalization {
    std::array<float, 3> mean{.485f, .456f, .406f};
    std::array<float, 3> std{.229f, .224f, .225f};
};

struct Detection {
    std::array<float, 4> xyxy;
    float score;
    std::size_t class_slot;
    std::size_t query;
};

}  // namespace rfdetr
