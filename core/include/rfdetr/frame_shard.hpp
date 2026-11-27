#pragma once

#include <cstddef>
#include <stdexcept>

namespace rfdetr {

// Several workers can share one camera by each keeping every `stride`-th frame (those with index % stride == offset).
// Shards of one camera partition its frame sequence, so no frame is processed twice. Default: keep every frame.
struct FrameShard {
    std::size_t stride{1}, offset{0};
    FrameShard() = default;
    FrameShard(std::size_t stride_, std::size_t offset_) : stride(stride_), offset(offset_) {
        if (stride_ < 1 || offset_ >= stride_) throw std::invalid_argument("Frame shard needs stride >= 1 and offset < stride");
    }
    bool selected(std::size_t index) const noexcept { return index % stride == offset; }
};

}  // namespace rfdetr
