#pragma once

#include "rfdetr/types.hpp"
#include <optional>
#include <vector>

namespace rfdetr {

struct DecodeConfig {
    // nullopt means no background slot. Negative indices follow Python indexing.
    std::optional<int> background_index;
    std::size_t top_k{300};
    float threshold{.25f};
};

class Decoder {
public:
    explicit Decoder(DecodeConfig config);
    const std::vector<Detection>& run(const float* boxes, const float* logits,
        std::size_t queries, std::size_t classes, int original_height, int original_width);

private:
    struct Candidate { float score; std::size_t index; };
    DecodeConfig config_;
    std::vector<Candidate> candidates_;
    std::vector<Detection> detections_;
};

}  // namespace rfdetr
