#pragma once

#include "rfdetr/preprocess.hpp"
#include "rfdetr/types.hpp"
#include <cstdint>
#include <vector>

namespace rfdetr {

// How a frame was placed on the model canvas; needed to map boxes back to the source frame.
struct LetterboxMeta {
    float scale{1.f};
    int pad_x{0}, pad_y{0};
    int source_height{0}, source_width{0};
};

// Worker-local workspace. Output remains valid until the next run().
// Aspect-preserving half-pixel bilinear resize into a constant-padded canvas;
// BGR uint8 -> RGB float32 NCHW in [0,1]. Padding follows Ultralytics LetterBox.
class LetterboxPreprocessor {
public:
    LetterboxPreprocessor(int height, int width, std::uint8_t pad_value = 114);
    const std::vector<float>& run(ImageView image);
    const LetterboxMeta& meta() const noexcept { return meta_; }

private:
    void configure(int source_height, int source_width);
    int height_, width_, new_height_{0}, new_width_{0};
    float pad_;
    LetterboxMeta meta_;
    std::vector<Preprocessor::Axis> ys_, xs_;
    std::vector<float> output_;
};

struct YoloConfig {
    std::size_t classes;
    float conf_threshold;
    float iou_threshold;
    std::size_t max_det;
};

// Output layout [channels = 4 + classes][anchors]: cx, cy, w, h in model-input pixels, then class
// scores (already sigmoided). Each anchor keeps its best class; NMS runs per class in model space
// (IoU > iou_threshold is suppressed); survivors are ordered by score then anchor, cut to max_det,
// mapped back through the letterbox and clipped to the source frame. Detection::query is the anchor.
class YoloDecoder {
public:
    explicit YoloDecoder(YoloConfig config);
    const std::vector<Detection>& run(const float* output, std::size_t channels, std::size_t anchors,
                                      const LetterboxMeta& meta);

private:
    struct Candidate { float score; std::size_t anchor, slot; std::array<float, 4> box; };
    YoloConfig config_;
    std::vector<Candidate> candidates_;
    std::vector<std::size_t> kept_;
    std::vector<Detection> detections_;
};

}  // namespace rfdetr
