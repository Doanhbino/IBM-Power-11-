#include "rfdetr/yolo.hpp"
#include <algorithm>
#include <cmath>
#include <string>

namespace rfdetr {

LetterboxPreprocessor::LetterboxPreprocessor(int height, int width, std::uint8_t pad_value)
    : height_(height), width_(width), pad_(static_cast<float>(pad_value) / 255.f) {
    if (height < 1 || width < 1) throw std::invalid_argument("Invalid input size");
    output_.resize(checked_product(checked_product(height, width), 3));
}

void LetterboxPreprocessor::configure(int source_height, int source_width) {
    // Ties-to-even (std::nearbyint) matches Python round(), which the NumPy reference uses.
    const double scale = std::min(static_cast<double>(height_) / source_height,
                                  static_cast<double>(width_) / source_width);
    new_width_ = std::max(1, static_cast<int>(std::nearbyint(source_width * scale)));
    new_height_ = std::max(1, static_cast<int>(std::nearbyint(source_height * scale)));
    meta_.scale = static_cast<float>(scale);
    meta_.pad_x = static_cast<int>(std::nearbyint((width_ - new_width_) / 2.0 - .1));
    meta_.pad_y = static_cast<int>(std::nearbyint((height_ - new_height_) / 2.0 - .1));
    meta_.source_height = source_height;
    meta_.source_width = source_width;
    ys_ = Preprocessor::axis(source_height, new_height_);
    xs_ = Preprocessor::axis(source_width, new_width_);
    std::fill(output_.begin(), output_.end(), pad_);  // content is rewritten every run; borders stay
}

const std::vector<float>& LetterboxPreprocessor::run(ImageView image) {
    if (!image.data || image.height < 1 || image.width < 1 ||
        image.stride_bytes < checked_product(image.width, 3))
        throw std::invalid_argument("Expected nonempty BGR uint8 image with valid stride");
    checked_product(image.height, image.stride_bytes);
    if (image.height != meta_.source_height || image.width != meta_.source_width)
        configure(image.height, image.width);
    const std::size_t plane = checked_product(height_, width_);
    for (int y = 0; y < new_height_; ++y) {
        const auto ay = ys_[y];
        const auto* row0 = image.data + ay.lo * image.stride_bytes;
        const auto* row1 = image.data + ay.hi * image.stride_bytes;
        for (int x = 0; x < new_width_; ++x) {
            const auto ax = xs_[x];
            for (std::size_t c = 0; c < 3; ++c) {
                const auto bgr = 2 - c;
                const float a = static_cast<float>(row0[3 * ax.lo + bgr]) / 255.f;
                const float b = static_cast<float>(row0[3 * ax.hi + bgr]) / 255.f;
                const float d = static_cast<float>(row1[3 * ax.lo + bgr]) / 255.f;
                const float e = static_cast<float>(row1[3 * ax.hi + bgr]) / 255.f;
                const float top = a * (1.f - ax.weight) + b * ax.weight;
                const float bottom = d * (1.f - ax.weight) + e * ax.weight;
                output_[c * plane + static_cast<std::size_t>(y + meta_.pad_y) * width_ + (x + meta_.pad_x)] =
                    top * (1.f - ay.weight) + bottom * ay.weight;
            }
        }
    }
    return output_;
}

YoloDecoder::YoloDecoder(YoloConfig config) : config_(config) {
    if (config.classes == 0 || config.max_det == 0 || !std::isfinite(config.conf_threshold) ||
        !std::isfinite(config.iou_threshold) || config.conf_threshold < 0.f || config.conf_threshold > 1.f ||
        config.iou_threshold < 0.f || config.iou_threshold > 1.f)
        throw std::invalid_argument("Invalid YOLO classes/max_det/thresholds");
}

namespace {
double iou(const std::array<float, 4>& a, const std::array<float, 4>& b) {
    const double inter = std::max(0., std::min<double>(a[2], b[2]) - std::max<double>(a[0], b[0])) *
                         std::max(0., std::min<double>(a[3], b[3]) - std::max<double>(a[1], b[1]));
    const double union_area = (static_cast<double>(a[2]) - a[0]) * (static_cast<double>(a[3]) - a[1]) +
                              (static_cast<double>(b[2]) - b[0]) * (static_cast<double>(b[3]) - b[1]) - inter;
    return union_area > 0 ? inter / union_area : 0.;
}
}  // namespace

const std::vector<Detection>& YoloDecoder::run(const float* output, std::size_t channels, std::size_t anchors,
                                               const LetterboxMeta& meta) {
    if (!output || anchors == 0 || !(meta.scale > 0.f) || meta.source_height < 1 || meta.source_width < 1)
        throw std::invalid_argument("Invalid YOLO output dimensions");
    if (channels != 4 + config_.classes)
        throw std::invalid_argument("num_classes=" + std::to_string(config_.classes) + " expects " +
                                    std::to_string(4 + config_.classes) + " channels but model output has " +
                                    std::to_string(channels) + " (" +
                                    std::to_string(channels >= 4 ? channels - 4 : 0) + " classes)");
    const auto elements = checked_product(channels, anchors);
    for (std::size_t i = 0; i < elements; ++i)
        if (!std::isfinite(output[i])) throw std::invalid_argument("Nonfinite model output");
    candidates_.clear();
    for (std::size_t a = 0; a < anchors; ++a) {
        std::size_t slot = 0;
        float score = output[4 * anchors + a];
        for (std::size_t k = 1; k < config_.classes; ++k) {
            const float value = output[(4 + k) * anchors + a];
            if (value > score) { score = value; slot = k; }
        }
        if (!(score > config_.conf_threshold)) continue;
        const float cx = output[a], cy = output[anchors + a], w = output[2 * anchors + a], h = output[3 * anchors + a];
        candidates_.push_back({score, a, slot, {cx - w / 2.f, cy - h / 2.f, cx + w / 2.f, cy + h / 2.f}});
    }
    std::sort(candidates_.begin(), candidates_.end(), [](const Candidate& l, const Candidate& r) {
        if (l.slot != r.slot) return l.slot < r.slot;
        return l.score != r.score ? l.score > r.score : l.anchor < r.anchor;
    });
    const double threshold = config_.iou_threshold;
    kept_.clear();  // indexes into candidates_, grouped by class
    std::size_t group_begin = 0;  // first entry of kept_ that belongs to the current class
    for (std::size_t i = 0; i < candidates_.size(); ++i) {
        if (i > 0 && candidates_[i].slot != candidates_[i - 1].slot) group_begin = kept_.size();
        bool suppressed = false;
        for (std::size_t k = group_begin; k < kept_.size() && !suppressed; ++k)
            suppressed = iou(candidates_[i].box, candidates_[kept_[k]].box) > threshold;
        if (!suppressed) kept_.push_back(i);
    }
    std::sort(kept_.begin(), kept_.end(), [&](std::size_t l, std::size_t r) {
        const auto &a = candidates_[l], &b = candidates_[r];
        return a.score != b.score ? a.score > b.score : a.anchor < b.anchor;
    });
    const auto count = std::min(config_.max_det, kept_.size());
    const float width = static_cast<float>(meta.source_width), height = static_cast<float>(meta.source_height);
    const float pad_x = static_cast<float>(meta.pad_x), pad_y = static_cast<float>(meta.pad_y);
    detections_.clear();
    detections_.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto& c = candidates_[kept_[i]];
        detections_.push_back({{std::clamp((c.box[0] - pad_x) / meta.scale, 0.f, width),
                                std::clamp((c.box[1] - pad_y) / meta.scale, 0.f, height),
                                std::clamp((c.box[2] - pad_x) / meta.scale, 0.f, width),
                                std::clamp((c.box[3] - pad_y) / meta.scale, 0.f, height)},
                               c.score, c.slot, c.anchor});
    }
    return detections_;
}

}  // namespace rfdetr
