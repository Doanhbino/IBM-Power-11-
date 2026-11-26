#include "rfdetr/decoder.hpp"
#include <algorithm>
#include <cmath>

namespace rfdetr {

Decoder::Decoder(DecodeConfig config) : config_(config) {
    if (config.top_k == 0 || !std::isfinite(config.threshold) ||
        config.threshold < 0.f || config.threshold > 1.f)
        throw std::invalid_argument("Invalid top-k/threshold");
}

const std::vector<Detection>& Decoder::run(const float* boxes, const float* logits,
    std::size_t queries, std::size_t classes, int height, int width) {
    if (!boxes || !logits || queries == 0 || classes == 0 || height < 1 || width < 1)
        throw std::invalid_argument("Invalid RF-DETR output dimensions");
    const auto elements = checked_product(queries, classes);
    for (std::size_t i = 0; i < checked_product(queries, 4); ++i)
        if (!std::isfinite(boxes[i])) throw std::invalid_argument("Nonfinite box output");
    std::optional<std::size_t> background;
    if (config_.background_index) {
        const auto index = static_cast<std::int64_t>(*config_.background_index);
        const auto resolved = index < 0 ? static_cast<std::int64_t>(classes) + index : index;
        if (resolved < 0 || static_cast<std::size_t>(resolved) >= classes)
            throw std::invalid_argument("Invalid background slot");
        background = static_cast<std::size_t>(resolved);
    }
    if (classes == 1 && background) throw std::invalid_argument("No foreground class");
    candidates_.clear();
    for (std::size_t i = 0; i < elements; ++i) {
        if (!std::isfinite(logits[i])) throw std::invalid_argument("Nonfinite class output");
        if (background && i % classes == *background) continue;
        const float score = 1.f / (1.f + std::exp(-std::clamp(logits[i], -88.f, 88.f)));
        candidates_.push_back({score, i});
    }
    const auto count = std::min(config_.top_k, candidates_.size());
    const auto higher = [](const Candidate& a, const Candidate& b) {
        return a.score != b.score ? a.score > b.score : a.index < b.index;
    };
    std::partial_sort(candidates_.begin(), candidates_.begin() + count, candidates_.end(), higher);
    detections_.clear();
    detections_.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto candidate = candidates_[i];
        if (candidate.score <= config_.threshold) break;
        const auto query = candidate.index / classes;
        const float* b = boxes + query * 4;
        detections_.push_back({{
            std::clamp((b[0] - b[2] / 2.f) * width, 0.f, static_cast<float>(width)),
            std::clamp((b[1] - b[3] / 2.f) * height, 0.f, static_cast<float>(height)),
            std::clamp((b[0] + b[2] / 2.f) * width, 0.f, static_cast<float>(width)),
            std::clamp((b[1] + b[3] / 2.f) * height, 0.f, static_cast<float>(height))},
            candidate.score, candidate.index % classes, query});
    }
    return detections_;
}

}  // namespace rfdetr
