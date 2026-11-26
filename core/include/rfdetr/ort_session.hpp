#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace rfdetr {

struct OrtConfig {
    std::string model, input_name, boxes_name, logits_name, profile_prefix;
    int height{}, width{}, threads{1};
};
struct OutputView {
    const float* boxes;
    const float* logits;
    std::size_t queries, classes;
};

// CPU-only ORT session; uses the linked MLAS binary without replacing its backend.
class OrtSession {
public:
    explicit OrtSession(const OrtConfig& config);
    ~OrtSession();
    OrtSession(const OrtSession&) = delete;
    OrtSession& operator=(const OrtSession&) = delete;
    OutputView run(const float* nchw, std::size_t elements);
    std::string end_profile();
    static std::string version();
    static std::string library_path();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace rfdetr
