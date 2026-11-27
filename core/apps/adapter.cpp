// Binary fixture bridge for independent NumPy/C++ parity tests, not a serving API.
#include "rfdetr/preprocess.hpp"
#include "rfdetr/decoder.hpp"
#include "rfdetr/yolo.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace {
template <typename T> std::vector<T> read(const std::string& path, std::size_t size) {
    std::ifstream file(path, std::ios::binary);
    std::vector<T> values(size);
    if (!file.read(reinterpret_cast<char*>(values.data()), rfdetr::checked_product(size, sizeof(T))))
        throw std::runtime_error("Cannot read fixture");
    if (file.peek() != std::ifstream::traits_type::eof()) throw std::runtime_error("Unexpected fixture size");
    return values;
}
}

int main(int argc, char** argv) {
    try {
        if (argc == 9 && std::string(argv[1]) == "preprocess") {
            const int ih = std::stoi(argv[3]), iw = std::stoi(argv[4]);
            if (ih < 1 || iw < 1) throw std::invalid_argument("Invalid image size");
            const auto stride = std::stoull(argv[5]);
            const auto bytes = read<std::uint8_t>(argv[2], rfdetr::checked_product(ih, stride));
            rfdetr::Preprocessor pre(std::stoi(argv[6]), std::stoi(argv[7]));
            const auto& output = pre.run({bytes.data(), ih, iw, stride});
            std::ofstream file(argv[8], std::ios::binary);
            file.write(reinterpret_cast<const char*>(output.data()), output.size() * sizeof(float));
            if (!file) throw std::runtime_error("Cannot write fixture");
        } else if (argc == 12 && std::string(argv[1]) == "decode") {
            const auto q = std::stoull(argv[4]), c = std::stoull(argv[5]);
            const auto boxes = read<float>(argv[2], rfdetr::checked_product(q, 4));
            const auto logits = read<float>(argv[3], rfdetr::checked_product(q, c));
            rfdetr::DecodeConfig cfg;
            if (std::string(argv[8]) != "none") cfg.background_index = std::stoi(argv[8]);
            cfg.top_k = std::stoull(argv[9]); cfg.threshold = std::stof(argv[10]);
            rfdetr::Decoder decoder(cfg);
            const auto& detections = decoder.run(boxes.data(), logits.data(), q, c, std::stoi(argv[6]), std::stoi(argv[7]));
            std::ofstream file(argv[11]); file << std::setprecision(9);
            for (const auto& detection : detections) {
                for (const auto x : detection.xyxy) file << x << ',';
                file << detection.score << ',' << detection.class_slot << ',' << detection.query << '\n';
            }
            if (!file) throw std::runtime_error("Cannot write fixture");
        } else if (argc == 11 && std::string(argv[1]) == "letterbox") {
            const int ih = std::stoi(argv[3]), iw = std::stoi(argv[4]);
            if (ih < 1 || iw < 1) throw std::invalid_argument("Invalid image size");
            const auto stride = std::stoull(argv[5]);
            const auto bytes = read<std::uint8_t>(argv[2], rfdetr::checked_product(ih, stride));
            rfdetr::LetterboxPreprocessor pre(std::stoi(argv[6]), std::stoi(argv[7]),
                                              static_cast<std::uint8_t>(std::stoi(argv[8])));
            const auto& output = pre.run({bytes.data(), ih, iw, stride});
            std::ofstream tensor(argv[9], std::ios::binary);
            tensor.write(reinterpret_cast<const char*>(output.data()), output.size() * sizeof(float));
            std::ofstream meta(argv[10]);
            meta << std::setprecision(9) << pre.meta().scale << ',' << pre.meta().pad_x << ',' << pre.meta().pad_y;
            if (!tensor || !meta) throw std::runtime_error("Cannot write fixture");
        } else if (argc == 15 && std::string(argv[1]) == "yolo-decode") {
            const auto channels = std::stoull(argv[3]), anchors = std::stoull(argv[4]);
            const auto values = read<float>(argv[2], rfdetr::checked_product(channels, anchors));
            const rfdetr::LetterboxMeta meta{static_cast<float>(std::stod(argv[7])), std::stoi(argv[8]),
                                             std::stoi(argv[9]), std::stoi(argv[5]), std::stoi(argv[6])};
            rfdetr::YoloDecoder decoder({std::stoull(argv[10]), std::stof(argv[11]), std::stof(argv[12]),
                                         std::stoull(argv[13])});
            const auto& detections = decoder.run(values.data(), channels, anchors, meta);
            std::ofstream file(argv[14]); file << std::setprecision(9);
            for (const auto& detection : detections) {
                for (const auto x : detection.xyxy) file << x << ',';
                file << detection.score << ',' << detection.class_slot << ',' << detection.query << '\n';
            }
            if (!file) throw std::runtime_error("Cannot write fixture");
        } else throw std::invalid_argument("Usage: adapter preprocess input IH IW stride H W output | decode boxes logits Q C IH IW background|none K threshold output | letterbox image IH IW stride H W pad tensor meta | yolo-decode output CHANNELS ANCHORS IH IW scale pad_x pad_y classes conf iou max_det out");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
