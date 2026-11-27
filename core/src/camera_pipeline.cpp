#include "rfdetr/bounded_queue.hpp"
#include "rfdetr/camera.hpp"
#include "rfdetr/decoder.hpp"
#include "rfdetr/frame_shard.hpp"
#include "rfdetr/ort_session.hpp"
#include "rfdetr/placement.hpp"
#include "rfdetr/preprocess.hpp"
#include "rfdetr/yolo.hpp"
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

namespace {
using Clock = std::chrono::steady_clock;
double ms(Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }
double stamp(Clock::time_point t) { return std::chrono::duration<double>(t.time_since_epoch()).count(); }

int integer(const cv::FileNode& node, const char* key, int fallback, int minimum = 0) {
    const auto value = node[key];
    if (value.empty()) return fallback;
    if (!value.isInt()) throw std::invalid_argument(std::string("Expected integer: ") + key);
    const int result = static_cast<int>(value);
    if (result < minimum) throw std::invalid_argument(std::string("Invalid range: ") + key);
    return result;
}
std::string text(const cv::FileNode& node, const char* key) {
    if (!node[key].isString()) throw std::invalid_argument(std::string("Expected string: ") + key);
    return static_cast<std::string>(node[key]);
}

float number(const cv::FileNode& node, const char* key) {
    if (!node[key].isReal() && !node[key].isInt()) throw std::invalid_argument(std::string("Explicit number required: ") + key);
    return static_cast<float>(static_cast<double>(node[key]));
}

// One model's preprocess -> infer -> postprocess, so the worker loop is model-agnostic and keeps three timed stages.
struct ModelPath {
    explicit ModelPath(const rfdetr::OrtConfig& config) : ort(config) {}
    virtual ~ModelPath() = default;
    virtual const std::vector<float>& preprocess(rfdetr::ImageView image) = 0;
    virtual void infer(const float* tensor, std::size_t elements) = 0;
    virtual const std::vector<rfdetr::Detection>& postprocess(int source_height, int source_width) = 0;
    rfdetr::OrtSession ort;
};
struct RfdetrPath final : ModelPath {
    RfdetrPath(const rfdetr::OrtConfig& config, rfdetr::Normalization normalization, rfdetr::DecodeConfig decode)
        : ModelPath(config), pre(config.height, config.width, normalization), decoder(decode) {}
    const std::vector<float>& preprocess(rfdetr::ImageView image) override { return pre.run(image); }
    void infer(const float* tensor, std::size_t elements) override { out = ort.run(tensor, elements); }
    const std::vector<rfdetr::Detection>& postprocess(int height, int width) override {
        return decoder.run(out.boxes, out.logits, out.queries, out.classes, height, width);
    }
    rfdetr::Preprocessor pre;
    rfdetr::Decoder decoder;
    rfdetr::OutputView out{};
};
struct YoloPath final : ModelPath {
    YoloPath(const rfdetr::OrtConfig& config, std::uint8_t pad_value, rfdetr::YoloConfig yolo)
        : ModelPath(config), pre(config.height, config.width, pad_value), decoder(yolo) {}
    const std::vector<float>& preprocess(rfdetr::ImageView image) override { return pre.run(image); }
    void infer(const float* tensor, std::size_t elements) override { out = ort.run_yolo(tensor, elements); }
    const std::vector<rfdetr::Detection>& postprocess(int, int) override {
        return decoder.run(out.data, out.channels, out.anchors, pre.meta());
    }
    rfdetr::LetterboxPreprocessor pre;
    rfdetr::YoloDecoder decoder;
    rfdetr::YoloOutputView out{};
};

struct Frame {
    cv::Mat image;
    std::size_t id;
    Clock::time_point read_begin, received;
    double pts_ms;
};
struct Camera {
    std::string id, source;
    rfdetr::BoundedQueue<Frame> queue;
    std::thread capture;
    std::atomic<std::size_t> received{0}, errors{0}, reconnects{0};
    std::size_t processed{0}, expired{0};
    rfdetr::FrameShard shard;  // keeps every stride-th frame of this camera; default keeps all
    std::size_t grabbed{0};    // frames read from the stream, including those another shard owns (capture thread only)
    Camera(std::string name, std::string path, std::size_t capacity, bool live)
        : id(std::move(name)), source(std::move(path)),
          queue(capacity, live ? rfdetr::QueuePolicy::drop_oldest : rfdetr::QueuePolicy::block) {}
    ~Camera() { queue.cancel(); if (capture.joinable()) capture.join(); }
};

}  // namespace

int rfdetr::run_camera_worker(const std::filesystem::path& config_path,
                              const std::atomic<bool>& interrupted) {
    cv::FileStorage file(config_path.string(), cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!file.isOpened()) throw std::runtime_error("Cannot read worker configuration");
    const auto cfg = file.root();
    // Captured before the worker narrows its own mask; capture threads may use any of these CPUs.
    const auto permitted_cpus = rfdetr::allowed_cpus();
    std::vector<int> capture_cpus;
    if (!cfg["capture_cpus"].empty()) {
        for (const auto& cpu : cfg["capture_cpus"]) capture_cpus.push_back(static_cast<int>(cpu));
        if (capture_cpus.empty()) throw std::invalid_argument("capture_cpus must not be empty when present");
    }
    if (!cfg["cpus"].empty()) {
        std::vector<int> cpus;
        for (const auto& cpu : cfg["cpus"]) cpus.push_back(static_cast<int>(cpu));
        rfdetr::bind_cpus(cpus);
    }
    if (!cfg["numa_node"].empty()) rfdetr::bind_memory(integer(cfg, "numa_node", 0));
    const auto mode = text(cfg, "mode");
    if (mode != "replay" && mode != "live") throw std::invalid_argument("mode must be replay or live");
    const bool live = mode == "live";
    const auto out = std::filesystem::path(text(cfg, "output"));
    if (!std::filesystem::create_directory(out)) throw std::runtime_error("Output directory must be new");
    const int height = integer(cfg, "height", 0, 1), width = integer(cfg, "width", 0, 1);
    const int threads = integer(cfg, "threads", 1, 1);
    if (static_cast<std::size_t>(threads) > rfdetr::allowed_cpus().size())
        throw std::invalid_argument("ORT threads exceed effective CPU mask");
    cv::setNumThreads(1);
    const std::string model_type = cfg["model_type"].empty() ? std::string("rfdetr") : text(cfg, "model_type");
    if (model_type != "rfdetr" && model_type != "yolo11") throw std::invalid_argument("model_type must be rfdetr or yolo11");
    const bool profiling = integer(cfg, "profile", 0) == 1;
    rfdetr::OrtConfig ort_config;
    ort_config.model = text(cfg, "model");
    ort_config.input_name = text(cfg, "input_name");
    ort_config.profile_prefix = profiling ? (out / "ort").string() : "";
    ort_config.height = height;
    ort_config.width = width;
    ort_config.threads = threads;
    ort_config.allow_spinning = integer(cfg, "allow_spinning", 0) == 1;
    if (!cfg["custom_ops_library"].empty()) ort_config.custom_ops_library = text(cfg, "custom_ops_library");
    ort_config.model_type = model_type;
    rfdetr::Normalization normalization;
    rfdetr::DecodeConfig decode;
    rfdetr::YoloConfig yolo{};
    std::uint8_t pad_value = 114;
    if (model_type == "rfdetr") {
        ort_config.boxes_name = text(cfg, "boxes_output");
        ort_config.logits_name = text(cfg, "logits_output");
        for (const auto key : {"mean", "std"}) if (!cfg[key].empty()) {
            if (!cfg[key].isSeq() || cfg[key].size() != 3) throw std::invalid_argument("Expected 3 normalization values");
            auto& values = std::string(key) == "mean" ? normalization.mean : normalization.std;
            for (int i = 0; i < 3; ++i) values[i] = static_cast<float>(cfg[key][i]);
        }
        if (cfg["background_index"].isInt()) decode.background_index = static_cast<int>(cfg["background_index"]);
        else if (!cfg["background_index"].isString() || text(cfg, "background_index") != "none")
            throw std::invalid_argument("Set background_index explicitly to integer or string none");
        decode.top_k = integer(cfg, "top_k", 300, 1);
        if (!cfg["threshold"].empty()) decode.threshold = static_cast<float>(cfg["threshold"]);
    } else {
        ort_config.output_name = text(cfg, "output_name");
        if (cfg["num_classes"].empty() || cfg["max_det"].empty())
            throw std::invalid_argument("Explicit num_classes and max_det required for yolo11");
        yolo = rfdetr::YoloConfig{static_cast<std::size_t>(integer(cfg, "num_classes", 0, 1)), number(cfg, "conf_threshold"),
                                  number(cfg, "iou_threshold"), static_cast<std::size_t>(integer(cfg, "max_det", 0, 1))};
        ort_config.classes = yolo.classes;
        const int pad = integer(cfg, "pad_value", 114, 0);
        if (pad > 255) throw std::invalid_argument("Invalid range: pad_value");
        pad_value = static_cast<std::uint8_t>(pad);
    }
    const auto init_start = Clock::now();
    std::unique_ptr<ModelPath> model;
    if (model_type == "rfdetr") model = std::make_unique<RfdetrPath>(ort_config, normalization, decode);
    else model = std::make_unique<YoloPath>(ort_config, pad_value, yolo);
    const double init_ms = ms(Clock::now() - init_start);
    cv::Mat black(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    const auto& tensor = model->preprocess({black.data, black.rows, black.cols, black.step});
    const auto warm_start = Clock::now();
    for (int i = 0; i < integer(cfg, "warmup", 10); ++i) model->infer(tensor.data(), tensor.size());
    const double warmup_ms = ms(Clock::now() - warm_start);
    const int capacity = integer(cfg, "queue_capacity", 2, 1);
    const int timeout = integer(cfg, "capture_timeout_ms", 5000, 1);
    const int retry_limit = integer(cfg, "reconnect_attempts", 3);
    const int retry_delay = integer(cfg, "reconnect_delay_ms", 1000);
    const int max_frames = integer(cfg, "max_frames_per_camera", 0);
    const double duration = cfg["duration_s"].empty() ? 0.0 : static_cast<double>(cfg["duration_s"]);
    const double max_age = cfg["max_frame_age_ms"].empty() ? 0.0 : static_cast<double>(cfg["max_frame_age_ms"]);
    if (!std::isfinite(duration) || duration < 0 || !std::isfinite(max_age) || max_age < 0 ||
        (live && duration <= 0)) throw std::invalid_argument("Live mode requires positive duration_s; invalid duration/age");
    if (!live && max_age > 0) throw std::invalid_argument("Replay must preserve frames; disable max_frame_age_ms");
    std::vector<std::unique_ptr<Camera>> cameras;
    for (const auto& source : cfg["sources"]) {
        const auto id = text(source, "id");
        if (id.empty() || id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.") != std::string::npos)
            throw std::invalid_argument("Camera ID must be a safe identifier");
        for (const auto& camera : cameras) if (camera->id == id) throw std::invalid_argument("Duplicate camera ID");
        cameras.push_back(std::make_unique<Camera>(id, text(source, "source"), capacity, live));
        if (!source["frame_stride"].empty() || !source["frame_offset"].empty()) {
            if (source["frame_stride"].empty() || source["frame_offset"].empty())
                throw std::invalid_argument("frame_stride and frame_offset must be set together");
            cameras.back()->shard = rfdetr::FrameShard(static_cast<std::size_t>(integer(source, "frame_stride", 1, 1)),
                                                       static_cast<std::size_t>(integer(source, "frame_offset", 0, 0)));
        }
    }
    if (cameras.empty()) throw std::invalid_argument("No cameras configured");
    std::ofstream samples(out / "samples.csv"), detections(out / "detections.jsonl");
    if (!samples || !detections) throw std::runtime_error("Cannot open output streams");
    samples << "camera_id,frame_id,source_pts_ms,received_monotonic_s,completed_monotonic_s,capture_ms,queue_age_ms,preprocess_ms,inference_ms,postprocess_ms,receive_to_result_ms,read_to_result_ms,detections\n";
    samples << std::setprecision(12); detections << std::setprecision(9);
    std::atomic<bool> stop{false}, failed{false};
    std::mutex wake_mutex;
    std::condition_variable wake;
    std::size_t generation = 0;
    const auto started = Clock::now();
    const auto notify = [&] { std::lock_guard<std::mutex> lock(wake_mutex); ++generation; wake.notify_one(); };
    const auto cancelled = [&] { return stop.load() || interrupted ||
        (duration > 0 && ms(Clock::now() - started) >= duration * 1000.0); };
    auto finish = [&] {
        stop = true;
        for (auto& camera : cameras) camera->queue.cancel();
        for (auto& camera : cameras) if (camera->capture.joinable()) camera->capture.join();
    };
    Clock::time_point last_completed = started;
    try {
        for (auto& camera : cameras) camera->capture = std::thread([&, camera = camera.get()] {
            try {
                // Decode (and any FFmpeg threads spawned from here) stays off the inference CPUs when requested.
                if (!capture_cpus.empty()) rfdetr::bind_cpus(capture_cpus, permitted_cpus);
                cv::VideoCapture capture;
                int retries = 0;
                while (!cancelled()) {
                    if (!capture.isOpened()) {
                        const bool remote = camera->source.find("://") != std::string::npos;
                        const bool opened = remote ? capture.open(camera->source, cv::CAP_FFMPEG,
                            {cv::CAP_PROP_OPEN_TIMEOUT_MSEC, timeout, cv::CAP_PROP_READ_TIMEOUT_MSEC, timeout})
                            : capture.open(camera->source);
                        if (!opened) {
                            ++camera->errors;
                            if (!live || retries++ >= retry_limit) { failed = true; break; }
                        }
                    }
                    if (capture.isOpened()) {
                        cv::Mat image;
                        const auto begin = Clock::now();
                        // grab() decodes; retrieve() converts to BGR. Frames owned by another shard skip the conversion.
                        bool ok = capture.grab();
                        const auto index = ok ? camera->grabbed++ : camera->grabbed;
                        if (ok && !camera->shard.selected(index)) continue;
                        if (ok) ok = capture.retrieve(image);
                        const auto received = Clock::now();
                        if (ok && !image.empty()) {
                            if (image.type() != CV_8UC3) throw std::runtime_error("Expected BGR uint8 decoded frame");
                            camera->received.fetch_add(1);
                            const auto id = index;  // source frame index: unique across the shards of one camera
                            const double pts = capture.get(cv::CAP_PROP_POS_MSEC);
                            if (!camera->queue.push({std::move(image), id, begin, received, std::isfinite(pts) ? pts : 0.0})) break;
                            notify();
                            retries = 0;
                            if (max_frames > 0 && camera->received >= static_cast<std::size_t>(max_frames)) break;
                            continue;
                        }
                        if (!live) break; // Replay EOF. Backend cannot distinguish corrupt EOF; validate frame IDs externally.
                        ++camera->errors;
                        capture.release();
                        if (retries++ >= retry_limit) { failed = true; break; }
                    }
                    ++camera->reconnects;
                    for (int elapsed = 0; elapsed < retry_delay && !cancelled(); elapsed += 20)
                        std::this_thread::sleep_for(std::chrono::milliseconds(std::min(20, retry_delay - elapsed)));
                }
                capture.release();
            } catch (...) { ++camera->errors; failed = true; }
            camera->queue.close(); notify();
        });
        std::size_t next = 0;
        while (!interrupted && !failed) {
            std::size_t observed;
            { std::lock_guard<std::mutex> lock(wake_mutex); observed = generation; }
            if (duration > 0 && ms(Clock::now() - started) >= duration * 1000.0) break;
            bool progressed = false, done = true;
            for (std::size_t offset = 0; offset < cameras.size(); ++offset) {
                const auto index = (next + offset) % cameras.size();
                auto& camera = *cameras[index];
                auto frame = camera.queue.try_pop();
                const auto state = camera.queue.stats();
                done = done && state.closed && state.pending == 0 && !frame;
                if (!frame) continue;
                progressed = true; next = (index + 1) % cameras.size();
                const auto begin = Clock::now();
                const double age = ms(begin - frame->received);
                if (max_age > 0 && age > max_age) { ++camera.expired; break; }
                const auto& data = model->preprocess({frame->image.data, frame->image.rows, frame->image.cols, frame->image.step});
                const auto pre_end = Clock::now();
                model->infer(data.data(), data.size());
                const auto inf_end = Clock::now();
                const auto& decoded = model->postprocess(frame->image.rows, frame->image.cols);
                const auto end = Clock::now(); last_completed = end;
                ++camera.processed;
                samples << camera.id << ',' << frame->id << ',' << frame->pts_ms << ',' << stamp(frame->received) << ',' << stamp(end)
                    << ',' << ms(frame->received - frame->read_begin) << ',' << age << ',' << ms(pre_end - begin)
                    << ',' << ms(inf_end - pre_end) << ',' << ms(end - inf_end) << ',' << ms(end - frame->received)
                    << ',' << ms(end - frame->read_begin) << ',' << decoded.size() << '\n';
                detections << "{\"camera_id\":\"" << camera.id << "\",\"frame_id\":" << frame->id << ",\"detections\":[";
                for (std::size_t i = 0; i < decoded.size(); ++i) {
                    const auto& detection = decoded[i];
                    if (i) detections << ',';
                    detections << "{\"xyxy\":[" << detection.xyxy[0] << ',' << detection.xyxy[1] << ',' << detection.xyxy[2]
                        << ',' << detection.xyxy[3] << "],\"score\":" << detection.score << ",\"class_slot\":" << detection.class_slot << '}';
                }
                detections << "]}\n";
                if (!samples || !detections) throw std::runtime_error("Output write failed");
                break; // One frame per scheduling turn; round-robin fairness.
            }
            if (done) break;
            if (!progressed) {
                std::unique_lock<std::mutex> lock(wake_mutex);
                wake.wait_for(lock, std::chrono::milliseconds(20), [&] { return generation != observed; });
            }
        }
        finish();
    } catch (...) { finish(); throw; }
    samples.close(); detections.close();
    if (!samples || !detections) throw std::runtime_error("Output flush failed");
    const double active_s = std::chrono::duration<double>(last_completed - started).count();
    cv::FileStorage result((out / "result.json").string(), cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
    result << "schema_version" << 1 << "status" << (failed ? "failed" : interrupted ? "interrupted" : "ok")
        << "backend" << "ORT CPU / linked MLAS; MMA usage requires independent hot-path evidence"
        << "model_type" << model_type << "input_height" << height << "input_width" << width
        << "ort_version" << rfdetr::OrtSession::version() << "runtime_library" << rfdetr::OrtSession::library_path()
        << "session_init_ms" << init_ms
        << "warmup_ms" << warmup_ms << "profile_instrumented" << static_cast<int>(profiling)
        << "active_start_monotonic_s" << stamp(started) << "active_end_monotonic_s" << stamp(last_completed)
        << "active_elapsed_s" << active_s << "shutdown_ms" << ms(Clock::now() - last_completed)
        << "profile_path" << model->ort.end_profile() << "cameras" << "[";
    std::size_t processed = 0;
    for (const auto& camera : cameras) {
        const auto state = camera->queue.stats();
        processed += camera->processed;
        result << "{" << "id" << camera->id << "frame_stride" << static_cast<int>(camera->shard.stride)
            << "frame_offset" << static_cast<int>(camera->shard.offset) << "received" << static_cast<double>(camera->received)
            << "processed" << static_cast<double>(camera->processed) << "queue_dropped" << static_cast<double>(state.dropped)
            << "shutdown_rejected" << static_cast<double>(camera->received - state.accepted)
            << "expired" << static_cast<double>(camera->expired) << "errors" << static_cast<double>(camera->errors)
            << "reconnects" << static_cast<double>(camera->reconnects) << "high_watermark" << static_cast<int>(state.high_watermark) << "}";
    }
    result << "]" << "images" << static_cast<double>(processed) << "throughput_fps" << (active_s > 0 ? processed / active_s : 0.0)
        << "timing_scope" << "capture threads start to last decoded result; warmup excluded; includes capture startup and output IO; not sensor-to-result or steady-state"
        << "queue_age_scope" << "local monotonic time after read to preprocessing start; source PTS is not synchronized sensor time";
    result.release();
    return failed || processed == 0 ? 2 : interrupted ? 130 : 0;
}
