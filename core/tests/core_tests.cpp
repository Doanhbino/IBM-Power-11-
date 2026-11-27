#include "rfdetr/bounded_queue.hpp"
#include "rfdetr/decoder.hpp"
#include "rfdetr/frame_shard.hpp"
#include "rfdetr/gemm.hpp"
#include "rfdetr/preprocess.hpp"
#include "rfdetr/placement.hpp"
#include "rfdetr/yolo.hpp"
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <system_error>
#include <future>
#include <thread>
#include <iostream>
#include <limits>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template <typename F> void throws(F function) {
    bool caught = false;
    try { function(); } catch (const std::exception&) { caught = true; }
    require(caught, "Expected exception");
}
void queues() {
    rfdetr::BoundedQueue<int> live(2, rfdetr::QueuePolicy::drop_oldest);
    live.push(1); live.push(2); live.push(3);
    require(live.try_pop() == 2, "Live queue must evict oldest");
    require(live.try_pop() == 3, "Live order");
    live.close(); require(!live.push(4), "Closed queue accepted item");
    const auto stats = live.stats();
    require(stats.accepted == 3 && stats.dropped == 1 && stats.popped == 2 && stats.high_watermark == 2, "Queue accounting");
    rfdetr::BoundedQueue<int> replay(1, rfdetr::QueuePolicy::block);
    replay.push(1);
    auto producer = std::async(std::launch::async, [&] { return replay.push(2); });
    const auto blocked = producer.wait_for(std::chrono::milliseconds(20));
    replay.cancel(); // Always release producer, even when the assertion would fail.
    require(blocked == std::future_status::timeout, "Replay did not block");
    require(!producer.get(), "Cancel did not unblock producer");
    require(replay.stats().dropped == 1, "Cancel accounting");
    rfdetr::BoundedQueue<int> drain(1, rfdetr::QueuePolicy::block);
    drain.push(7); drain.close(); require(drain.try_pop() == 7, "close must drain");
}
void adapter() {
    std::uint8_t pixel[]{0, 0, 255};
    rfdetr::Preprocessor pre(1, 1, {{0, 0, 0}, {1, 1, 1}});
    const auto& x = pre.run({pixel, 1, 1, 3});
    require(x == std::vector<float>({1, 0, 0}), "BGR/RGB conversion");
    throws([&] { pre.run({pixel, 1, 2, 3}); });
    float boxes[]{.5f, .5f, .5f, .5f};
    float logits[]{5, 4, 10};
    rfdetr::Decoder decoder({-1, 2, .5f});
    const auto& d = decoder.run(boxes, logits, 1, 3, 100, 200);
    require(d.size() == 2 && d[0].class_slot == 0 && d[1].class_slot == 1, "Global top-k/background");
    require(d[0].xyxy == std::array<float, 4>({50, 25, 150, 75}), "cxcywh conversion");
    logits[0] = std::numeric_limits<float>::quiet_NaN();
    throws([&] { decoder.run(boxes, logits, 1, 3, 100, 200); });
    throws([] { rfdetr::Decoder invalid({std::nullopt, 0, .5f}); });
}
void memory_binding() {
    // The kernel reads maxnode-1 bits, so binding to a real node must work (regression: node+1 gave EINVAL).
    if (!std::filesystem::exists("/sys/devices/system/node/node0")) return;
    try { rfdetr::bind_memory(0); }
    catch (const std::system_error& error) {
        if (error.code().value() != EPERM && error.code().value() != ENOSYS) throw;  // sandbox forbids the syscall
    }
    throws([] { rfdetr::bind_memory(-1); });
}
void frame_shards() {
    // Shards of one camera must partition the frame sequence: every index goes to exactly one shard.
    for (std::size_t stride : {1, 2, 3, 5}) {
        std::vector<int> hits(60, 0);
        for (std::size_t offset = 0; offset < stride; ++offset) {
            const rfdetr::FrameShard shard(stride, offset);
            for (std::size_t index = 0; index < hits.size(); ++index) hits[index] += shard.selected(index);
        }
        for (int count : hits) require(count == 1, "Frame shards must partition the stream");
    }
    require(rfdetr::FrameShard{}.selected(7), "Default shard keeps every frame");
    require(rfdetr::FrameShard(3, 1).selected(4) && !rfdetr::FrameShard(3, 1).selected(5), "Shard offset");
    throws([] { rfdetr::FrameShard(0, 0); });
    throws([] { rfdetr::FrameShard(3, 3); });
}
void capture_binding() {
    // A capture thread must be able to move to CPUs the (narrowed) worker thread no longer sees,
    // as long as they were permitted before the worker narrowed its own mask.
    const auto permitted = rfdetr::allowed_cpus();
    if (permitted.size() < 2) return;
    std::exception_ptr failure;
    std::thread worker([&] {
        try {
            rfdetr::bind_cpus({permitted[0]});
            require(rfdetr::allowed_cpus() == std::vector<int>{permitted[0]}, "Worker mask narrowed");
            throws([&] { rfdetr::bind_cpus({permitted[1]}); });                 // legacy check sees only the narrowed mask
            std::thread capture([&] {
                try {
                    rfdetr::bind_cpus({permitted[1]}, permitted);
                    require(rfdetr::allowed_cpus() == std::vector<int>{permitted[1]}, "Capture mask applied");
                    throws([&] { rfdetr::bind_cpus({permitted.back() + 1}, permitted); });  // outside permitted set
                } catch (...) { failure = std::current_exception(); }
            });
            capture.join();
            require(rfdetr::allowed_cpus() == std::vector<int>{permitted[0]}, "Worker mask unchanged by capture thread");
        } catch (...) { if (!failure) failure = std::current_exception(); }
    });
    worker.join();
    if (failure) std::rethrow_exception(failure);
}
void yolo() {
    // 1x1 blue pixel (BGR) into 4x4: scale 4, no padding, RGB planes 0,0,1.
    std::uint8_t pixel[]{255, 0, 0};
    rfdetr::LetterboxPreprocessor tiny(4, 4);
    const auto& blue = tiny.run({pixel, 1, 1, 3});
    require(tiny.meta().scale == 4.f && tiny.meta().pad_x == 0 && tiny.meta().pad_y == 0, "1x1 letterbox meta");
    require(blue.size() == 48 && blue[0] == 0.f && blue[16] == 0.f && blue[32] == 1.f, "1x1 letterbox RGB planes");
    // 2x4 white image into 8x8: scale 2, content rows 2..5, pad value 114/255 elsewhere.
    std::vector<std::uint8_t> white(2 * 4 * 3, 255);
    rfdetr::LetterboxPreprocessor wide(8, 8);
    const auto& canvas = wide.run({white.data(), 2, 4, 12});
    require(wide.meta().scale == 2.f && wide.meta().pad_x == 0 && wide.meta().pad_y == 2, "wide letterbox meta");
    require(canvas[0] == 114.f / 255.f && canvas[2 * 8] == 1.f && canvas[6 * 8] == 114.f / 255.f, "wide letterbox rows");
    throws([&] { wide.run({white.data(), 2, 4, 5}); });
    throws([] { rfdetr::LetterboxPreprocessor invalid(0, 4); });

    // Layout is [channel][anchor]: cx, cy, w, h, score(class0), score(class1).
    const float out[]{50, 50, 50,  50, 50, 50,  20, 20, 20,  20, 20, 20,  .9f, .8f, 0,  0, 0, .7f};
    const rfdetr::LetterboxMeta identity{1.f, 0, 0, 100, 100};
    rfdetr::YoloDecoder decoder({2, .25f, .5f, 300});
    const auto& d = decoder.run(out, 6, 3, identity);
    require(d.size() == 2, "Per-class NMS keeps the other class and drops the duplicate");
    require(d[0].class_slot == 0 && d[0].query == 0 && std::abs(d[0].score - .9f) < 1e-6f, "Best detection");
    require(d[0].xyxy == std::array<float, 4>({40, 40, 60, 60}), "cxcywh to xyxy");
    require(d[1].class_slot == 1 && d[1].query == 2, "Other class survives");
    const float edge[]{95, 95, 20, 20, .9f, 0};
    const auto& clipped = decoder.run(edge, 6, 1, identity);
    require(clipped.size() == 1 && clipped[0].xyxy == std::array<float, 4>({85, 85, 100, 100}), "Clip to frame");
    const float inner[]{60, 70, 20, 10, .9f, 0};  // model-space xyxy = 50,65,70,75
    const auto& mapped = decoder.run(inner, 6, 1, {.5f, 10, 20, 200, 200});
    require(mapped.size() == 1 && mapped[0].xyxy == std::array<float, 4>({80, 90, 120, 110}), "Undo letterbox scale and padding");
    float bad[18]; std::copy(out, out + 18, bad);
    throws([&] { decoder.run(out, 7, 3, identity); });
    bad[12] = std::numeric_limits<float>::quiet_NaN();
    throws([&] { decoder.run(bad, 6, 3, identity); });
    throws([] { rfdetr::YoloDecoder invalid({2, 1.5f, .5f, 300}); });
    throws([] { rfdetr::YoloDecoder invalid({0, .25f, .5f, 300}); });
    throws([] { rfdetr::YoloDecoder invalid({2, .25f, .5f, 0}); });
}
void gemm(rfdetr::GemmBackend backend) {
    // Sizes cross the 8x16 tile edges and the 256-deep K block (REDP-5612 section 4.1).
    for (std::size_t m : {1, 3, 4, 5, 8, 9, 17}) for (std::size_t n : {1, 3, 4, 7, 16, 17, 33})
        for (std::size_t k : {1, 2, 5, 17, 256, 257, 300}) for (bool transposed : {false, true}) {
            std::vector<float> a(m * k), b(n * k), out(m * n + 4, 123.f);
            for (std::size_t i = 0; i < a.size(); ++i) a[i] = (static_cast<int>(i % 13) - 6) / 7.f;
            for (std::size_t i = 0; i < b.size(); ++i) b[i] = (static_cast<int>(i % 11) - 5) / 3.f;
            rfdetr::MatrixView bv{b.data(), k, n, transposed ? 1 : n, transposed ? k : 1};
            rfdetr::PackedRhs packed(bv);
            rfdetr::GemmWorkspace workspace;
            workspace.run({a.data(), m, k, k, 1}, packed, out.data(), backend);
            for (std::size_t i = 0; i < m; ++i) for (std::size_t j = 0; j < n; ++j) {
                double expected = 0;
                for (std::size_t p = 0; p < k; ++p)
                    expected += static_cast<double>(a[i * k + p]) * b[p * bv.row_stride + j * bv.col_stride];
                require(std::abs(out[i * n + j] - expected) < 1e-4 + 1e-4 * std::abs(expected), "GEMM numerical mismatch");
            }
            require(out[m * n] == 123.f, "GEMM tail wrote beyond output");
        }
}

void packed_alignment() {
    // 64-byte aligned panels let the MMA kernel use aligned vector (lxvp) loads.
    std::vector<float> b(40 * 37, 1.f);
    const rfdetr::PackedRhs rhs({b.data(), 40, 37, 37, 1});
    for (std::size_t column : {0, 16, 32})
        require(reinterpret_cast<std::uintptr_t>(rhs.panel(column)) % 64 == 0, "Packed panel alignment");
}
void shared_rhs(rfdetr::GemmBackend backend) {
    const float a[]{1, 4, 2, 5, 3, 6};  // Transposed storage of [[1,2,3],[4,5,6]].
    const float b[]{1, 0, 0, 1, 1, 1};
    const rfdetr::PackedRhs rhs({b, 3, 2, 2, 1});
    auto calculate = [&] {
        rfdetr::GemmWorkspace workspace;
        std::vector<float> out(4);
        workspace.run({a, 2, 3, 1, 2}, rhs, out.data(), backend);
        return out;
    };
    auto first = std::async(std::launch::async, calculate);
    auto second = std::async(std::launch::async, calculate);
    require(first.get() == std::vector<float>({4, 5, 10, 11}), "Transposed A/shared B correctness");
    require(second.get() == std::vector<float>({4, 5, 10, 11}), "Worker-local GEMM workspace");
}
}

int main() {
    try {
        queues(); adapter(); yolo(); frame_shards(); packed_alignment(); gemm(rfdetr::GemmBackend::reference); shared_rhs(rfdetr::GemmBackend::reference);
        require(!rfdetr::allowed_cpus().empty(), "Empty effective cpuset");
        throws([] { rfdetr::bind_cpus({-1}); });
        memory_binding(); capture_binding();
        const auto capability = rfdetr::mma_capability();
        if (capability.compiled && capability.exposed) {
            gemm(rfdetr::GemmBackend::mma); shared_rhs(rfdetr::GemmBackend::mma);
        }
        else {
            float a[]{1}; rfdetr::PackedRhs b({a, 1, 1, 1, 1}); rfdetr::GemmWorkspace workspace;
            throws([&] { workspace.run({a, 1, 1, 1, 1}, b, a, rfdetr::GemmBackend::mma); });
        }
        std::cout << "core tests passed; MMA execution " << (capability.compiled && capability.exposed ? "tested" : "unavailable") << '\n';
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
