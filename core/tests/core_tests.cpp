#include "rfdetr/bounded_queue.hpp"
#include "rfdetr/decoder.hpp"
#include "rfdetr/gemm.hpp"
#include "rfdetr/preprocess.hpp"
#include "rfdetr/placement.hpp"
#include <chrono>
#include <cmath>
#include <future>
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
void gemm(rfdetr::GemmBackend backend) {
    for (std::size_t m : {1, 3, 4, 5, 9}) for (std::size_t n : {1, 3, 4, 7})
        for (std::size_t k : {1, 2, 5, 17}) for (bool transposed : {false, true}) {
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
        queues(); adapter(); gemm(rfdetr::GemmBackend::reference); shared_rhs(rfdetr::GemmBackend::reference);
        require(!rfdetr::allowed_cpus().empty(), "Empty effective cpuset");
        throws([] { rfdetr::bind_cpus({-1}); });
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
