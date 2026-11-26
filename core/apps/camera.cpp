#include "rfdetr/camera.hpp"
#include <opencv2/core/utils/logger.hpp>
#include <iostream>
#include <stdexcept>
#include <csignal>

namespace {
static_assert(std::atomic<bool>::is_always_lock_free, "Signal flag must be lock-free");
std::atomic<bool> interrupted{false};
void on_signal(int) { interrupted.store(true, std::memory_order_relaxed); }
}

int main(int argc, char** argv) {
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    try {
        if (argc != 2) throw std::invalid_argument("Usage: rfdetr_camera worker-config.json");
        return rfdetr::run_camera_worker(argv[1], interrupted);
    } catch (...) {
        // Upstream exceptions may embed source URLs. Detailed sources stay private.
        std::cerr << "Camera worker failed: inspect model contract, placement, inputs and output permissions. Source details omitted.\n";
        return 2;
    }
}
