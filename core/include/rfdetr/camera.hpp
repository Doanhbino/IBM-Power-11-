#pragma once

#include <atomic>
#include <filesystem>

namespace rfdetr {

// Placement precedes inference/capture allocation. All capture threads are joined.
int run_camera_worker(const std::filesystem::path& configuration,
                      const std::atomic<bool>& stop_signal);

}  // namespace rfdetr
