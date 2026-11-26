#pragma once

#include <vector>

namespace rfdetr {

std::vector<int> allowed_cpus();
// Apply BEFORE constructing sessions, buffers or capture threads. Throws on failure.
void bind_cpus(const std::vector<int>& cpus);
void bind_memory(int numa_node);

}  // namespace rfdetr
