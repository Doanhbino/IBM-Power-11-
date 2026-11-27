#pragma once

#include <vector>

namespace rfdetr {

std::vector<int> allowed_cpus();
// Apply BEFORE constructing sessions, buffers or capture threads. Throws on failure.
void bind_cpus(const std::vector<int>& cpus);
// Same, but validated against `permitted` instead of the calling thread's current mask. A worker narrows its own
// mask first, so a capture thread needs the set that was effective before that (captured earlier by the caller).
void bind_cpus(const std::vector<int>& cpus, const std::vector<int>& permitted);
void bind_memory(int numa_node);

}  // namespace rfdetr
