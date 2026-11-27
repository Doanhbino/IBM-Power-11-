#include "rfdetr/placement.hpp"
#include <algorithm>
#include <cerrno>
#include <memory>
#include <stdexcept>
#include <system_error>
#ifdef __linux__
#include <linux/mempolicy.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace rfdetr {
#ifdef __linux__
namespace {
struct CpuMaskDeleter { void operator()(cpu_set_t* mask) const noexcept { CPU_FREE(mask); } };
}
#endif

std::vector<int> allowed_cpus() {
#ifdef __linux__
    for (int count = 128; count <= (1 << 20); count *= 2) {
        const auto bytes = CPU_ALLOC_SIZE(count);
        std::unique_ptr<cpu_set_t, CpuMaskDeleter> mask(CPU_ALLOC(count));
        if (!mask) throw std::bad_alloc();
        CPU_ZERO_S(bytes, mask.get());
        if (sched_getaffinity(0, bytes, mask.get()) == 0) {
            std::vector<int> result;
            for (int cpu = 0; cpu < count; ++cpu)
                if (CPU_ISSET_S(cpu, bytes, mask.get())) result.push_back(cpu);
            return result;
        }
        if (errno != EINVAL) throw std::system_error(errno, std::generic_category(), "sched_getaffinity");
    }
    throw std::runtime_error("CPU mask exceeds supported size");
#else
    throw std::runtime_error("CPU placement requires Linux");
#endif
}

void bind_cpus(const std::vector<int>& cpus) {
    if (cpus.empty()) throw std::invalid_argument("CPU mask must not be empty");
    const auto allowed = allowed_cpus();
    for (int cpu : cpus)
        if (!std::binary_search(allowed.begin(), allowed.end(), cpu))
            throw std::invalid_argument("Requested CPU outside effective affinity");
#ifdef __linux__
    const int count = std::max(cpus.back(), *std::max_element(cpus.begin(), cpus.end())) + 1;
    const auto bytes = CPU_ALLOC_SIZE(count);
    std::unique_ptr<cpu_set_t, CpuMaskDeleter> mask(CPU_ALLOC(count));
    if (!mask) throw std::bad_alloc();
    CPU_ZERO_S(bytes, mask.get());
    for (int cpu : cpus) CPU_SET_S(cpu, bytes, mask.get());
    if (sched_setaffinity(0, bytes, mask.get()))
        throw std::system_error(errno, std::generic_category(), "sched_setaffinity");
    auto expected = cpus;
    std::sort(expected.begin(), expected.end());
    expected.erase(std::unique(expected.begin(), expected.end()), expected.end());
    if (allowed_cpus() != expected) throw std::runtime_error("Effective affinity differs from request");
#endif
}

void bind_memory(int node) {
    if (node < 0 || node > 65535) throw std::invalid_argument("Invalid NUMA node");
#ifdef __linux__
    constexpr auto bits = sizeof(unsigned long) * 8;
    // The kernel reads maxnode-1 bits (libnuma also passes size+1), so node N needs maxnode = N + 2,
    // and the mask must be long enough for that many bits.
    const auto maxnode = static_cast<unsigned long>(node) + 2;
    std::vector<unsigned long> mask(maxnode / bits + 1, 0);
    mask[node / bits] |= 1UL << (node % bits);
    if (syscall(SYS_set_mempolicy, MPOL_BIND, mask.data(), maxnode))
        throw std::system_error(errno, std::generic_category(), "set_mempolicy MPOL_BIND");
#else
    throw std::runtime_error("NUMA memory policy requires Linux");
#endif
}

}  // namespace rfdetr
