#include "rfdetr/gemm.hpp"
#include "rfdetr/types.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

namespace {
std::size_t positive(const char* value) {
    std::string text(value);
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("Expected positive integer");
    const auto n = std::stoull(text);
    if (n == 0) throw std::invalid_argument("Expected positive integer");
    return n;
}
double milliseconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--capability") {
            const auto c = rfdetr::mma_capability();
            std::cout << "{\"compiled\":" << (c.compiled ? "true" : "false")
                      << ",\"exposed\":" << (c.exposed ? "true" : "false") << "}\n";
            return 0;
        }
        if (argc < 5 || argc > 7)
            throw std::invalid_argument("Usage: rfdetr_gemm M N K reference|mma [iterations=20] [transpose_b=0|1]");
        const auto m = positive(argv[1]), n = positive(argv[2]), k = positive(argv[3]);
        const auto name = std::string(argv[4]);
        if (name != "reference" && name != "mma") throw std::invalid_argument("Unknown GEMM backend");
        const auto backend = name == "mma" ? rfdetr::GemmBackend::mma : rfdetr::GemmBackend::reference;
        const auto iterations = argc > 5 ? positive(argv[5]) : 20;
        const bool transposed = argc > 6 && std::string(argv[6]) == "1";
        if (argc > 6 && std::string(argv[6]) != "0" && !transposed)
            throw std::invalid_argument("transpose_b must be 0 or 1");
        std::vector<float> a(rfdetr::checked_product(m, k)), b(rfdetr::checked_product(k, n));
        std::vector<float> c(rfdetr::checked_product(m, n));
        std::mt19937 generator(42);
        std::uniform_real_distribution<float> distribution(-1.f, 1.f);
        for (auto& v : a) v = distribution(generator);
        for (auto& v : b) v = distribution(generator);
        const rfdetr::MatrixView bv{b.data(), k, n, transposed ? 1 : n, transposed ? k : 1};
        auto start = std::chrono::steady_clock::now();
        const rfdetr::PackedRhs packed(bv);
        const double packing_ms = milliseconds(start);
        rfdetr::GemmWorkspace workspace;
        workspace.run({a.data(), m, k, k, 1}, packed, c.data(), backend);
        double max_abs = 0, max_scaled = 0;
        bool correct = true;
        // Independent FP64 oracle, including odd tails and transposed storage.
        for (std::size_t i = 0; i < m; ++i) for (std::size_t j = 0; j < n; ++j) {
            double expected = 0;
            for (std::size_t p = 0; p < k; ++p)
                expected += static_cast<double>(a[i * k + p]) * b[p * bv.row_stride + j * bv.col_stride];
            const double error = std::abs(c[i * n + j] - expected);
            max_abs = std::max(max_abs, error);
            max_scaled = std::max(max_scaled, error / (1e-4 + 1e-3 * std::abs(expected)));
            correct = correct && std::isfinite(c[i * n + j]) && error <= 1e-4 + 1e-3 * std::abs(expected);
        }
        if (!correct) throw std::runtime_error("GEMM numerical gate failed against FP64 oracle");
        std::vector<double> samples;
        for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
            start = std::chrono::steady_clock::now();
            workspace.run({a.data(), m, k, k, 1}, packed, c.data(), backend);
            samples.push_back(milliseconds(start));
        }
        std::cout << std::setprecision(10) << "{\"schema_version\":1,\"m\":" << m << ",\"n\":" << n
                  << ",\"k\":" << k << ",\"transpose_b\":" << (transposed ? "true" : "false")
                  << ",\"backend\":\"" << rfdetr::backend_name(backend) << "\",\"correct\":true"
                  << ",\"max_abs_error\":" << max_abs << ",\"max_scaled_error\":" << max_scaled
                  << ",\"pack_b_ms\":" << packing_ms << ",\"samples_ms\":[";
        for (std::size_t i = 0; i < samples.size(); ++i) std::cout << (i ? "," : "") << samples[i];
        std::cout << "],\"scope\":\"microbenchmark: reusable packed B; timer includes A packing, kernel and output copy; not RF-DETR hot-path proof\"}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
