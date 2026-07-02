#include "asterion/core.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace {
bool power_two(std::uint64_t value) { return value && !(value & (value - 1)); }
bool finite_positive(double value) { return std::isfinite(value) && value > 0; }
bool checked_tensor(std::uint64_t a, std::uint64_t b) {
    return a <= static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) / b;
}
int tail(std::span<const double> values, std::span<const double> weights,
         double quantile, double& result) {
    if (values.empty() || values.size() != weights.size() || !std::isfinite(quantile) ||
        quantile < 0 || quantile >= 1) return 1;
    std::vector<std::pair<double,double>> sorted;
    sorted.reserve(values.size());
    double total = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i]) || !finite_positive(weights[i])) return 1;
        sorted.emplace_back(values[i], weights[i]);
        total += weights[i];
    }
    if (!finite_positive(total)) return 1;
    std::sort(sorted.begin(), sorted.end(), std::greater<>());
    const double mass = (1 - quantile) * total;
    double remaining = mass;
    double cost = 0;
    for (const auto& [value,weight] : sorted) {
        const double take = std::min(weight, remaining);
        cost += value * take;
        remaining -= take;
        if (remaining <= total * 1e-14) break;
    }
    result = cost / mass;
    return std::isfinite(result) ? 0 : 1;
}
} // namespace

extern "C" std::uint32_t asterion_abi_version(void) { return 2; }

extern "C" int asterion_verify_schedule(std::uint32_t operation, const std::uint64_t* shape,
                                      std::size_t rank, const std::uint64_t* fields,
                                      std::size_t field_count) {
    if (!shape || !fields || field_count != 10 || operation > 4) return 1;
    const std::size_t expected_rank = operation == 0 ? 1 : operation == 4 ? 3 : 2;
    if (rank != expected_rank) return 1;
    for (std::size_t i = 0; i < rank; ++i)
        if (!shape[i] || shape[i] > (1ULL << 24)) return 1;
    if (operation == 4) {
        if (!checked_tensor(shape[0],shape[2]) || !checked_tensor(shape[2],shape[1]) ||
            !checked_tensor(shape[0],shape[1])) return 1;
    } else if (rank == 2 && !checked_tensor(shape[0],shape[1])) return 1;
    for (std::size_t i = 0; i < 7; ++i) if (!power_two(fields[i])) return 1;
    const auto block=fields[0], warps=fields[1], rows=fields[2];
    if (block < 32 || block > 4096 || warps > 8 || rows > 8) return 1;
    for (std::size_t i=3; i<6; ++i) if (fields[i] < 16 || fields[i] > 64) return 1;
    if (fields[6] > 4 || fields[7] > 1 || fields[8] != 1) return 1;
    if (operation != 4 && fields[6] != 1) return 1;
    if (fields[9] && (!power_two(fields[9]) || fields[9] > 16384)) return 1;
    if (operation >= 1 && operation <= 3) {
        std::uint64_t padded = 1;
        while (padded < shape[1]) padded *= 2;
        if (fields[9]) padded = fields[9];
        if (shape[1] > padded) return 1;
        if (padded > 16384 || rows*padded > 65536 || fields[7] == 0) return 1;
    }
    if (operation == 4 && fields[7] == 0) return 1;
    return 0;
}

extern "C" int asterion_cvar(const double* values, const double* weights, std::size_t count,
                            double quantile, double* output) {
    if (!values || !weights || !output || !count) return 1;
    try {
        double value=0;
        int status=tail({values,count},{weights,count},quantile,value);
        if (!status) *output=value;
        return status;
    } catch (...) { return 2; } // Never allow a C++ exception across the ABI.
}

extern "C" int asterion_objective(const double* ratios, const double* weights, std::size_t count,
                                 double risk, double quantile, double* output) {
    if (!ratios || !weights || !output || !count || !std::isfinite(risk) || risk < 0) return 1;
    try {
        std::vector<double> logs;
        logs.reserve(count);
        double sum=0, total=0;
        for (std::size_t i=0; i<count; ++i) {
            if (!finite_positive(ratios[i]) || !finite_positive(weights[i])) return 1;
            const double value=std::log(ratios[i]);
            logs.push_back(value);
            sum += value*weights[i];
            total += weights[i];
        }
        double upper=0;
        int status=tail(logs,{weights,count},quantile,upper);
        if (status) return status;
        double value=sum/total+risk*upper;
        if (!std::isfinite(value)) return 1;
        *output=value;
        return 0;
    } catch (...) { return 2; }
}
