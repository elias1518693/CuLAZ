#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct alignas(4) PointFormat2 {
    int32_t  X;
    int32_t  Y;
    int32_t  Z;
    uint16_t Intensity;
    uint8_t  ReturnMask;
    uint8_t  Classification;
    int8_t   ScanAngleRank;
    uint8_t  UserData;
    uint16_t PointSourceID;
    uint16_t Red;
    uint16_t Green;
    uint16_t Blue;
};

struct GpuBenchmarkResult {
    uint64_t points = 0;
    std::vector<double> iteration_ms;
    double stats_ms = 0.0;
};

struct PointCloudStats {
    uint64_t point_count = 0;
    int32_t min_X = INT32_MAX;
    int32_t min_Y = INT32_MAX;
    int32_t min_Z = INT32_MAX;
    int32_t max_X = INT32_MIN;
    int32_t max_Y = INT32_MIN;
    int32_t max_Z = INT32_MIN;
    uint64_t intensity_sum = 0;
    uint64_t red_sum = 0;
    uint64_t green_sum = 0;
    uint64_t blue_sum = 0;
    uint64_t classification_counts[32] = {};
};


struct CulazFileInput {
    std::string filename;
    const std::vector<uint8_t>* raw_file_data = nullptr;
    size_t raw_byte_size = 0;
    const std::vector<uint64_t>* chunk_offsets = nullptr;
    uint32_t num_chunks = 0;
    uint32_t chunk_size = 0;
    uint64_t total_points = 0;
    std::vector<PointFormat2>* points = nullptr;
    PointCloudStats* stats = nullptr;
};

int decompress_files(
    const std::vector<CulazFileInput>& files,
    bool copy_results = true,
    int benchmark_iterations = 20,
    uint32_t max_batch_chunks = 0,
    GpuBenchmarkResult* benchmark_result = nullptr);

int decompress(const std::vector<uint8_t>& raw_file_data,
    uint32_t num_chunks,
    uint32_t chunk_size,
    const std::vector<uint64_t>& chunk_offsets,
    uint64_t actual_total_points,
    std::vector<PointFormat2>& points,
    bool copy_results = true,
    int benchmark_iterations = 20,
    uint32_t max_batch_chunks = 0,
    GpuBenchmarkResult* benchmark_result = nullptr);
