#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <chrono>
#include <execution>
#include <atomic>
#include <filesystem>
#include <cctype>
#include <utility>
#include <locale>
#include <array>
#include <iterator>
#include <format>
#include <print>

#include "laszip_api.h"

#include "culaz.cuh"
using namespace std;
using std::chrono::high_resolution_clock;

static long long start_time = high_resolution_clock::now().time_since_epoch().count();

inline double now() {
	auto now = std::chrono::high_resolution_clock::now();
	long long nanosSinceStart = now.time_since_epoch().count() - start_time;

	double secondsSinceStart = double(nanosSinceStart) / 1'000'000'000.0;

	return secondsSinceStart;
}

static uint64_t point_record_count(const laszip_header* header) {
    return header->number_of_point_records != 0
        ? header->number_of_point_records
        : header->extended_number_of_point_records;
}

// ---------------------------------------------------------------
// Format support check
// ---------------------------------------------------------------

static uint16_t read_le16(const uint8_t* p) {
    return uint16_t(p[0] | (p[1] << 8));
}

static uint32_t read_le32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
        (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

static string describe_point_format(int format) {
    switch (format) {
    case 0:  return "XYZ, intensity, classification";
    case 1:  return "format 0 + GPS time";
    case 2:  return "format 0 + RGB";
    case 3:  return "format 0 + GPS time + RGB";
    case 4:  return "format 1 + waveform";
    case 5:  return "format 3 + waveform";
    case 6:  return "LAS 1.4 base format with GPS time";
    case 7:  return "LAS 1.4 format 6 + RGB";
    case 8:  return "LAS 1.4 format 7 + NIR";
    case 9:  return "LAS 1.4 format 6 + waveform";
    case 10: return "LAS 1.4 format 8 + waveform";
    default: return "unknown";
    }
}

static string describe_item_type(uint16_t type) {
    static const char* names[] = {
        "BYTE", "SHORT", "INT", "LONG", "FLOAT", "DOUBLE",
        "POINT10", "GPSTIME11", "RGB12", "WAVEPACKET13",
        "POINT14", "RGB14", "RGBNIR14", "WAVEPACKET14", "BYTE14"
    };
    if (type < std::size(names)) {
        return names[type];
    }
    return "unknown item type " + to_string(type);
}

static string check_format_support(const string& filename, uint32_t& chunk_size) {
    constexpr uint16_t LASZIP_VLR_RECORD_ID = 22204;
    constexpr uint16_t COMPRESSOR_POINTWISE_CHUNKED = 2;
    constexpr uint16_t ITEM_BYTE = 0;
    constexpr uint16_t ITEM_POINT10 = 6;
    constexpr uint16_t ITEM_RGB12 = 8;
    constexpr uint32_t VARIABLE_CHUNK_SIZE = 0xFFFFFFFFu;

    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        return "the file could not be opened";
    }
    auto read_at = [&](uint64_t offset, void* destination, size_t size) {
        file.clear();
        file.seekg(static_cast<std::streamoff>(offset));
        return bool(file.read(
            static_cast<char*>(destination),
            static_cast<std::streamsize>(size)));
    };

    uint8_t header[107];
    if (!read_at(0, header, sizeof(header)) ||
        std::memcmp(header, "LASF", 4) != 0) {
        return "it is not a LAS/LAZ file (missing 'LASF' signature)";
    }
    const uint16_t header_size = read_le16(header + 94);
    const uint32_t vlr_count = read_le32(header + 100);
    const int point_format = header[104] & 0x3F;

    std::vector<uint8_t> laszip_vlr;
    uint64_t vlr_offset = header_size;
    for (uint32_t i = 0; i < vlr_count; ++i) {
        uint8_t vlr_header[54];
        if (!read_at(vlr_offset, vlr_header, sizeof(vlr_header))) {
            return "its variable length records are truncated";
        }
        const string user_id(
            reinterpret_cast<const char*>(vlr_header + 2),
            strnlen(reinterpret_cast<const char*>(vlr_header + 2), 16));
        const uint16_t record_id = read_le16(vlr_header + 18);
        const uint16_t record_length = read_le16(vlr_header + 20);
        if (user_id == "laszip encoded" && record_id == LASZIP_VLR_RECORD_ID) {
            laszip_vlr.resize(record_length);
            if (!read_at(vlr_offset + sizeof(vlr_header),
                    laszip_vlr.data(), laszip_vlr.size())) {
                return "its LASzip VLR is truncated";
            }
            break;
        }
        vlr_offset += sizeof(vlr_header) + record_length;
    }

    if (laszip_vlr.empty()) {
        return "it is not LASzip-compressed (no LASzip VLR found); "
            "CuLAZ only decodes .laz files";
    }
    if (point_format != 2) {
        return std::format(
            "point data format {} ({}) is not supported; CuLAZ only decodes "
            "point format 2 (XYZ, intensity, classification, RGB)",
            point_format, describe_point_format(point_format));
    }
    if (laszip_vlr.size() < 34) {
        return "its LASzip VLR is too short";
    }

    const uint16_t compressor = read_le16(laszip_vlr.data() + 0);
    const uint16_t coder = read_le16(laszip_vlr.data() + 2);
    const uint32_t vlr_chunk_size = read_le32(laszip_vlr.data() + 12);
    const uint16_t item_count = read_le16(laszip_vlr.data() + 32);

    if (compressor != COMPRESSOR_POINTWISE_CHUNKED) {
        static const char* compressor_names[] = {
            "no", "pointwise (unchunked)", "pointwise chunked",
            "layered chunked"
        };
        const string name = compressor < std::size(compressor_names)
            ? compressor_names[compressor]
            : "unknown (id " + to_string(compressor) + ")";
        return std::format(
            "it uses {} compression; CuLAZ only supports pointwise chunked "
            "compression (the LASzip default for point formats 0-5)", name);
    }
    if (coder != 0) {
        return std::format(
            "it uses an unknown entropy coder (id {})", coder);
    }
    if (laszip_vlr.size() < 34 + size_t(item_count) * 6) {
        return "its LASzip VLR item list is truncated";
    }

    bool has_point10 = false;
    bool has_rgb12 = false;
    for (uint16_t i = 0; i < item_count; ++i) {
        const uint8_t* item = laszip_vlr.data() + 34 + size_t(i) * 6;
        const uint16_t type = read_le16(item + 0);
        const uint16_t size = read_le16(item + 2);
        const uint16_t version = read_le16(item + 4);

        if (type == ITEM_BYTE) {
            return std::format(
                "it stores {} extra bytes per point, which CuLAZ does not "
                "support", size);
        }
        if (type != ITEM_POINT10 && type != ITEM_RGB12) {
            return std::format(
                "it contains the unsupported LASzip item {}",
                describe_item_type(type));
        }
        if (version != 2) {
            return std::format(
                "its {} item is compressed with LASzip item version {}; "
                "CuLAZ only supports version 2", describe_item_type(type),
                version);
        }
        has_point10 |= type == ITEM_POINT10;
        has_rgb12 |= type == ITEM_RGB12;
    }
    if (!has_point10 || !has_rgb12) {
        return "its LASzip item list does not contain both POINT10 and RGB12";
    }

    if (vlr_chunk_size == VARIABLE_CHUNK_SIZE) {
        return "it uses variable-sized chunks; CuLAZ only supports chunks "
            "with a fixed number of points";
    }
    if (vlr_chunk_size == 0) {
        return "its LASzip VLR specifies a chunk size of 0";
    }
    chunk_size = vlr_chunk_size;
    return {};
}

// ---------------------------------------------------------------
// LAZ metadata
// ---------------------------------------------------------------

int load_laz_metadata(const std::string& filename,
    uint32_t& num_chunks,
    std::vector<uint64_t>& chunk_offsets,
    uint64_t& actual_total_points,
    size_t& raw_file_size,
    std::array<double, 3>& scale,
    std::array<double, 3>& offset)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for binary reading.\n";
        return 1;
    }
    const std::streamsize size = file.tellg();
    if (size < 0) {
        std::cerr << "Failed to determine file size.\n";
        return 1;
    }
    raw_file_size = static_cast<size_t>(size);
    file.close();

    // Use LASzip only to parse the header and chunk table.
    laszip_POINTER laszip_reader = nullptr;
    laszip_create(&laszip_reader);

    laszip_BOOL is_compressed;
    laszip_request_compatibility_mode(laszip_reader, true);

    if (laszip_open_reader(laszip_reader, filename.c_str(), &is_compressed) != 0) {
        std::cerr << "Failed to open LASzip reader.\n";
        laszip_destroy(laszip_reader);
        return 1;
    }

    // Seeking to point 0 forces the reader to load the chunk table.
    laszip_seek_point(laszip_reader, 0);

    laszip_U32 internal_num_chunks = 0;
    laszip_I64* internal_starts = nullptr;

    if (laszip_get_chunk_starts(laszip_reader, &internal_num_chunks, &internal_starts) != 0) {
        laszip_CHAR* err = nullptr;
        laszip_get_error(laszip_reader, &err);
        std::cerr << "Error getting chunks: " << err << "\n";
        laszip_close_reader(laszip_reader);
        laszip_destroy(laszip_reader);
        return 1;
    }
    num_chunks = internal_num_chunks;
    chunk_offsets.clear();
    chunk_offsets.reserve(num_chunks);
    for (laszip_U32 i = 0; i < num_chunks; ++i) {
        chunk_offsets.push_back(static_cast<uint64_t>(internal_starts[i]));
    }
    free(internal_starts);
    laszip_header* lazHeader = nullptr;
    laszip_get_header_pointer(laszip_reader, &lazHeader);
    if (!lazHeader) {
        std::println(stderr, "Failed to get LAS header.");
        laszip_close_reader(laszip_reader);
        laszip_destroy(laszip_reader);
        return 1;
    }
    actual_total_points = point_record_count(lazHeader);
    scale = {lazHeader->x_scale_factor, lazHeader->y_scale_factor,
        lazHeader->z_scale_factor};
    offset = {lazHeader->x_offset, lazHeader->y_offset, lazHeader->z_offset};
    laszip_close_reader(laszip_reader);
    laszip_destroy(laszip_reader);

    return 0;
}

template <typename LazPoint>
bool comparePoints(
    const LazPoint* laz_point,
    const PointFormat2& p,
    uint64_t index) {
    bool is_match = true;

    // 1. Check Coordinates
    if (laz_point->X != p.X) is_match = false;
    if (laz_point->Y != p.Y) is_match = false;
    if (laz_point->Z != p.Z) is_match = false;

    // 2. Check Attributes
    if (laz_point->intensity != p.Intensity) is_match = false;
    if (laz_point->classification != p.Classification) is_match = false;

    // 3. Check RGB
    if (laz_point->rgb[0] != p.Red) is_match = false;
    if (laz_point->rgb[1] != p.Green) is_match = false;
    if (laz_point->rgb[2] != p.Blue) is_match = false;


    if (!is_match) {
        std::cerr << "\n========== MISMATCH AT POINT " << index << " ==========\n";

        std::cerr << "[LAZ Data] "
            << "X=" << laz_point->X << ", "
            << "Y=" << laz_point->Y << ", "
            << "Z=" << laz_point->Z << " | "
            << "Int=" << laz_point->intensity << " | "
            << "Class=" << (int)laz_point->classification << " | "
            << "RGB=(" << laz_point->rgb[0] << "," << laz_point->rgb[1] << "," << laz_point->rgb[2] << ")\n";

        std::cerr << "[GPU Data] "
            << "X=" << p.X << ", "
            << "Y=" << p.Y << ", "
            << "Z=" << p.Z << " | "
            << "Int=" << p.Intensity << " | "
            << "Class=" << (int)p.Classification << " | "
            << "RGB=(" << p.Red << "," << p.Green << "," << p.Blue << ")\n";

        std::cerr << "--- Specific Differences ---\n";
        if (laz_point->X != p.X) std::cerr << " -> X differs: LAZ=" << laz_point->X << " vs GPU=" << p.X << "\n";
        if (laz_point->Y != p.Y) std::cerr << " -> Y differs: LAZ=" << laz_point->Y << " vs GPU=" << p.Y << "\n";
        if (laz_point->Z != p.Z) std::cerr << " -> Z differs: LAZ=" << laz_point->Z << " vs GPU=" << p.Z << "\n";
        if (laz_point->intensity != p.Intensity) std::cerr << " -> Intensity differs: LAZ=" << laz_point->intensity << " vs GPU=" << p.Intensity << "\n";
        if (laz_point->classification != p.Classification) std::cerr << " -> Class differs: LAZ=" << (int)laz_point->classification << " vs GPU=" << (int)p.Classification << "\n";
        if (laz_point->rgb[0] != p.Red) std::cerr << " -> Red differs: LAZ=" << laz_point->rgb[0] << " vs GPU=" << p.Red << "\n";
        if (laz_point->rgb[1] != p.Green) std::cerr << " -> Green differs: LAZ=" << laz_point->rgb[1] << " vs GPU=" << p.Green << "\n";
        if (laz_point->rgb[2] != p.Blue) std::cerr << " -> Blue differs: LAZ=" << laz_point->rgb[2] << " vs GPU=" << p.Blue << "\n";

        std::cerr << "============================================\n";
    }

    return is_match;
}

class punct_facet : public std::numpunct<char> {
protected:
	char do_decimal_point() const { return '.'; };
	char do_thousands_sep() const { return '\''; };
	string do_grouping() const { return "\3"; }
};

inline std::locale getSaneLocale(){
	return std::locale(std::cout.getloc(), new punct_facet);
}

// ---------------------------------------------------------------
// CPU baseline
// ---------------------------------------------------------------

struct CpuBenchmarkResult {
    uint64_t points = 0;
    std::vector<double> iteration_ms;
};

int benchmark_laszip_cpu_multithreaded(
    const string& file,
    int benchmark_iterations,
    CpuBenchmarkResult* benchmark_result = nullptr) {

	uint64_t total_points = 0;

	{ // First, read number of points from laz file
		laszip_POINTER laszip_reader = nullptr;
		laszip_header* lazHeader = nullptr;

		laszip_create(&laszip_reader);
		if (!laszip_reader) {
			std::println(stderr, "Failed to create laszip reader.");
			return 1;
		}

		laszip_BOOL is_compressed;
		if (laszip_open_reader(laszip_reader, file.c_str(), &is_compressed) != 0) {
			std::println(stderr, "Failed to open {}", file);
			laszip_destroy(laszip_reader);
			return 1;
		}
		laszip_get_header_pointer(laszip_reader, &lazHeader);
		if (!lazHeader) {
			std::println(stderr, "Failed to get LAS header.");
			laszip_close_reader(laszip_reader);
			laszip_destroy(laszip_reader);
			return 1;
		}

		total_points = point_record_count(lazHeader);

		laszip_close_reader(laszip_reader);
		laszip_destroy(laszip_reader);
	}

	// Create workload (batches)
	constexpr uint64_t BATCH_SIZE = 1'000'000;
	uint64_t numBatches = (total_points + BATCH_SIZE - 1) / BATCH_SIZE;
	vector<int> batches(numBatches);
	for(int i = 0; i < numBatches; i++) {
		batches[i] = i;
	}

    struct CpuRunResult {
        double seconds;
        uint64_t points;
        bool succeeded;
    };

    auto run_once = [&]() -> CpuRunResult {
        atomic_uint64_t totalPointsProcessed = 0;
        atomic_bool failed = false;
        const double t_start = now();

        for_each(std::execution::par, batches.begin(), batches.end(), [&](int batchIndex){
            laszip_POINTER laszip_reader = nullptr;
            laszip_point* laz_point = nullptr;
            bool reader_open = false;

            auto cleanup = [&]() {
                if (reader_open) {
                    laszip_close_reader(laszip_reader);
                }
                if (laszip_reader) {
                    laszip_destroy(laszip_reader);
                }
            };

            laszip_create(&laszip_reader);
            if (!laszip_reader) {
                failed = true;
                return;
            }

            laszip_BOOL is_compressed;
            if (laszip_open_reader(laszip_reader, file.c_str(), &is_compressed) != 0) {
                failed = true;
                cleanup();
                return;
            }
            reader_open = true;

            if (laszip_get_point_pointer(laszip_reader, &laz_point) != 0 || !laz_point) {
                failed = true;
                cleanup();
                return;
            }

            const uint64_t firstPoint = uint64_t(batchIndex) * BATCH_SIZE;
            const uint64_t pointsInBatch =
                std::min(BATCH_SIZE, total_points - firstPoint);
            if (laszip_seek_point(laszip_reader, firstPoint) != 0) {
                failed = true;
                cleanup();
                return;
            }

            uint64_t pointsProcessed = 0;
            for (; pointsProcessed < pointsInBatch; ++pointsProcessed) {
                if (laszip_read_point(laszip_reader) != 0) {
                    failed = true;
                    break;
                }
            }

            totalPointsProcessed += pointsProcessed;
            cleanup();
        });

        const double duration = now() - t_start;
        const uint64_t pointsProcessed = totalPointsProcessed.load();
        return {
            duration,
            pointsProcessed,
            !failed.load() && pointsProcessed == total_points
        };
    };

    println("Warming up multithreaded CPU benchmark...");
    if (!run_once().succeeded) {
        std::println(stderr, "CPU benchmark warm-up failed.");
        return 1;
    }

    const int N = std::max(1, benchmark_iterations);
    double sum_seconds = 0.0;
    double sum_seconds2 = 0.0;
    std::vector<double> iteration_ms;
    iteration_ms.reserve(N);
    for (int iteration = 0; iteration < N; ++iteration) {
        const CpuRunResult result = run_once();
        if (!result.succeeded) {
            std::println(stderr, "CPU benchmark iteration {} failed.", iteration + 1);
            return 1;
        }

        sum_seconds += result.seconds;
        sum_seconds2 += result.seconds * result.seconds;
        iteration_ms.push_back(result.seconds * 1'000.0);
    }

    const double avg_seconds = sum_seconds / double(N);
    const double variance =
        (sum_seconds2 / double(N)) - (avg_seconds * avg_seconds);
    const double std_seconds = variance > 0.0 ? std::sqrt(variance) : 0.0;
    const double avg_ms = avg_seconds * 1'000.0;
    const double std_ms = std_seconds * 1'000.0;
    const double points_per_ms =
        avg_ms > 0.0 ? double(total_points) / avg_ms : 0.0;

    if (benchmark_result) {
        benchmark_result->points = total_points;
        benchmark_result->iteration_ms = std::move(iteration_ms);
    }

    println("== CPU BENCHMARK (Multithreaded) ==");
	println("points         {:L}", total_points);
	println("CPU time       {:.3f} ms (avg over {} runs)", avg_ms, N);
	println("stddev         {:.3f} ms", std_ms);
    println("throughput     {:.1f} points/ms", points_per_ms);
    return 0;
}

struct AggregateBenchmarkResult {
    uint64_t files = 0;
    uint64_t points = 0;
    std::vector<double> iteration_ms;
};

template <typename BenchmarkResult>
void add_benchmark_result(
    AggregateBenchmarkResult& aggregate,
    const BenchmarkResult& result) {
    if (result.iteration_ms.empty()) {
        return;
    }
    if (aggregate.iteration_ms.empty()) {
        aggregate.iteration_ms.assign(result.iteration_ms.size(), 0.0);
    }
    if (aggregate.iteration_ms.size() != result.iteration_ms.size()) {
        throw std::runtime_error(
            "Benchmark iteration count changed between files.");
    }

    ++aggregate.files;
    aggregate.points += result.points;
    for (size_t i = 0; i < result.iteration_ms.size(); ++i) {
        aggregate.iteration_ms[i] += result.iteration_ms[i];
    }
}

void print_benchmark_summary(
    const char* label,
    const AggregateBenchmarkResult& result) {
    if (result.iteration_ms.empty()) {
        return;
    }

    const double sum_ms =
        std::accumulate(result.iteration_ms.begin(),
            result.iteration_ms.end(), 0.0);
    const double avg_ms = sum_ms / double(result.iteration_ms.size());
    double variance_ms = 0.0;
    for (const double ms : result.iteration_ms) {
        const double difference = ms - avg_ms;
        variance_ms += difference * difference;
    }
    variance_ms /= double(result.iteration_ms.size());
    const double std_ms = std::sqrt(variance_ms);
    const double points_per_ms =
        avg_ms > 0.0 ? double(result.points) / avg_ms : 0.0;

    std::cout << "\n== " << label << " SUMMARY (ALL FILES) ==\n";
    println("files          {:L}", result.files);
    println("points         {:L}", result.points);
    println("decode time    {:.3f} ms (avg over {} complete runs)",
        avg_ms, result.iteration_ms.size());
    println("stddev         {:.3f} ms", std_ms);
    println("throughput     {:.1f} points/ms", points_per_ms);
}

// ---------------------------------------------------------------
// Point cloud statistics
// ---------------------------------------------------------------

static string classification_name(int classification) {
    static const char* names[] = {
        "never classified", "unclassified", "ground", "low vegetation",
        "medium vegetation", "high vegetation", "building",
        "low point (noise)", "model key-point", "water",
        nullptr, nullptr, "overlap"
    };
    if (classification < int(std::size(names)) && names[classification]) {
        return names[classification];
    }
    return "reserved";
}

static void print_point_cloud_stats(
    const string& filename,
    const PointCloudStats& stats,
    const std::array<double, 3>& scale,
    const std::array<double, 3>& offset) {
    if (stats.point_count == 0) {
        return;
    }
    const double n = double(stats.point_count);
    auto world = [&](int axis, int32_t value) {
        return double(value) * scale[axis] + offset[axis];
    };
    const double min_x = world(0, stats.min_X);
    const double min_y = world(1, stats.min_Y);
    const double min_z = world(2, stats.min_Z);
    const double max_x = world(0, stats.max_X);
    const double max_y = world(1, stats.max_Y);
    const double max_z = world(2, stats.max_Z);

    println("\n== STATISTICS: {} ==",
        std::filesystem::path(filename).filename().string());
    println("points         {:L}", stats.point_count);
    println("min            ({:.3f}, {:.3f}, {:.3f})", min_x, min_y, min_z);
    println("max            ({:.3f}, {:.3f}, {:.3f})", max_x, max_y, max_z);
    println("extent         {:.3f} x {:.3f} x {:.3f}",
        max_x - min_x, max_y - min_y, max_z - min_z);
    println("mean intensity {:.1f}", double(stats.intensity_sum) / n);
    println("mean RGB       ({:.1f}, {:.1f}, {:.1f})",
        double(stats.red_sum) / n, double(stats.green_sum) / n,
        double(stats.blue_sum) / n);
    println("classes");
    for (int c = 0; c < 32; ++c) {
        const uint64_t count = stats.classification_counts[c];
        if (count == 0) {
            continue;
        }
        println("  {:>2} {:<18} {:>15L} ({:5.1f}%)",
            c, classification_name(c), count, 100.0 * double(count) / n);
    }
}

// ---------------------------------------------------------------
// Command line
// ---------------------------------------------------------------

static void print_usage(const string& program) {
    println(
        "Usage: {} <file.laz | directory> [options]\n"
        "\n"
        "Decodes LAZ point clouds on the GPU and reports decode throughput.\n"
        "As an example of using the decoded points without copying them to the\n"
        "host, it also computes per-file statistics (bounds, classes, means)\n"
        "on the GPU. If a directory is given, all .laz files in it are decoded\n"
        "together.\n"
        "\n"
        "Supported input: LAS point format 2 (XYZ, intensity, classification, RGB)\n"
        "compressed with LASzip's default pointwise chunked scheme\n"
        "(item version 2, fixed chunk size, no extra bytes).\n"
        "\n"
        "Options:\n"
        "  --iterations N     timed runs per benchmark after one warm-up (default 1)\n"
        "  --gpu-only         skip the multithreaded LASzip CPU baseline\n"
        "  --validate         copy decoded points to the host and compare the first\n"
        "                     10,000,000 points of each file against LASzip\n"
        "  --no-stats         skip the GPU point cloud statistics\n"
        "  --batch-chunks N   maximum chunks per GPU batch\n"
        "                     (default: one chunk per resident warp)\n"
        "  -h, --help         show this help",
        program);
}

int main(int argc, char** argv){

	std::locale::global(getSaneLocale());

    const string program = argc > 0
        ? std::filesystem::path(argv[0]).filename().string()
        : string("CuLAZ");
    string input;
    bool run_cpu_benchmark = true;
    bool run_validation = false;
    bool compute_stats = true;
    int benchmark_iterations = 1;
    uint32_t max_batch_chunks = 0;

    auto parse_count = [&](int& i, const string& option) -> int {
        if (i + 1 >= argc) {
            throw std::invalid_argument(option + " requires a value");
        }
        const string value = argv[++i];
        size_t parsed = 0;
        int result = 0;
        try {
            result = std::stoi(value, &parsed);
        }
        catch (const std::exception&) {
            parsed = 0;
        }
        if (parsed != value.size() || result < 1) {
            throw std::invalid_argument(
                option + " expects a positive integer, got '" + value + "'");
        }
        return result;
    };

    try {
        for (int i = 1; i < argc; ++i) {
            const string arg = argv[i];
            if (arg == "-h" || arg == "--help") {
                print_usage(program);
                return 0;
            }
            else if (arg == "--gpu-only") {
                run_cpu_benchmark = false;
            }
            else if (arg == "--validate") {
                run_validation = true;
            }
            else if (arg == "--no-stats") {
                compute_stats = false;
            }
            else if (arg == "--iterations") {
                benchmark_iterations = parse_count(i, arg);
            }
            else if (arg == "--batch-chunks") {
                max_batch_chunks =
                    static_cast<uint32_t>(parse_count(i, arg));
            }
            else if (arg.starts_with("-")) {
                throw std::invalid_argument("unknown option '" + arg + "'");
            }
            else if (!input.empty()) {
                throw std::invalid_argument(
                    "only one input path may be given");
            }
            else {
                input = arg;
            }
        }
    }
    catch (const std::invalid_argument& error) {
        std::println(stderr, "Error: {}. Run '{} --help' for usage.",
            error.what(), program);
        return 1;
    }

    if (input.empty()) {
        print_usage(program);
        return 1;
    }

    namespace fs = std::filesystem;
    const fs::path input_path(input);
    std::error_code filesystem_error;
    const bool input_is_directory =
        fs::is_directory(input_path, filesystem_error);
    if (filesystem_error) {
        std::cerr << "Failed to inspect input path '" << input << "': "
            << filesystem_error.message() << "\n";
        return 1;
    }

    std::vector<fs::path> files;
    if (input_is_directory) {
        fs::directory_iterator iterator(input_path, filesystem_error);
        const fs::directory_iterator end;
        while (!filesystem_error && iterator != end) {
            const fs::directory_entry& entry = *iterator;
            std::error_code entry_error;
            if (entry.is_regular_file(entry_error)) {
                string extension = entry.path().extension().string();
                std::transform(extension.begin(), extension.end(), extension.begin(),
                    [](unsigned char c) {
                        return static_cast<char>(std::tolower(c));
                    });
                if (extension == ".laz") {
                    files.push_back(entry.path());
                }
            }
            iterator.increment(filesystem_error);
        }
        if (filesystem_error) {
            std::cerr << "Failed while reading directory '" << input << "': "
                << filesystem_error.message() << "\n";
            return 1;
        }
        std::sort(files.begin(), files.end());
        if (files.empty()) {
            std::cerr << "No .laz files found in directory '" << input << "'.\n";
            return 1;
        }
        std::cout << "Found " << files.size() << " LAZ files in '"
            << input << "'.\n";
    }
    else {
        if (!fs::is_regular_file(input_path, filesystem_error)) {
            std::cerr << "Input path is not a regular file or directory: '"
                << input << "'.\n";
            return 1;
        }
        files.push_back(input_path);
    }

    AggregateBenchmarkResult gpu_summary;
    AggregateBenchmarkResult cpu_summary;

    struct PreparedFile {
        size_t source_index;
        string filename;
        size_t raw_file_size = 0;
        uint32_t num_chunks = 0;
        uint32_t chunk_size = 0;
        std::vector<uint64_t> chunk_offsets;
        uint64_t total_points = 0;
        std::array<double, 3> scale = {};
        std::array<double, 3> offset = {};
        std::vector<PointFormat2> decoded_points;
        PointCloudStats stats;
    };

    std::vector<PreparedFile> prepared_files;
    prepared_files.reserve(files.size());
    std::vector<bool> file_failed(files.size(), false);
    size_t skipped_files = 0;

    for (size_t i = 0; i < files.size(); ++i) {
        const string filename = files[i].string();
        if (files.size() > 1) {
            std::cout << "\n=== Loading file " << (i + 1) << " / "
                << files.size() << ": " << filename << " ===\n";
        }

        uint32_t chunk_size = 0;
        const string unsupported = check_format_support(filename, chunk_size);
        if (!unsupported.empty()) {
            std::println(stderr, "Skipping '{}': {}.", filename, unsupported);
            ++skipped_files;
            continue;
        }

        if (run_cpu_benchmark) {
            CpuBenchmarkResult cpu_result;
            if (benchmark_laszip_cpu_multithreaded(
                    filename, benchmark_iterations, &cpu_result) != 0) {
                file_failed[i] = true;
                std::cerr << "CPU benchmark failed for '" << filename
                    << "'.\n";
            }
            else {
                add_benchmark_result(cpu_summary, cpu_result);
            }
        }

        PreparedFile prepared;
        prepared.source_index = i;
        prepared.filename = filename;
        prepared.chunk_size = chunk_size;
        if (load_laz_metadata(
                filename,
                prepared.num_chunks,
                prepared.chunk_offsets,
                prepared.total_points,
                prepared.raw_file_size,
                prepared.scale,
                prepared.offset) != 0) {
            file_failed[i] = true;
            std::cerr << "Initialization failed for '" << filename << "'.\n";
            continue;
        }

        std::cout << "Successfully indexed " << prepared.raw_file_size
            << " bytes.\n";
        std::cout << "Found "
            << (prepared.num_chunks > 0 ? prepared.num_chunks - 1 : 0)
            << " chunks of " << prepared.chunk_size << " points.\n";
        prepared_files.push_back(std::move(prepared));
    }

    if (!prepared_files.empty()) {
        std::vector<CulazFileInput> gpu_files;
        gpu_files.reserve(prepared_files.size());
        for (PreparedFile& file : prepared_files) {
            gpu_files.push_back({
                file.filename,
                nullptr,
                file.raw_file_size,
                &file.chunk_offsets,
                file.num_chunks,
                file.chunk_size,
                file.total_points,
                &file.decoded_points,
                compute_stats ? &file.stats : nullptr
            });
        }

        GpuBenchmarkResult gpu_result;
        if (decompress_files(
                gpu_files,
                run_validation,
                benchmark_iterations,
                max_batch_chunks,
                &gpu_result) != 0) {
            std::cerr << "Combined GPU decompression failed.\n";
            for (const PreparedFile& file : prepared_files) {
                file_failed[file.source_index] = true;
            }
        }
        else {
            gpu_summary.files = prepared_files.size();
            gpu_summary.points = gpu_result.points;
            gpu_summary.iteration_ms = std::move(gpu_result.iteration_ms);
            if (compute_stats) {
                for (const PreparedFile& file : prepared_files) {
                    print_point_cloud_stats(
                        file.filename, file.stats, file.scale, file.offset);
                }
            }
        }
    }

    auto validate_file = [&](const PreparedFile& file) -> bool {
        laszip_POINTER laszip_reader = nullptr;
        laszip_header* lazHeader = nullptr;
        laszip_point* laz_point = nullptr;

        laszip_create(&laszip_reader);
        if (!laszip_reader) {
            std::println(stderr, "Failed to create laszip reader.");
            return false;
        }

        laszip_BOOL is_compressed;
        laszip_request_compatibility_mode(laszip_reader, true);

        if (laszip_open_reader(
                laszip_reader, file.filename.c_str(), &is_compressed) != 0) {
            std::println(stderr, "Failed to open {}", file.filename);
            laszip_destroy(laszip_reader);
            return false;
        }

        laszip_get_header_pointer(laszip_reader, &lazHeader);
        if (!lazHeader) {
            std::println(stderr, "Failed to get LAS header.");
            laszip_close_reader(laszip_reader);
            laszip_destroy(laszip_reader);
            return false;
        }
        laszip_get_point_pointer(laszip_reader, &laz_point);
        if (!laz_point) {
            std::println(stderr, "Failed to get LAS point pointer.");
            laszip_close_reader(laszip_reader);
            laszip_destroy(laszip_reader);
            return false;
        }

        const uint64_t total_points = point_record_count(lazHeader);
        const uint64_t pointLimit =
            std::min<uint64_t>(total_points, 10'000'000);

        std::println(
            "Reading {} points from '{}'...", pointLimit, file.filename);

        bool allMatch = true;
        for (uint64_t i = 0; i < pointLimit; i++) {
            if (laszip_read_point(laszip_reader) != 0) {
                allMatch = false;
                std::println(stderr,
                    "Warning: Error reading point {}. Stopping.", i);
                break;
            }

            const auto& p = file.decoded_points[size_t(i)];
            if (!comparePoints(laz_point, p, i)) {
                allMatch = false;
                std::cout << "Found mismatch at " << i << ".\n";
                break;
            }
        }

        if (allMatch) {
            println("{:L} tested points match!", pointLimit);
        }

        laszip_close_reader(laszip_reader);
        laszip_destroy(laszip_reader);
        return allMatch;
    };

    if (run_validation && !gpu_summary.iteration_ms.empty()) {
        for (size_t i = 0; i < prepared_files.size(); ++i) {
            const PreparedFile& file = prepared_files[i];
            if (prepared_files.size() > 1) {
                std::cout << "\n=== Validating file " << (i + 1) << " / "
                    << prepared_files.size() << ": " << file.filename
                    << " ===\n";
            }
            if (!validate_file(file)) {
                file_failed[file.source_index] = true;
                std::cerr << "Validation failed for '" << file.filename
                    << "'.\n";
            }
        }
    }

    print_benchmark_summary("GPU", gpu_summary);
    if (run_cpu_benchmark) {
        print_benchmark_summary("CPU", cpu_summary);
    }

    if (skipped_files != 0 && files.size() > 1) {
        std::println(stderr, "\n{} of {} files skipped (unsupported format).",
            skipped_files, files.size());
    }
    const int failed_files = static_cast<int>(
        std::count(file_failed.begin(), file_failed.end(), true));
    if (failed_files != 0) {
        std::cerr << failed_files << " of " << files.size()
            << " files failed.\n";
        return 1;
    }
    return skipped_files == files.size() ? 1 : 0;
}
