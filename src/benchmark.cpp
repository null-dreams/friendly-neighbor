#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <string>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <sqlite3.h>
#include "distance.hpp"

// Linux performance counter headers
#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <unistd.h>

static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                            int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

int get_perf_event_paranoid() {
    std::ifstream file("/proc/sys/kernel/perf_event_paranoid");
    int val = -1;
    if (file >> val) {
        return val;
    }
    return -1;
}
#endif

// Platform-independent TSC reader
#if defined(__x86_64__) || defined(_M_X64)
    #ifdef _MSC_VER
        #include <intrin.h>
    #else
        #include <x86intrin.h>
    #endif
    // Reads TSC and acts as a serialization barrier to prevent instruction reordering
    inline uint64_t read_tsc() {
        unsigned int aux;
        return __rdtscp(&aux);
    }
    const std::string ARCH_LABEL = "x86_64 (RDTSCP)";
#elif defined(__aarch64__)
    inline uint64_t read_tsc() {
        // Reads the physical counter register on ARM64 (ticks at a constant system frequency)
        uint64_t val;
        asm volatile("mrs %0, cntvct_el0" : "=r" (val));
        return val;
    }
    const std::string ARCH_LABEL = "ARM64 (CNTVCT_EL0)";
#else
    #include <chrono>
    inline uint64_t read_tsc() {
        return std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }
    const std::string ARCH_LABEL = "Generic Fallback (System Clock)";
#endif

// Helper class for cache miss measurements
class CacheMissMeasurer {
private:
    int fd_misses = -1;
    int fd_refs = -1;
    bool supported = false;
public:
    CacheMissMeasurer() {
#ifdef __linux__
        struct perf_event_attr pe;
        std::memset(&pe, 0, sizeof(struct perf_event_attr));
        pe.type = PERF_TYPE_HARDWARE;
        pe.size = sizeof(struct perf_event_attr);
        pe.disabled = 1;
        pe.exclude_kernel = 1;
        pe.exclude_hv = 1;

        pe.config = PERF_COUNT_HW_CACHE_MISSES;
        fd_misses = perf_event_open(&pe, 0, -1, -1, 0);

        pe.config = PERF_COUNT_HW_CACHE_REFERENCES;
        fd_refs = perf_event_open(&pe, 0, -1, -1, 0);

        if (fd_misses != -1 && fd_refs != -1) {
            supported = true;
        } else {
            if (fd_misses != -1) { close(fd_misses); fd_misses = -1; }
            if (fd_refs != -1) { close(fd_refs); fd_refs = -1; }
        }
#endif
    }

    ~CacheMissMeasurer() {
#ifdef __linux__
        if (fd_misses != -1) close(fd_misses);
        if (fd_refs != -1) close(fd_refs);
#endif
    }

    void start() {
#ifdef __linux__
        if (supported) {
            ioctl(fd_misses, PERF_EVENT_IOC_RESET, 0);
            ioctl(fd_refs, PERF_EVENT_IOC_RESET, 0);
            ioctl(fd_misses, PERF_EVENT_IOC_ENABLE, 0);
            ioctl(fd_refs, PERF_EVENT_IOC_ENABLE, 0);
        }
#endif
    }

    std::pair<int64_t, int64_t> stop() {
#ifdef __linux__
        if (!supported) return {-1, -1};
        ioctl(fd_misses, PERF_EVENT_IOC_DISABLE, 0);
        ioctl(fd_refs, PERF_EVENT_IOC_DISABLE, 0);
        int64_t misses = 0;
        int64_t refs = 0;
        if (read(fd_misses, &misses, sizeof(misses)) != sizeof(misses)) misses = -1;
        if (read(fd_refs, &refs, sizeof(refs)) != sizeof(refs)) refs = -1;
        return {misses, refs};
#else
        return {-1, -1};
#endif
    }

    bool is_supported() const { return supported; }
};

// Helper to get RSS memory footprint from /proc/self/status on Linux (returns in KB)
void get_memory_footprint(size_t& rss_kb, size_t& peak_rss_kb) {
    rss_kb = 0;
    peak_rss_kb = 0;
#ifdef __linux__
    std::ifstream file("/proc/self/status");
    std::string line;
    while (std::getline(file, line)) {
        if (line.compare(0, 6, "VmRSS:") == 0) {
            std::stringstream ss(line.substr(6));
            ss >> rss_kb;
        } else if (line.compare(0, 6, "VmHWM:") == 0) {
            std::stringstream ss(line.substr(6));
            ss >> peak_rss_kb;
        }
    }
#endif
}

// SQLite C-API Custom Function registration
static void l2_distance_sqlite(sqlite3_context* context, int argc, sqlite3_value** argv) {
    if (argc != 2) {
        std::string err_msg = "l2_distance requires exactly 2 arguments, but instead received " + std::to_string(argc);
        sqlite3_result_error(context, err_msg.c_str(), -1);
        return;
    }

    if (sqlite3_value_type(argv[0]) != SQLITE_BLOB || sqlite3_value_type(argv[1]) != SQLITE_BLOB) {
        sqlite3_result_error(context, "Both arguments to l2_distance must be BLOBs", -1);
        return;
    }

    const void* blob0 = sqlite3_value_blob(argv[0]);
    const void* blob1 = sqlite3_value_blob(argv[1]);
    int bytes0 = sqlite3_value_bytes(argv[0]);
    int bytes1 = sqlite3_value_bytes(argv[1]);

    if (bytes0 != bytes1) {
        sqlite3_result_error(context, "BLOBs must be of equal length", -1);
        return;
    }

    if (bytes0 % sizeof(float) != 0) {
        sqlite3_result_error(context, "BLOB size must be a multiple of sizeof(float)", -1);
        return;
    }

    size_t dimensions = bytes0 / sizeof(float);
    if (dimensions == 0) {
        sqlite3_result_error(context, "Vector cannot be empty", -1);
        return;
    }

    const float* ptr0 = nullptr;
    const float* ptr1 = nullptr;
    std::vector<float> vec1;
    std::vector<float> vec2;

    if (reinterpret_cast<std::uintptr_t>(blob0) % alignof(float) == 0) {
        ptr0 = reinterpret_cast<const float*>(blob0);
    } else {
        vec1.resize(dimensions);
        std::memcpy(vec1.data(), blob0, bytes0);
        ptr0 = vec1.data();
    }

    if (reinterpret_cast<std::uintptr_t>(blob1) % alignof(float) == 0) {
        ptr1 = reinterpret_cast<const float*>(blob1);
    } else {
        vec2.resize(dimensions);
        std::memcpy(vec2.data(), blob1, bytes1);
        ptr1 = vec2.data();
    }

    float distance = vector_engine::l2_distance_squared(ptr0, ptr1, dimensions);
    sqlite3_result_double(context, static_cast<double>(distance));
}

int main() {
    const int NUM_VECTORS = 10000;
    const int DIMENSIONS = 128;
    const int NUM_QUERIES = 200;

    std::cout << "--- C++ Engine Benchmark (Extended Details) ---" << std::endl;
    std::cout << "Target Arch : " << ARCH_LABEL << std::endl;
    std::cout << "Vectors     : " << NUM_VECTORS << std::endl;
    std::cout << "Dimensions  : " << DIMENSIONS << std::endl;
    std::cout << "Queries     : " << NUM_QUERIES << std::endl;

    // Report initial process footprint
    size_t init_rss = 0, init_peak = 0;
    get_memory_footprint(init_rss, init_peak);

    // 1. Database Setup
    sqlite3* db;
    if (sqlite3_open(":memory:", &db) != SQLITE_OK) {
        std::cerr << "Failed to open in-memory database: " << sqlite3_errmsg(db) << std::endl;
        return 1;
    }

    if (sqlite3_create_function_v2(db, "l2_distance", 2, SQLITE_UTF8 | SQLITE_DETERMINISTIC, 
                                   nullptr, l2_distance_sqlite, nullptr, nullptr, nullptr) != SQLITE_OK) {
        std::cerr << "Failed to register custom SQLite function: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_close(db);
        return 1;
    }

    char* errMsg = nullptr;
    if (sqlite3_exec(db, "CREATE TABLE vec_table (id TEXT PRIMARY KEY, embedding BLOB NOT NULL);", nullptr, nullptr, &errMsg) != SQLITE_OK) {
        std::cerr << "Failed to create table: " << (errMsg ? errMsg : "unknown error") << std::endl;
        if (errMsg) sqlite3_free(errMsg);
        sqlite3_close(db);
        return 1;
    }

    // 2. Pre-generate Dataset
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    std::vector<std::vector<float>> dataset(NUM_VECTORS, std::vector<float>(DIMENSIONS));
    for (int i = 0; i < NUM_VECTORS; ++i) {
        for (int d = 0; d < DIMENSIONS; ++d) {
            dataset[i][d] = distribution(generator);
        }
    }

    std::vector<std::vector<float>> queries(NUM_QUERIES, std::vector<float>(DIMENSIONS));
    for (int i = 0; i < NUM_QUERIES; ++i) {
        for (int d = 0; d < DIMENSIONS; ++d) {
            queries[i][d] = distribution(generator);
        }
    }

    // 3. Ingestion (Warmup Database)
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, "INSERT INTO vec_table (id, embedding) VALUES (?, ?);", -1, &stmt, nullptr) != SQLITE_OK) {
        std::cerr << "Failed to prepare insert statement: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_close(db);
        return 1;
    }

    if (sqlite3_exec(db, "BEGIN TRANSACTION;", nullptr, nullptr, &errMsg) != SQLITE_OK) {
        std::cerr << "Failed to begin transaction: " << (errMsg ? errMsg : "unknown error") << std::endl;
        if (errMsg) sqlite3_free(errMsg);
        sqlite3_finalize(stmt);
        sqlite3_close(db);
        return 1;
    }

    for (int i = 0; i < NUM_VECTORS; ++i) {
        std::string id = "id_" + std::to_string(i);
        sqlite3_bind_text(stmt, 1, id.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_blob(stmt, 2, dataset[i].data(), DIMENSIONS * sizeof(float), SQLITE_STATIC);
        
        if (sqlite3_step(stmt) != SQLITE_DONE) {
            std::cerr << "Failed to execute insert step: " << sqlite3_errmsg(db) << std::endl;
            sqlite3_exec(db, "ROLLBACK TRANSACTION;", nullptr, nullptr, nullptr);
            sqlite3_finalize(stmt);
            sqlite3_close(db);
            return 1;
        }
        sqlite3_reset(stmt);
    }

    if (sqlite3_exec(db, "COMMIT TRANSACTION;", nullptr, nullptr, &errMsg) != SQLITE_OK) {
        std::cerr << "Failed to commit transaction: " << (errMsg ? errMsg : "unknown error") << std::endl;
        if (errMsg) sqlite3_free(errMsg);
        sqlite3_finalize(stmt);
        sqlite3_close(db);
        return 1;
    }
    sqlite3_finalize(stmt);

    // 4. Prepare Query Statement
    sqlite3_stmt* query_stmt;
    if (sqlite3_prepare_v2(db, "SELECT id, l2_distance(embedding, ?) AS dist FROM vec_table ORDER BY dist ASC LIMIT 10;", -1, &query_stmt, nullptr) != SQLITE_OK) {
        std::cerr << "Failed to prepare query statement: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_close(db);
        return 1;
    }

    std::vector<uint64_t> query_cycles;
    query_cycles.reserve(NUM_QUERIES);

    // Warm-up cache run (unmeasured)
    for (int i = 0; i < 5; ++i) {
        sqlite3_bind_blob(query_stmt, 1, queries[i].data(), DIMENSIONS * sizeof(float), SQLITE_STATIC);
        while (sqlite3_step(query_stmt) == SQLITE_ROW) {}
        sqlite3_reset(query_stmt);
    }

    // 5. Timed Queries via TSC & Cache Counter Setup
    CacheMissMeasurer cache_measurer;
    cache_measurer.start();

    for (int i = 0; i < NUM_QUERIES; ++i) {
        sqlite3_bind_blob(query_stmt, 1, queries[i].data(), DIMENSIONS * sizeof(float), SQLITE_STATIC);

        // --- Critical Timing Section ---
        uint64_t start_cycles = read_tsc();

        while (sqlite3_step(query_stmt) == SQLITE_ROW) {
            // Force compiler to keep these variables to prevent optimization-out of loops
            volatile const unsigned char* res_id = sqlite3_column_text(query_stmt, 0);
            volatile double dist = sqlite3_column_double(query_stmt, 1);
            (void)res_id; // Suppress unused var warnings
            (void)dist;
        }

        uint64_t end_cycles = read_tsc();
        // ------------------------------

        query_cycles.push_back(end_cycles - start_cycles);
        sqlite3_reset(query_stmt);
    }

    std::pair<int64_t, int64_t> cache_stats = cache_measurer.stop();

    // Query SQLite specific memory footprint before tearing down
    sqlite3_int64 sq_curr_mem = sqlite3_memory_used();
    sqlite3_int64 sq_high_mem = sqlite3_memory_highwater(0);
    int sq_cache_used = 0;
    int sq_cache_high = 0;
    sqlite3_db_status(db, SQLITE_DBSTATUS_CACHE_USED, &sq_cache_used, &sq_cache_high, 0);

    sqlite3_finalize(query_stmt);
    sqlite3_close(db);

    // Report final process footprint
    size_t post_rss = 0, post_peak = 0;
    get_memory_footprint(post_rss, post_peak);

    // 6. Report Cycle-Based Performance
    if (query_cycles.empty()) {
        std::cout << "\nNo queries performed." << std::endl;
        return 0;
    }

    uint64_t total_cycles = std::accumulate(query_cycles.begin(), query_cycles.end(), 0ULL);
    double avg_cycles_per_query = static_cast<double>(total_cycles) / query_cycles.size();
    double avg_cycles_per_vector = avg_cycles_per_query / NUM_VECTORS;

    // Find min and max for variance tracking
    uint64_t min_cycles = query_cycles[0];
    uint64_t max_cycles = query_cycles[0];
    for (auto c : query_cycles) {
        if (c < min_cycles) min_cycles = c;
        if (c > max_cycles) max_cycles = c;
    }

    std::cout << "\n--- Performance Results (TSC CPU Cycles) ---" << std::endl;
    std::cout << "Avg Cycles per Query  : " << std::fixed << std::setprecision(1) << avg_cycles_per_query << " cycles" << std::endl;
    std::cout << "Avg Cycles per Vector : " << std::fixed << std::setprecision(1) << avg_cycles_per_vector << " cycles  (per vector comparison)" << std::endl;
    std::cout << "Min Cycles per Query  : " << min_cycles << " cycles" << std::endl;
    std::cout << "Max Cycles per Query  : " << max_cycles << " cycles" << std::endl;
    std::cout << "Total Query Cycles    : " << total_cycles << " cycles" << std::endl;

    std::cout << "\n--- Hardware Performance Counters ---" << std::endl;
    if (cache_measurer.is_supported()) {
        int64_t misses = cache_stats.first;
        int64_t refs = cache_stats.second;
        double miss_rate = (refs > 0) ? (static_cast<double>(misses) / refs) * 100.0 : 0.0;
        double misses_per_query = static_cast<double>(misses) / NUM_QUERIES;

        std::cout << "Cache References      : " << refs << std::endl;
        std::cout << "Cache Misses          : " << misses << std::endl;
        std::cout << "Cache Miss Rate       : " << std::fixed << std::setprecision(3) << miss_rate << " %" << std::endl;
        std::cout << "Cache Misses / Query  : " << std::fixed << std::setprecision(1) << misses_per_query << std::endl;
    } else {
        std::cout << "Cache References      : N/A" << std::endl;
        std::cout << "Cache Misses          : N/A" << std::endl;
        std::cout << "Cache Miss Rate       : N/A" << std::endl;
#ifdef __linux__
        int paranoid = get_perf_event_paranoid();
        std::cout << "Note                  : perf_event_open restricted by kernel (perf_event_paranoid = " << paranoid << ")" << std::endl;
#else
        std::cout << "Note                  : perf_event_open not supported on this platform" << std::endl;
#endif
    }

    std::cout << "\n--- Memory Footprint ---" << std::endl;
    std::cout << "Process Initial RSS   : " << std::fixed << std::setprecision(2) << (static_cast<double>(init_rss) / 1024.0) << " MB" << std::endl;
    std::cout << "Process Current RSS   : " << std::fixed << std::setprecision(2) << (static_cast<double>(post_rss) / 1024.0) << " MB" << std::endl;
    std::cout << "Process Peak RSS (HWM): " << std::fixed << std::setprecision(2) << (static_cast<double>(post_peak) / 1024.0) << " MB" << std::endl;
    std::cout << "SQLite Current Memory : " << std::fixed << std::setprecision(2) << (static_cast<double>(sq_curr_mem) / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "SQLite Peak Memory    : " << std::fixed << std::setprecision(2) << (static_cast<double>(sq_high_mem) / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "SQLite DB Page Cache  : " << std::fixed << std::setprecision(2) << (static_cast<double>(sq_cache_used) / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "--------------------------------------------" << std::endl;

    return 0;
}