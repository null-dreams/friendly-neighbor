#include <sqlite3ext.h>
SQLITE_EXTENSION_INIT1;

#include "distance.hpp"
#include <vector>
#include <string>
#include <cstring>
#include <cstdint>

static void l2_distance_sqlite(
    sqlite3_context* context,
    int argc,
    sqlite3_value** argv
) {
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

#ifdef _WIN32
__declspec(dllexport)
#endif
extern "C" int sqlite3_extension_init(
    sqlite3 *db,
    char **pzErrMsg,
    const sqlite3_api_routines *pApi
) {
    SQLITE_EXTENSION_INIT2(pApi);
    int rc = sqlite3_create_function_v2(
        db,
        "l2_distance",
        2,
        SQLITE_UTF8 | SQLITE_DETERMINISTIC,
        nullptr,
        l2_distance_sqlite,
        nullptr,
        nullptr,
        nullptr
    );
    return rc == SQLITE_OK ? SQLITE_OK : SQLITE_ERROR;
}