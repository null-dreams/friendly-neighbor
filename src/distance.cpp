#include "distance.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace vector_engine {

static float l2_distance_squared_scalar(const float* a, const float* b, size_t dimensions) {
    float sum = 0.0f;
    for (size_t i = 0; i < dimensions; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sum;
}

#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx")))
static float l2_distance_squared_avx(const float* a, const float* b, size_t dimensions) {
    float total_sum = 0.0f;
    size_t i = 0;
    __m256 sum_accumulator = _mm256_setzero_ps();

    for (; i + 7 < dimensions; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum_accumulator = _mm256_add_ps(sum_accumulator, _mm256_mul_ps(diff, diff));
    }

    alignas(32) float temp[8];
    _mm256_storeu_ps(temp, sum_accumulator);
    total_sum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];

    for (; i < dimensions; ++i) {
        float diff = a[i] - b[i];
        total_sum += diff * diff;
    }

    return total_sum;
}
#endif

#if defined(__aarch64__)
static float l2_distance_squared_neon(const float* a, const float* b, size_t dimensions) {
    float total_sum = 0.0f;
    size_t i = 0;
    float32x4_t sum_accumulator = vdupq_n_f32(0.0f);

    for (; i + 3 < dimensions; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(va, vb);
        sum_accumulator = vaddq_f32(sum_accumulator, vmulq_f32(diff, diff));
    }

    total_sum = vaddvq_f32(sum_accumulator);

    for (; i < dimensions; ++i) {
        float diff = a[i] - b[i];
        total_sum += diff * diff;
    }

    return total_sum;
}
#endif

float l2_distance_squared(const float* a, const float* b, size_t dimensions) {
#if defined(__x86_64__) || defined(_M_X64)
    #if defined(__GNUC__) || defined(__clang__)
    if (__builtin_cpu_supports("avx")) {
        return l2_distance_squared_avx(a, b, dimensions);
    }
    #endif
#elif defined(__aarch64__)
    return l2_distance_squared_neon(a, b, dimensions);
#endif

    return l2_distance_squared_scalar(a, b, dimensions);
}

} // namespace vector_engine