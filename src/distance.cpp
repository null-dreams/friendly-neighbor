#include "distance.hpp"

namespace vector_engine {
    float l2_distance_squared(const float* a, const float* b, size_t dimensions) {
        float sum = 0.0f;
        for (size_t i = 0; i < dimensions; ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }
}