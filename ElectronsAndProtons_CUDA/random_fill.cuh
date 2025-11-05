#pragma once
#include <random>
#include <type_traits>

template <typename T>
void fillRandom(T* array, size_t size, T min, T max)
{
    static_assert(std::is_arithmetic<T>::value, "T must be numeric");

    // Random engine — static to avoid reseeding every call
    static std::random_device rd;
    static std::mt19937 gen(rd());

    if constexpr (std::is_integral<T>::value) {
        std::uniform_int_distribution<T> dist(min, max);
        for (size_t i = 0; i < size; ++i) {
            array[i] = dist(gen);
        }
    }
    else { // floating point
        std::uniform_real_distribution<T> dist(min, max);
        for (size_t i = 0; i < size; ++i) {
            array[i] = dist(gen);
        }
    }
}