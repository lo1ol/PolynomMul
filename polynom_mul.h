#pragma once

#include <wmmintrin.h>
#include <stdint.h>

/**
 * @brief mul_pol_intel128 Умножает два многочлена \f$ x \f$ и \f$ y \f$ по заданному модулю \f$ G \f$ используя команды процессора intel
 * @param x первый многочлен
 * @param y второй многочлен
 * @param g \f$ G - x^128 \f$
 * @param qplus многочлен такой, что \f$ deg(x^256 - G^128 * (qplus + x^128)) < 128  \f$
 * @return x * y mod G
 */
__m128i mul_pol_intel128(__m128i x, __m128i y, __m128i g, __m128i qplus);

/**
 * @brief mul_pol_recursive128 Умножает два многочлена \f$ x \f$ и \f$ y \f$ по заданному модулю \f$ G \f$ используя рекурсивный алгоритм
 * @param result x * y mod G
 * @param x первый многочлен
 * @param y второй многочлен
 * @param g \f$ G - x^128 \f$
 */
void mul_pol_recursive128(uint64_t result[2], const uint64_t x[2], const uint64_t y[2], const uint64_t g[2]);

/**
 * @brief print_polynom_ar печатает полином в стандартный поток
 * @param polynom полином, который выводим
 */
void print_polynom_ar(const uint64_t polynom[2]);

/**
 * @brief print_polynom_ar считывает степени многочлена при которых коэффицент равен 1 из стандартного потока
 * @param polynom полином,  который считываем
 */
void scan_polynom_ar(uint64_t polynom[2]);
