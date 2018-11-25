#include <wmmintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <emmintrin.h>

/**
 * @brief shift_left_by_count побитовый сдвиг коэффициентов полинома. РЕЗУЛЬТАТ И ТО ЧТО СДВИГАЕМ -- ДОЛЖНЫ БЫТЬ РАЗНЫМИ (в плане объектов)
 * @param r результат
 * @param c полином, который сдвигаем
 * @param n значение, на которое сдвигаем
 */
void shift_left_by_count(uint64_t r[2], const uint64_t c[2], size_t n)
{
    r[0] = c[0] << n;
    r[1] = (c[1] << n) | (c[0] >> (64 - n));
}

/**
 * @brief get_bit_n возвращает коэффицент полинома \f$ x \f$ при заданной степени n
 * @param x \f$ x \f$
 * @param \f$ n \f$ степень
 * @return коэффициент \f$ x \f$ при степени \f$ n \f$
 */
int get_bit_n (const uint64_t x[2], size_t n) {
    if (n < 64) {
        return (x[0] >> n) & 1;
    } else {
        return (x[1] >> (n-64)) & 1;
    }
}

/**
 * @brief set_bit_n устанавливает значение коэффициента при заданной степени
 * @param res
 * @param x
 * @param n
 */
void set_bit_n (uint64_t res[2], const uint64_t x[2], size_t n) {
    if (n < 64) {
        res[0] = x[0] | (1ull << n);
    } else {
        res[1] = x[1] | (1ull << (n-64));
    }
}


void mul_pol_recursive128(uint64_t res[2], const uint64_t a[2], const uint64_t b[2], const uint64_t g[2])
{
    // используется формула \f$ \sum_{i=0}^127 a_i c_i(x) \f, где \f$ c_i(x) = b_i * c_{i-1}(x) + (g(x) * c(i-1, 127)) \f$
    // \f$ c_0 = a \f$
    res[0] = 0;
    res[1] = 0;
    uint64_t c[2], tmp[2];

    for (size_t i = 0; i != 64 * 2; ++i) {
        if (i == 0) {
            tmp[0] = b[0];
            tmp[1] = b[1];
        } else {
             shift_left_by_count(tmp, c, 1);
             if ((c[1] >> 63) & 1) {
                 tmp[0] ^= g[0];
                 tmp[1] ^= g[1];
             }

        }
        if (get_bit_n(a, i)) {
            res[0] ^= tmp[0];
            res[1] ^= tmp[1];
        }
        c[0] = tmp[0];
        c[1] = tmp[1];
    }
}


void print_polynom_ar(const uint64_t polynom[2])
{
    int first_was = 0;
    for (size_t i = 64 *2; i != 0 ; --i) {
        if (get_bit_n(polynom, i-1)) {
            if (first_was) {
                printf(" + ");
            }
            if (i == 1) {
                printf("1");
            } else{
                printf("x^%zu", i-1);
            }

            first_was = 1;
        }
    }
    if (! first_was) {
        printf("0");
    }
}

void scan_polynom_ar(uint64_t polynom[2])
{
    polynom[0] = 0;
    polynom[1] = 0;
    int pos = -1;
    while (1) {
        scanf("%d", &pos);
        if (pos < 0) {
            return;
        }
        set_bit_n(polynom, polynom, (size_t) pos);
    }
}

/**
 * @brief mod_pol возвращает \f$ c mod G\f$
 * @param c \f$ c \f$
 * @param g \f$ G - x^128\f$
 * @param qplus многочлен такой, что \f$ deg(x^256 - G^128 * (qplus + x^128)) < 128  \f$
 * @return \f$ c mod G\f$
 */
__m128i mod_pol(__m128i c, __m128i g, __m128i qplus)
{
    __m128i result;
    __m128i lu, ul, ll, uu;
    lu = _mm_clmulepi64_si128(c, qplus, 0x10);
    ul = _mm_clmulepi64_si128(c, qplus, 0x01);
    uu = _mm_clmulepi64_si128(c, qplus, 0x11);

    result = _mm_xor_si128(uu, _mm_srli_si128(ul, 8));
    result = _mm_xor_si128(result, _mm_srli_si128(lu, 8)); // на этом моменте result = M_128(с(x) * (qplus))
    result = _mm_xor_si128(result, c); // на этом моменте result = M_128(с(x) * (qplus + x^128))

    ll = _mm_clmulepi64_si128(result, g, 0x00);
    lu = _mm_clmulepi64_si128(result, g, 0x10);
    ul = _mm_clmulepi64_si128(result, g, 0x01);

    result = _mm_xor_si128(ll, _mm_slli_si128(lu, 8));
    result = _mm_xor_si128(result, _mm_slli_si128(ul, 8)); // result = L_128(g(x) * M_128(с(x) * (qplus)))

    return result;
}


__m128i mul_pol_intel128(__m128i x, __m128i y, __m128i g, __m128i qplus)
{
    __m128i less;
    __m128i big;
    __m128i ll = _mm_clmulepi64_si128(x, y, 0x00);
    __m128i lu = _mm_clmulepi64_si128(x, y, 0x10);
    __m128i ul = _mm_clmulepi64_si128(x, y, 0x01);
    __m128i uu = _mm_clmulepi64_si128(x, y, 0x11);

    less = _mm_xor_si128(ll, _mm_slli_si128(lu, 8));
    less = _mm_xor_si128(less, _mm_slli_si128(ul, 8)); // получаем младшие коэффиценты многочлена

    big = _mm_xor_si128(uu, _mm_srli_si128(ul, 8));
    big = _mm_xor_si128(big, _mm_srli_si128(lu, 8)); // получаем старшие коэффиценты многочлена

    big = mod_pol(big, g, qplus); // узнаем значение старших коэффиентов после приведения по модулю G

    return _mm_xor_si128(big, less); // складываем старшую (после приведения) и младшую часть коэффиентов многочлена
}

