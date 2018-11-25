#include <stdint.h>
#include <inttypes.h>
#include "polynom_mul.h"
#include <stdio.h>
#include <wmmintrin.h>
#include <time.h>
#include <math.h>

/**
 * @brief make_time_test Тестирует время переменожения многочленов \f$ x\f$ и \f$ y\f$ по модулю \f$ G\f$ используя для этого рекурсивный алгоритм и комманды intel
 * @param xar \f$ x \f$
 * @param yar \f$ y \f$
 * @param gar \f$ G - x^128 \f$
 * @param qplus многочлен такой, что \f$ deg(x^256 - G^128 * (qplus + x^128)) < 128  \f$
 */
void make_time_test(const uint64_t xar[2], const uint64_t yar[2], const uint64_t gar[2], __m128i qplus);

/**
 * @brief get_int_len длину числа value в десятичной системе счисления
 * @param value value
 * @return длина value
 */
int get_int_len (size_t value);

int main()
{
    uint64_t xar[2] = {1, 1};
    uint64_t yar[2] = {1, 3};
    uint64_t gar[2] = {135, 0};
    uint64_t resar[2];
    printf("Input powers of x where coefficient is 1 (to end input -- input -1):\n");
    scan_polynom_ar(xar);
    printf("Input powers of y where coefficient is 1 (to end input -- input -1):\n");
    scan_polynom_ar(yar);

    mul_pol_recursive128(resar, xar, yar, gar);
    print_polynom_ar(xar);
    printf(" * ");
    print_polynom_ar(yar);
    printf("\n =\nresult by recursive:\t");

    print_polynom_ar(resar);
    printf("\n");

    __m128i x, y, z, g;
    x = _mm_set_epi64x(xar[1], xar[0]);
    y = _mm_set_epi64x(yar[1], yar[0]);
    g = _mm_set_epi64x(gar[1], gar[0]);

    z = mul_pol_intel128(x, y, g, g);
    uint64_t* zu = (uint64_t*) &z;
    printf(" =\nresult by intel:\t");
    print_polynom_ar(zu);
    printf("\n mod x^128 + ");
    print_polynom_ar(gar);
    printf("\n");

    make_time_test(xar, yar ,gar, g);
}


void make_time_test(const uint64_t xar[2], const uint64_t yar[2], const uint64_t gar[2], __m128i qplus)
{
    clock_t start, end;
    double total;
    const size_t n = 10000000;

    __m128i x, y, z, g;
    uint64_t resar[2];

    x = _mm_set_epi64x(xar[1], xar[0]);
    y = _mm_set_epi64x(yar[1], yar[0]);
    g = _mm_set_epi64x(gar[1], gar[0]);

    start = clock();
    for(size_t i = 0; i != n; ++i) {
        mul_pol_recursive128(resar, xar, yar, gar);
    }
    end = clock();
    total = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Recursive result: \t%f secs for 10^%d muls\n", total, get_int_len(n)-1);

    start = clock();
    for(size_t i = 0; i != n; ++i) {
        z = mul_pol_intel128(x, y, g, qplus);
    }
    end = clock();
    total = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Intel result: \t\t%f secs for 10^%d muls\n", total, get_int_len(n)-1);
}

int get_int_len (size_t value){
  int l=1;
  while(value>9){ l++; value/=10; }
  return l;
}
