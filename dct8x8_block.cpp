
#include <immintrin.h>
#include "dct8x8_block.h"

void
dct8x8_block (float *in_8x8, float *out, int stride)
{
  double c1 = 0.980785;
  double c2 = 0.923880;
  double c3 = 0.831470;
  double c4 = 0.707107;
  double c5 = 0.555570;
  double c6 = 0.382683;
  double c7 = 0.195090;
    __m512d vc4 = _mm512_set1_pd(c4);
    __m512d vc2 = _mm512_set1_pd(c2);
    __m512d vc6 = _mm512_set1_pd(c6);
    __m512d vc1 = _mm512_set1_pd(c1);
    __m512d vc7 = _mm512_set1_pd(c7);
    __m512d vc3 = _mm512_set1_pd(c3);
    __m512d vc5 = _mm512_set1_pd(c5);

  double One_D_DCT_Row_8x8[8][8];

  __m512d f0 = _mm512_set_pd(in_8x8[ 0*stride + 0],in_8x8[ 1*stride + 0],in_8x8[ 2*stride + 0],in_8x8[ 3*stride + 0],in_8x8[ 4*stride + 0],in_8x8[ 5*stride + 0],in_8x8[ 6*stride + 0],in_8x8[ 7*stride + 0]);
    __m512d f1 = _mm512_set_pd(in_8x8[ 0*stride + 1],in_8x8[ 1*stride + 1],in_8x8[ 2*stride + 1],in_8x8[ 3*stride + 1],in_8x8[ 4*stride + 1],in_8x8[ 5*stride + 1],in_8x8[ 6*stride + 1],in_8x8[ 7*stride + 1]);
    __m512d f2 = _mm512_set_pd(in_8x8[ 0*stride + 2],in_8x8[ 1*stride + 2],in_8x8[ 2*stride + 2],in_8x8[ 3*stride + 2],in_8x8[ 4*stride + 2],in_8x8[ 5*stride + 2],in_8x8[ 6*stride + 2],in_8x8[ 7*stride + 2]);
    __m512d f3 = _mm512_set_pd(in_8x8[ 0*stride + 3],in_8x8[ 1*stride + 3],in_8x8[ 2*stride + 3],in_8x8[ 3*stride + 3],in_8x8[ 4*stride + 3],in_8x8[ 5*stride + 3],in_8x8[ 6*stride + 3],in_8x8[ 7*stride + 3]);
    __m512d f4 = _mm512_set_pd(in_8x8[ 0*stride + 4],in_8x8[ 1*stride + 4],in_8x8[ 2*stride + 4],in_8x8[ 3*stride + 4],in_8x8[ 4*stride + 4],in_8x8[ 5*stride + 4],in_8x8[ 6*stride + 4],in_8x8[ 7*stride + 4]);
    __m512d f5 = _mm512_set_pd(in_8x8[ 0*stride + 5],in_8x8[ 1*stride + 5],in_8x8[ 2*stride + 5],in_8x8[ 3*stride + 5],in_8x8[ 4*stride + 5],in_8x8[ 5*stride + 5],in_8x8[ 6*stride + 5],in_8x8[ 7*stride + 5]);
    __m512d f6 = _mm512_set_pd(in_8x8[ 0*stride + 6],in_8x8[ 1*stride + 6],in_8x8[ 2*stride + 6],in_8x8[ 3*stride + 6],in_8x8[ 4*stride + 6],in_8x8[ 5*stride + 6],in_8x8[ 6*stride + 6],in_8x8[ 7*stride + 6]);
    __m512d f7 = _mm512_set_pd(in_8x8[ 0*stride + 7],in_8x8[ 1*stride + 7],in_8x8[ 2*stride + 7],in_8x8[ 3*stride + 7],in_8x8[ 4*stride + 7],in_8x8[ 5*stride + 7],in_8x8[ 6*stride + 7],in_8x8[ 7*stride + 7]);

    __m512d i0 = _mm512_add_pd(f0, f7);
    __m512d i1 = _mm512_add_pd(f1, f6);
    __m512d i2 = _mm512_add_pd(f2, f5);
    __m512d i3 = _mm512_add_pd(f3, f4);
    __m512d i4 = _mm512_sub_pd(f3, f4);
    __m512d i5 = _mm512_sub_pd(f2, f5);
    __m512d i6 = _mm512_sub_pd(f1, f6);
    __m512d i7 = _mm512_sub_pd(f0, f7);

    __m512d j0 = _mm512_add_pd(i0, i3);
    __m512d j1 = _mm512_add_pd(i1, i2);
    __m512d j2 = _mm512_sub_pd(i1, i2);
    __m512d j3 = _mm512_sub_pd(i0, i3);
    __m512d j4 = i4;
    __m512d j5 = _mm512_mul_pd(_mm512_sub_pd(i6, i5), vc4) ;
    __m512d j6 = _mm512_mul_pd(_mm512_add_pd(i6, i5), vc4) ;
    __m512d j7 = i7;


    __m512d k0 = _mm512_mul_pd(_mm512_add_pd(j0, j1), vc4) ;
    __m512d k1 = _mm512_mul_pd(_mm512_sub_pd(j0, j1), vc4) ;
    __m512d k2 = _mm512_add_pd(_mm512_mul_pd(j2, vc6), _mm512_mul_pd(j3, vc2)) ;
    __m512d k3 = _mm512_sub_pd(_mm512_mul_pd(j3, vc6), _mm512_mul_pd(j2, vc2)) ;
    __m512d k4 = _mm512_add_pd(j4, j5);
    __m512d k5 = _mm512_sub_pd(j4, j5);
    __m512d k6 = _mm512_sub_pd(j7, j6);
    __m512d k7 = _mm512_add_pd(j7, j6);

    __m512d scale_factor = _mm512_set1_pd(0.5);
    __m512d F0 = _mm512_scalef_round_pd(k0, scale_factor, _MM_FROUND_CUR_DIRECTION);
    __m512d F1 = _mm512_scalef_round_pd(_mm512_add_pd(_mm512_mul_pd(k4, vc7), _mm512_mul_pd(k7, vc1)), scale_factor, _MM_FROUND_CUR_DIRECTION);
    __m512d F2 = _mm512_scalef_round_pd(k2, scale_factor, _MM_FROUND_CUR_DIRECTION);
    __m512d F3 = _mm512_scalef_round_pd(_mm512_sub_pd(_mm512_mul_pd(k6, vc3), _mm512_mul_pd(k5, vc5)), scale_factor, _MM_FROUND_CUR_DIRECTION);
    __m512d F4 = _mm512_scalef_round_pd(k1, scale_factor, _MM_FROUND_CUR_DIRECTION);
    __m512d F5 = _mm512_scalef_round_pd(_mm512_add_pd(_mm512_mul_pd(k5, vc3), _mm512_mul_pd(k6, vc5)), scale_factor, _MM_FROUND_CUR_DIRECTION);
    __m512d F6 = _mm512_scalef_round_pd(k3, scale_factor, _MM_FROUND_CUR_DIRECTION);
    __m512d F7 = _mm512_scalef_round_pd(_mm512_sub_pd(_mm512_mul_pd(k7, vc7), _mm512_mul_pd(k4, vc1)), scale_factor, _MM_FROUND_CUR_DIRECTION);


    double ff0[8], ff1[8], ff2[8], ff3[8], ff4[8], ff5[8], ff6[8], ff7[8];
    _mm512_storeu_pd(ff0, F0);
    _mm512_storeu_pd(ff1, F1);
    _mm512_storeu_pd(ff2, F2);
    _mm512_storeu_pd(ff3, F3);
    _mm512_storeu_pd(ff4, F4);
    _mm512_storeu_pd(ff5, F5);
    _mm512_storeu_pd(ff6, F6);
    _mm512_storeu_pd(ff7, F7);

    for (int row_number = 0; row_number < 8; row_number++) {
        // DCT coefficient assignment
        One_D_DCT_Row_8x8[row_number][0] = ff0[row_number];
        One_D_DCT_Row_8x8[row_number][1] = ff1[row_number];
        One_D_DCT_Row_8x8[row_number][2] = ff2[row_number];
        One_D_DCT_Row_8x8[row_number][3] = ff3[row_number];
        One_D_DCT_Row_8x8[row_number][4] = ff4[row_number];
        One_D_DCT_Row_8x8[row_number][5] = ff5[row_number];
        One_D_DCT_Row_8x8[row_number][6] = ff6[row_number];
        One_D_DCT_Row_8x8[row_number][7] = ff7[row_number];
    }




  f0 = _mm512_loadu_pd(One_D_DCT_Row_8x8[0]);
    f1 = _mm512_loadu_pd(One_D_DCT_Row_8x8[1]);
    f2 = _mm512_loadu_pd(One_D_DCT_Row_8x8[2]);
    f3 = _mm512_loadu_pd(One_D_DCT_Row_8x8[3]);
    f4 = _mm512_loadu_pd(One_D_DCT_Row_8x8[4]);
    f5 = _mm512_loadu_pd(One_D_DCT_Row_8x8[5]);
    f6 = _mm512_loadu_pd(One_D_DCT_Row_8x8[6]);
    f7 = _mm512_loadu_pd(One_D_DCT_Row_8x8[7]);

    i0 = _mm512_add_pd(f0, f7);
    i1 = _mm512_add_pd(f1, f6);
    i2 = _mm512_add_pd(f2, f5);
    i3 = _mm512_add_pd(f3, f4);
    i4 = _mm512_sub_pd(f3, f4);
    i5 = _mm512_sub_pd(f2, f5);
    i6 = _mm512_sub_pd(f1, f6);
    i7 = _mm512_sub_pd(f0, f7);

    j0 = _mm512_add_pd(i0, i3);
    j1 = _mm512_add_pd(i1, i2);
    j2 = _mm512_sub_pd(i1, i2);
    j3 = _mm512_sub_pd(i0, i3);
     j4 = i4;
     j5 = _mm512_sub_pd(i6, i5);
     j6 = _mm512_add_pd(i6, i5);
     j7 = i7;


     k0 = _mm512_mul_pd(_mm512_add_pd(j0, j1), vc4);
     k1 = _mm512_mul_pd(_mm512_sub_pd(j0, j1), vc4);
     k2 = _mm512_add_pd(_mm512_mul_pd(j2, vc6), _mm512_mul_pd(j3, vc2));
     k3 = _mm512_sub_pd(_mm512_mul_pd(j3, vc6), _mm512_mul_pd(j2, vc2));
     k4 = _mm512_add_pd(j4, j5);
     k5 = _mm512_sub_pd(j4, j5);
     k6 = _mm512_sub_pd(j7, j6);
     k7 = _mm512_add_pd(j7, j6);


     F0 = _mm512_scalef_round_pd(k0, scale_factor, _MM_FROUND_CUR_DIRECTION);
     F1 = _mm512_scalef_round_pd(_mm512_add_pd(_mm512_mul_pd(k4, vc7), _mm512_mul_pd(k7, vc1)), scale_factor, _MM_FROUND_CUR_DIRECTION);
     F2 = _mm512_scalef_round_pd(k2, scale_factor, _MM_FROUND_CUR_DIRECTION);
     F3 = _mm512_scalef_round_pd(_mm512_sub_pd(_mm512_mul_pd(k6, vc3), _mm512_mul_pd(k5, vc5)), scale_factor, _MM_FROUND_CUR_DIRECTION);
     F4 = _mm512_scalef_round_pd(k1, scale_factor, _MM_FROUND_CUR_DIRECTION);
     F5 = _mm512_scalef_round_pd(_mm512_add_pd(_mm512_mul_pd(k5, vc3), _mm512_mul_pd(k6, vc5)), scale_factor, _MM_FROUND_CUR_DIRECTION);
     F6 = _mm512_scalef_round_pd(k3, scale_factor, _MM_FROUND_CUR_DIRECTION);
     F7 = _mm512_scalef_round_pd(_mm512_sub_pd(_mm512_mul_pd(k7, vc7), _mm512_mul_pd(k4, vc1)), scale_factor, _MM_FROUND_CUR_DIRECTION);

    _mm256_storeu_ps( &out[0 * stride], _mm512_cvtpd_ps(F0));
    _mm256_storeu_ps( &out[1 * stride], _mm512_cvtpd_ps(F1));
    _mm256_storeu_ps( &out[2 * stride], _mm512_cvtpd_ps(F2));
    _mm256_storeu_ps( &out[3 * stride], _mm512_cvtpd_ps(F3));
    _mm256_storeu_ps( &out[4 * stride], _mm512_cvtpd_ps(F4));
    _mm256_storeu_ps( &out[5 * stride], _mm512_cvtpd_ps(F5));
    _mm256_storeu_ps( &out[6 * stride], _mm512_cvtpd_ps(F6));
    _mm256_storeu_ps( &out[7 * stride], _mm512_cvtpd_ps(F7));

}
