#ifndef _SPN_MMATH_H
#define _SPN_MMATH_H

// Matrix math header

// Trick to create a contiguous memory block matrix.
#define new_matrix(M, t, n, m) t** M = new t*[n]; if (n) { M[0] = new t[n*m]; for (size_t \
    _MMATH_H_TMP_i=1;_MMATH_H_TMP_i<(size_t)n;++_MMATH_H_TMP_i) M[_MMATH_H_TMP_i]=M[0]+\
    _MMATH_H_TMP_i*m; }

#define del_matrix(M, n) if (n) { delete[] M[0]; } delete[] M;

#endif
