import numpy as np

cimport numpy as np
from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_IsValid
from libc.stdlib cimport free

np.import_array()

cdef extern from "rcont.h":
    int rcont_check(double* n, const double* m, int nr, const double* r, int nc, const double* c)
    int rcont1(double*, int, const double*, int, const double*, int**, void* rstate);
    int rcont2(double*, int, const double*, int, const double*, double*, void* rstate);


def rcont(int n, np.ndarray[double, ndim=1] r, np.ndarray[double, ndim=1] c, int method, rng):
    cdef int m_shape[3]
    m_shape[0] = n
    m_shape[1] = r.shape[0]
    m_shape[2] = c.shape[0]
    cdef np.ndarray[double, ndim=3] ma = np.empty(m_shape, dtype=np.double)
    cdef int status = 0
    cdef int* work = NULL;
    cdef double ntot = 0;

    cap = rng._bit_generator.capsule
    cdef const char* cap_name = "BitGenerator"
    if not PyCapsule_IsValid(cap, cap_name):
        raise ValueError("invalid pointer to generator")
    cdef void* rstate = PyCapsule_GetPointer(cap, cap_name)

    for i in range(n):
        if method == 0:
            status = rcont1(&ma[i,0,0], m_shape[1], &r[0], m_shape[2], &c[0], &work, rstate)
        elif method == 1:
            print(ntot, i, r, c)
            status = rcont2(&ma[i,0,0], m_shape[1], &r[0], m_shape[2], &c[0], &ntot, rstate)
        else:
            raise ValueError("method must be 0 or 1")
        if status != 0:
            break

    if work:
        free(work)

    if status == 1:
        raise RuntimeError("null pointer encountered in memory access")
    elif status == 2:
        raise ValueError("number of rows or columns < 2")
    elif status == 3:
        raise ValueError("negative entries in row or col")
    elif status == 4:
        raise ValueError("sum(row) != sum(col)")
    elif status == 5:
          raise ValueError("total number of entries <= 0")

    return ma
