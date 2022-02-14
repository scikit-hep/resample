#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/random/bitgen.h>

int rcont(double*, int, const double*, int, const double*, double*, bitgen_t*);
int rcont_naive(double*, int, const double*, int, const double*, int**, bitgen_t*);

static PyObject* rcont_wrap(PyObject *self, PyObject *args) {
  int n = -1, method = -1;
  PyObject *r = NULL, *c = NULL, *rng = NULL,
    *ra = NULL, *ca = NULL, *ma = NULL,
    *bitgen = NULL, *cap = NULL;
  int* work = 0; // pointer to workspace for rcont_naive, allocated by rcont_naive

  if(!PyArg_ParseTuple(args, "iO!O!iO",
    &n,
    &PyArray_Type, &r,
    &PyArray_Type, &c,
    &method,
    &rng))
    return NULL;

  ra = PyArray_FROM_OTF(r, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (!ra) goto fail;
  ca = PyArray_FROM_OTF(c, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (!ca) goto fail;

  if (PyArray_NDIM(ra) != 1) {
    PyErr_SetString(PyExc_ValueError, "r must be 1d");
    goto fail;
  }

  if (PyArray_NDIM(ca) != 1) {
    PyErr_SetString(PyExc_ValueError, "c must be 1d");
    goto fail;
  }

  npy_intp* r_shape = PyArray_DIMS(ra);
  npy_intp* c_shape = PyArray_DIMS(ca);

  npy_intp m_shape[3] = {n, *r_shape, *c_shape};
  ma = PyArray_SimpleNew(3, m_shape, NPY_DOUBLE);

  bitgen = PyObject_GetAttrString(rng, "_bit_generator");
  if (!bitgen) goto fail;

  cap = PyObject_GetAttrString(bitgen, "capsule");
  if (!cap) goto fail;

  bitgen_t* rstate = (bitgen_t *)PyCapsule_GetPointer(cap, "BitGenerator");
  if (!rstate) goto fail;

  const double* r_ptr = (const double*)PyArray_DATA(ra);
  const double* c_ptr = (const double*)PyArray_DATA(ca);

  double ntot = 0; // indicator to run/skip expensive one-time checks, filled by rcont
  for (int i = 0; i < m_shape[0]; ++i) {
    int status = 0;
    double* m_ptr = (double*)PyArray_GETPTR3(ma, i, 0, 0);
    switch (method) {
    case 0:
      // Patefield's algorithm. Generally recommended for any table.
      status = rcont(m_ptr, r_shape[0], r_ptr, c_shape[0], c_ptr, &ntot, rstate);
      break;
    case 1:
      // Naive algorithm. Only useful to check implementation of Patefield's algorithm.
      status = rcont_naive(m_ptr, r_shape[0], r_ptr, c_shape[0], c_ptr, &work, rstate);
      break;
    default:
      PyErr_SetString(PyExc_ValueError, "method must be 0 or 1");
      goto fail;
    }
    switch(status) {
      case 1:
        // this should never happen and is only listed for completeness
        PyErr_SetString(PyExc_RuntimeError, "null pointer encountered");
        goto fail;
      case 2:
        PyErr_SetString(PyExc_ValueError, "number of rows < 2");
        goto fail;
      case 3:
        PyErr_SetString(PyExc_ValueError, "number of columns < 2");
        goto fail;
      case 4:
        PyErr_SetString(PyExc_ValueError, "negative entries in row or col");
        goto fail;
      case 5:
        PyErr_SetString(PyExc_ValueError, "sum(row) != sum(col)");
        goto fail;
      case 6:
        PyErr_SetString(PyExc_ValueError, "total number of entries <= 0");
        goto fail;
      default:
        break;
    }
  }

  // clean up
  Py_DECREF(ra);
  Py_DECREF(ca);
  Py_DECREF(bitgen);
  Py_DECREF(cap);
  if (work)
    free(work);

  return ma;

fail:
  Py_XDECREF(ra);
  Py_XDECREF(ca);
  Py_XDECREF(ma);
  Py_XDECREF(bitgen);
  Py_XDECREF(cap);
  if (work)
    free(work);

  return NULL;
}

static PyMethodDef methods[] = {
  {"rcont", rcont_wrap, METH_VARARGS},
  {NULL, NULL, 0, NULL}
};

static PyModuleDef mod = {
  PyModuleDef_HEAD_INIT,
  "resample._ext",
  NULL,
  -1,
  methods
};

PyMODINIT_FUNC
PyInit__ext(void) {
  PyObject* m = PyModule_Create(&mod);
  if (!m)
    return NULL;
  import_array();
  return m;
}
