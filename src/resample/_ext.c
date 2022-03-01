#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/random/bitgen.h>

int rcont1(double*, int, const double*, int, const double*, int**, bitgen_t*);
int rcont2(double*, int, const double*, int, const double*, double*, bitgen_t*);

static PyObject* rcont_wrap(PyObject *self, PyObject *args) {
  int n = -1, method = -1;
  PyObject *r = NULL, *c = NULL, *rng = NULL, *bitgen = NULL, *cap = NULL;
  PyArrayObject *ra = NULL, *ca = NULL, *ma = NULL;
  int* work = 0; // pointer to workspace for rcont_naive, allocated by rcont_naive

  if(!PyArg_ParseTuple(args, "iO!O!iO",
    &n,
    &PyArray_Type, &r,
    &PyArray_Type, &c,
    &method,
    &rng))
    return NULL;

  ra = (PyArrayObject*)PyArray_FROM_OTF(r, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (!ra) goto fail;
  ca = (PyArrayObject*)PyArray_FROM_OTF(c, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
  if (!ca) goto fail;

  if (PyArray_NDIM(ra) != 1) {
    PyErr_SetString(PyExc_ValueError, "r must be 1d");
    goto fail;
  }

  if (PyArray_NDIM(ca) != 1) {
    PyErr_SetString(PyExc_ValueError, "c must be 1d");
    goto fail;
  }

  npy_intp nr = *PyArray_DIMS(ra);
  npy_intp nc = *PyArray_DIMS(ca);

  npy_intp m_shape[3] = {n, nr, nc};
  ma = (PyArrayObject*)PyArray_SimpleNew(3, m_shape, NPY_DOUBLE);

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
      status = rcont1(m_ptr, nr, r_ptr, nc, c_ptr, &work, rstate);
      break;
    case 1:
      status = rcont2(m_ptr, nr, r_ptr, nc, c_ptr, &ntot, rstate);
      break;
    default:
      PyErr_SetString(PyExc_ValueError, "method must be 0 or 1");
      goto fail;
    }
    switch(status) {
      case 1:
        PyErr_SetString(PyExc_RuntimeError, "null pointer encountered in memory access");
        goto fail;
      case 2:
        PyErr_SetString(PyExc_ValueError, "number of rows or columns < 2");
        goto fail;
      case 3:
        PyErr_SetString(PyExc_ValueError, "negative entries in row or col");
        goto fail;
      case 4:
        PyErr_SetString(PyExc_ValueError, "sum(row) != sum(col)");
        goto fail;
      case 5:
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

  return (PyObject*)ma;

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
